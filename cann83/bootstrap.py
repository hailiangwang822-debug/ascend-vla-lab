"""ModelArts node entry: OBS assets -> isolated environment -> durable 64-card SFT."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import threading
import traceback
import uuid

from obs_state import Store, atomic_json, digest, restore_node, safe_relative, validate_remote, configured_root
from runtime_contract import check_base_runtime, validate_node_entry, clean_child_environment

ROOT = Path(__file__).resolve().parents[1]


def specification(env):
    obs_root = configured_root(env)
    nodes = int(env.get('MA_NUM_HOSTS', '8'))
    node = int(env.get('VC_TASK_INDEX', '0'))
    if nodes != 8 or not 0 <= node < 8:
        raise ValueError('This package requires 8 nodes x 8 devices = 64 cards')
    run = env.get('RUN_NAME', 'starvla_spatial_64p_001')
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', run) or '..' in run:
        raise ValueError('Invalid RUN_NAME')
    resume = env.get('RESUME', 'none')
    if resume not in ('none', 'latest') and not resume.isdigit():
        raise ValueError('RESUME must be none, latest or a numeric saved step')
    steps, interval = int(env.get('TRAIN_STEPS', '1000')), int(env.get('SAVE_INTERVAL', '100'))
    if steps < 1 or interval < 1:
        raise ValueError('TRAIN_STEPS and SAVE_INTERVAL must be positive')
    asset_remote = validate_remote(env.get('ASSETS_OBS', obs_root + '/assets/starvla-qwen25-spatial-v1'), root=obs_root)
    cache = Path(env.get('STARVLA_CACHE', '/cache/data/starvla-64p-v1')) / f'node-{node:03d}'
    return {'obs_root': obs_root, 'nodes': nodes, 'node': node, 'world_size': 64, 'run': run, 'resume': resume,
            'steps': steps, 'save_interval': interval, 'asset_remote': asset_remote,
            'run_remote': obs_root + '/runs/' + run, 'cache': str(cache),
            'startup': 'cd "${MA_JOB_DIR}/A" && bash run_cloud.sh train'}


def secondary_entry(env):
    # Compatibility name. The supported platform contract is ONE entry per
    # node, just like the known-good VLM job. RTF/device metadata is not proof
    # of a per-card entry and must never silently skip an entire worker node.
    validate_node_entry(env)
    return False


def verified_copy(store, remote, local, size, sha):
    local = Path(local)
    if local.is_file() and local.stat().st_size == size and digest(local) == sha:
        print('ASSET_VERIFIED_EXISTING', local.name, flush=True)
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    part = local.with_name(local.name + '.download')
    store.copy(remote, part)
    if part.stat().st_size != size or digest(part) != sha:
        raise ValueError('Asset size/SHA256 mismatch: ' + remote)
    part.replace(local)
    print('ASSET_DOWNLOADED', local.name, flush=True)


def extract_verified(path, destination):
    # Both archives contain only plain relative files; reject links and traversal.
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            safe_relative(member.name)
            if not member.isfile():
                raise ValueError('Unexpected non-file archive member: ' + member.name)
        tar.extractall(destination, filter='data')


def prepare_assets(store, spec):
    cache, remote = Path(spec['cache']), spec['asset_remote']
    assets = cache / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((ROOT / 'cann83/bundle-manifest.json').read_text())
    for row in bundle['archives']:
        path = cache / 'downloads' / row['filename']
        verified_copy(store, remote + '/' + row['filename'], path, row['bytes'], row['sha256'])
        marker = assets / ('.' + row['filename'] + '.sha256')
        if not marker.is_file() or marker.read_text() != row['sha256']:
            extract_verified(path, assets)
            marker.write_text(row['sha256'])
    manifest = json.loads((ROOT / 'cann83/assets-manifest.json').read_text())
    def download(row):
        verified_copy(store, remote + '/' + row['path'], assets / row['path'], row['bytes'], row['hash'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, [row for row in manifest['files'] if row['group'] == 'model']))
    wheelhouse = assets / 'wheelhouse'
    wheels = json.loads((wheelhouse / 'manifest.json').read_text())
    if wheels['errors']:
        raise ValueError('Wheel bundle incomplete')
    for row in wheels['files']:
        p = wheelhouse / row['filename']
        if not p.is_file() or p.stat().st_size != row['bytes'] or digest(p) != row['sha256']:
            raise ValueError('Wheel missing/corrupt: ' + str(p))
    return assets


def upload_logs(store, local, remote):
    for base in (local, local / 'run_config'):
        if base.is_dir():
            for path in base.iterdir():
                if path.is_file() and path.suffix in {'.log', '.json', '.jsonl', '.yaml', '.txt'}:
                    store.copy(path, remote + '/' + path.relative_to(local).as_posix())


def execute(spec):
    for name in ('MA_NUM_HOSTS', 'VC_TASK_INDEX', 'VC_WORKER_HOSTS'):
        if not os.environ.get(name):
            raise ValueError('Missing ModelArts environment variable: ' + name)
    from launch import validate_topology
    validate_topology(spec['nodes'], spec['node'], 8, os.environ)
    check_base_runtime(ROOT)
    cache = Path(spec['cache'])
    cache.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
    output_root = cache / 'outputs' / attempt
    local = output_root / spec['run']
    local.mkdir(parents=True, exist_ok=True)
    remote_logs = spec['run_remote'] + f'/logs/node-{spec["node"]:03d}/' + attempt
    store = Store()
    stopped = threading.Event()

    def log_worker():
        while not stopped.wait(60):
            try:
                upload_logs(store, local, remote_logs)
            except Exception as e:
                print('LOG_UPLOAD_RETRY_NEXT_MINUTE', str(e), flush=True)

    atomic_json(local / 'bootstrap.json', {**spec, 'attempt': attempt, 'bootstrap_python': sys.executable})
    # Check write permissions on every node before large transfers or NPU initialization.
    store.copy(local / 'bootstrap.json', remote_logs + '/bootstrap.json')
    monitor = threading.Thread(target=log_worker, daemon=True)
    monitor.start()
    child = None
    try:
        # Rank0 owns the run reservation. Other nodes wait at distributed startup;
        # testing existence on all nodes would race with rank0's creation.
        if spec['node'] == 0:
            marker = spec['run_remote'] + '/_RUN.json'
            if store.exists(marker):
                if spec['resume'] == 'none':
                    raise ValueError('RUN_NAME already exists. Resume it explicitly or use a new RUN_NAME')
            elif spec['resume'] != 'none':
                raise ValueError('Unknown resume run')
            else:
                store.write(marker, {**spec, 'attempt': attempt}, local / '_RUN.json')
        assets = prepare_assets(store, spec)
        env = clean_child_environment(os.environ)
        env.update(STARVLA_OBS_ROOT=spec['obs_root'], MODEL_DIR=str(assets / 'models/Qwen2.5-VL-3B-Instruct'),
                   DATA_ROOT=str(assets / 'datasets'), OUTPUT_ROOT=str(output_root),
                   RUN_NAME=spec['run'], TRAIN_STEPS=str(spec['steps']), SAVE_INTERVAL=str(spec['save_interval']),
                   NNODES='8', NODE_RANK=str(spec['node']), NPROC_PER_NODE='8', GLOBAL_BATCH='64',
                   MASTER_ADDR=os.environ['VC_WORKER_HOSTS'].split(',')[0].strip(),
                   MICRO_BATCH='1', DATA_WORKERS='0', WARMUP_STEPS=str(max(2, spec['steps'] // 20)),
                   STARVLA_BOOTSTRAP_PYTHON=sys.executable, STARVLA_OBS_RUN=spec['run_remote'],
                   STARVLA_RUNTIME=str(cache / 'runtime' / attempt),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                   TOKENIZERS_PARALLELISM='false', NO_ALBUMENTATIONS_UPDATE='1',
                   ALBUMENTATIONS_OFFLINE='1', PYTHONUNBUFFERED='1',
                   PYTHONNOUSERSITE='1',
                   HCCL_WHITELIST_DISABLE='1', HCCL_CONNECT_TIMEOUT='7200', HCCL_EXEC_TIMEOUT='14400',
                   HCCL_EVENT_TIMEOUT='14520', ACL_DEVICE_SYNC_TIMEOUT='14700',
                   ASCEND_SLOG_PRINT_TO_STDOUT='0', ASCEND_GLOBAL_LOG_LEVEL='3')
        if spec['resume'] != 'none':
            resume_dir = cache / 'restores' / attempt
            restored = restore_node(store, spec['run_remote'], spec['resume'], spec['node'], 8,
                                    resume_dir, cache / 'receipts' / attempt)
            if restored['step'] > spec['steps']:
                raise ValueError('Saved step exceeds TRAIN_STEPS')
            env['STARVLA_RESUME_DIR'] = str(resume_dir)
        venv = cache / 'venv'
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(venv)], check=True)
        python = venv / 'bin/python'
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps',
                        '--disable-pip-version-check',
                        '--find-links', str(assets / 'wheelhouse'), '-r', str(ROOT / 'cann83/requirements.lock')],
                       env=env, check=True)
        with (local / 'console.log').open('w', buffering=1) as log:
            child = subprocess.Popen([str(python), '-u', str(ROOT / 'cann83/launch.py')],
                                     cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, start_new_session=True)
            def forward_signal(signum, frame):
                atomic_json(local / 'termination.json', {'signal': signum, 'last_committed_state': 'see OBS LATEST.json'})
                if child.poll() is None:
                    os.killpg(child.pid, signum)
            signal.signal(signal.SIGTERM, forward_signal)
            signal.signal(signal.SIGINT, forward_signal)
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
            code = child.wait()
        atomic_json(local / 'exit.json', {'exit_code': code})
        if code:
            raise RuntimeError('Training process exited with status ' + str(code))
    except BaseException:
        (local / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        stopped.set()
        monitor.join(timeout=15)
        try:
            upload_logs(store, local, remote_logs)
        except Exception:
            if not active_error:
                raise
            print('FINAL_LOG_UPLOAD_FAILED (original error preserved)', traceback.format_exc(), flush=True)


def main():
    import fcntl
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preview', 'train'])
    args = p.parse_args()
    spec = specification(os.environ)
    print(json.dumps(spec, indent=2), flush=True)
    names = ('MA_RUN_METHOD', 'MA_NUM_HOSTS', 'VC_TASK_INDEX', 'VC_WORKER_HOSTS',
             'MA_LOCAL_TASK_INDEX', 'RANK_TABLE_FILE', 'ASCEND_DEVICE_ID',
             'LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'ASCEND_RT_VISIBLE_DEVICES')
    print('NODE_ENTRY', json.dumps({k: os.environ[k] for k in names if k in os.environ}), flush=True)
    print('BOOTSTRAP_PYTHON', sys.executable, sys.version, flush=True)
    if args.mode == 'preview':
        return
    secondary_entry(os.environ)
    cache = Path(spec['cache'])
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / '.launcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another launcher is active on this node. Use one custom bash entry per node; do not wrap it in a per-card launcher.') from exc
        execute(spec)


if __name__ == '__main__':
    main()
