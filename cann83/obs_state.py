"""ModelArts MoXing transfer with committed, per-node checkpoint inventories."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from obs_backend import select_backend

def configured_root(env=None):
    env = os.environ if env is None else env
    value = env.get('STARVLA_OBS_ROOT', '').rstrip('/')
    if not re.fullmatch(r's3://[a-z0-9][a-z0-9.-]*/[A-Za-z0-9_./-]+', value):
        raise ValueError('Set STARVLA_OBS_ROOT=s3://your-bucket/your-prefix/vla')
    segments = value[5:].split('/')
    if any(part in ('', '.', '..') for part in segments):
        raise ValueError('STARVLA_OBS_ROOT has an invalid path segment')
    return value


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '\\' in value or not p.parts:
        raise ValueError('Invalid relative path')
    return p


def validate_remote(value, root=None):
    prefix = (configured_root() if root is None else configured_root({"STARVLA_OBS_ROOT": root})) + "/"
    if not value.startswith(prefix) or any(c in value for c in ('?', '#', '\\', '%')):
        raise ValueError('OBS path must stay under ' + prefix)
    safe_relative(value[len(prefix):].rstrip('/'))
    return value.rstrip('/')


def retry(function, *args):
    for attempt in range(4):
        try:
            return function(*args)
        except Exception:
            if attempt == 3:
                raise
            print(f'OBS transfer retry {attempt + 2}/4', flush=True)
            time.sleep(min(2 ** attempt, 8))


class Store:
    def __init__(self, backend=None):
        if backend is None:
            backend = select_backend()
        self.backend = backend

    def exists(self, path):
        return retry(self.backend.exists, path)

    def copy(self, src, dst):
        retry(self.backend.copy, str(src), str(dst))

    def read(self, remote, temp):
        temp = Path(temp)
        temp.parent.mkdir(parents=True, exist_ok=True)
        self.copy(remote, temp)
        return json.loads(temp.read_text(encoding='utf-8'))

    def write(self, remote, value, temp):
        atomic_json(temp, value)
        self.copy(temp, remote)


def inventory(folder):
    rows = []
    for p in sorted(Path(folder).rglob('*')):
        if p.is_symlink():
            raise ValueError('Checkpoint cannot contain symlinks')
        if p.is_file() and not p.name.startswith(('_NODE_COMPLETE', '_OBS_COMMITTED', '_RESTORED', '_COMPLETE')):
            rows.append({'path': p.relative_to(folder).as_posix(),
                         'bytes': p.stat().st_size, 'sha256': digest(p)})
    return rows


def validate_node(meta, node, nodes, step, fingerprint):
    if any(meta.get(k) != v for k, v in {'node': node, 'nodes': nodes, 'step': step,
                                        'fingerprint': fingerprint}.items()):
        raise ValueError('Checkpoint node identity mismatch')
    paths = {r['path'] for r in meta['files']}
    for rank in range(node * 8, (node + 1) * 8):
        if f'random_states_{rank}.pkl' not in paths:
            raise ValueError('Missing RNG state for rank ' + str(rank))
        if f'rank_state_{rank}.pt' not in paths:
            raise ValueError('Missing explicit rank/NPU RNG state for rank ' + str(rank))
        expected = f'pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
        if expected not in paths:
            raise ValueError('Missing ZeRO optimizer/master weights: ' + expected)
    required = {'progress.json', 'custom_checkpoint_0.pkl', 'config.full.yaml', 'dataset_statistics.json'}
    if node == 0:
        required.add('pytorch_model/mp_rank_00_model_states.pt')
    if not required <= paths:
        raise ValueError('Checkpoint incomplete: ' + str(required - paths))
    for row in meta['files']:
        safe_relative(row['path'])
        if row['bytes'] <= 0 or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('Invalid file inventory')


def upload_node(store, local, remote, node, nodes):
    local, remote = Path(local), validate_remote(remote)
    if store.exists(remote + '/_COMPLETE.json'):
        raise ValueError('Refusing to overwrite an already committed checkpoint')
    progress = json.loads((local / 'progress.json').read_text())
    meta = {'schema_version': 1, 'node': node, 'nodes': nodes,
            'step': progress['completed_steps'], 'fingerprint': progress['fingerprint'],
            'files': inventory(local)}
    validate_node(meta, node, nodes, meta['step'], meta['fingerprint'])
    destination = f'{remote}/node-{node:03d}'
    for row in meta['files']:
        p = local / row['path']
        store.copy(p, destination + '/' + row['path'])
    if inventory(local) != meta['files']:
        raise RuntimeError('Checkpoint changed during upload; no completion marker published')
    store.write(destination + '/_NODE_COMPLETE.json', meta, local / '_NODE_COMPLETE.json')
    print('OBS_NODE_COMPLETE', meta['step'], node, flush=True)
    return meta


def commit_step(store, remote, nodes, step, fingerprint, temp):
    remote = validate_remote(remote)
    temp = Path(temp)
    receipts = []
    for node in range(nodes):
        marker = f'{remote}/node-{node:03d}/_NODE_COMPLETE.json'
        meta = store.read(marker, temp / f'node-{node:03d}.json')
        validate_node(meta, node, nodes, step, fingerprint)
        receipts.append({'node': node, 'manifest_sha256': digest(temp / f'node-{node:03d}.json')})
    result = {'schema_version': 1, 'step': step, 'nodes': nodes, 'world_size': nodes * 8,
              'fingerprint': fingerprint, 'node_manifests': receipts,
              'state': 'model+fp32-master+optimizer+scheduler+rng+data-cursor'}
    store.write(remote + '/_COMPLETE.json', result, temp / '_COMPLETE.json')
    run_remote = remote.rsplit('/checkpoints/', 1)[0]
    store.write(run_remote + '/LATEST.json', {'checkpoint': remote, **result}, temp / 'LATEST.json')
    print('OBS_CHECKPOINT_COMMITTED', step, remote, flush=True)
    return result


def restore_node(store, run_remote, selector, node, nodes, local, temp):
    run_remote = validate_remote(run_remote)
    temp, local = Path(temp), Path(local)
    if selector == 'latest':
        latest = store.read(run_remote + '/LATEST.json', temp / 'LATEST.json')
        remote = validate_remote(latest['checkpoint'])
        if not remote.startswith(run_remote + '/checkpoints/'):
            raise ValueError('LATEST points outside this run')
    else:
        if not selector.isdigit():
            raise ValueError('Resume must be latest or a numeric step')
        remote = f'{run_remote}/checkpoints/step-{int(selector):08d}'
    complete = store.read(remote + '/_COMPLETE.json', temp / '_COMPLETE.json')
    if complete['nodes'] != nodes or complete['world_size'] != nodes * 8:
        raise ValueError('Resume requires the same node/card topology')
    by_node = {r['node']: r['manifest_sha256'] for r in complete['node_manifests']}
    metas = {}
    for source_node in sorted({0, node}):
        mp = temp / f'node-{source_node:03d}.json'
        meta = store.read(f'{remote}/node-{source_node:03d}/_NODE_COMPLETE.json', mp)
        if digest(mp) != by_node[source_node]:
            raise ValueError('Committed node manifest hash mismatch')
        validate_node(meta, source_node, nodes, complete['step'], complete['fingerprint'])
        metas[source_node] = meta
    rows = [(node, row) for row in metas[node]['files']]
    if node != 0:
        rows += [(0, row) for row in metas[0]['files']
                 if row['path'] == 'pytorch_model/mp_rank_00_model_states.pt']
    local.mkdir(parents=True, exist_ok=True)
    for source_node, row in rows:
        p = local / row['path']
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_file() and p.stat().st_size == row['bytes'] and digest(p) == row['sha256']:
            continue
        partial = p.with_suffix(p.suffix + '.download')
        store.copy(f'{remote}/node-{source_node:03d}/' + row['path'], partial)
        if partial.stat().st_size != row['bytes'] or digest(partial) != row['sha256']:
            raise ValueError('Downloaded checkpoint checksum mismatch: ' + row['path'])
        partial.replace(p)
    atomic_json(local / '_RESTORED.json', complete)
    print('OBS_RESTORED', complete['step'], node, flush=True)
    return complete


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['upload', 'commit', 'restore', 'logs', 'init', 'export'])
    p.add_argument('--local', type=Path, required=True)
    p.add_argument('--remote', required=True)
    p.add_argument('--node', type=int, default=0)
    p.add_argument('--nodes', type=int, default=8)
    p.add_argument('--step', type=int)
    p.add_argument('--fingerprint')
    p.add_argument('--resume', default='latest')
    a = p.parse_args()
    remote = validate_remote(a.remote)
    store = Store()
    if a.command == 'upload':
        upload_node(store, a.local, remote, a.node, a.nodes)
    elif a.command == 'commit':
        commit_step(store, remote, a.nodes, a.step, a.fingerprint, a.local)
    elif a.command == 'restore':
        restore_node(store, remote, a.resume, a.node, a.nodes, a.local, a.local.parent / 'restore-receipts')
    elif a.command == 'export':
        rows = inventory(a.local)
        for row in rows:
            store.copy(a.local / row['path'], remote + '/' + row['path'])
        store.write(remote + '/_COMPLETE.json', {'step': a.step, 'files': rows}, a.local / '_COMPLETE.json')
    elif a.command == 'init':
        if a.node == 0:
            if store.exists(remote + '/_RUN.json'):
                if a.resume == 'none':
                    raise ValueError('RUN_NAME already exists in OBS; use resume or a new run name')
            else:
                if a.resume != 'none':
                    raise ValueError('Cannot resume an unknown run')
                store.write(remote + '/_RUN.json', {'run': remote, 'nodes': a.nodes}, a.local / '_RUN.json')
    else:
        # Small, changing logs are copied periodically; never traverse large checkpoints.
        for base in [a.local, a.local / 'run_config']:
            if base.is_dir():
                for path in base.iterdir():
                    if path.is_file() and path.suffix in {'.json', '.jsonl', '.log', '.yaml', '.txt'}:
                        rel = path.relative_to(a.local).as_posix()
                        store.copy(path, remote + '/' + rel)


if __name__ == '__main__':
    main()
