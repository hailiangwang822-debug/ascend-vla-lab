"""One invocation per node. Runs upstream Accelerate/DeepSpeed, with no nested launch."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def launch_spec(env, root):
    nodes = int(env.get('NNODES', env.get('MA_NUM_HOSTS', '1')))
    rank = int(env.get('NODE_RANK', env.get('VC_TASK_INDEX', '0')))
    per_node = int(env.get('NPROC_PER_NODE', '8'))
    hosts = env.get('VC_WORKER_HOSTS', '').split(',')
    master = env.get('MASTER_ADDR') or (hosts[0].strip() if hosts[0] else '127.0.0.1')
    if not 0 <= rank < nodes or per_node < 1:
        raise ValueError('Invalid node topology')
    if nodes > 1 and master in ('127.0.0.1', 'localhost'):
        raise ValueError('Multi-node run needs MASTER_ADDR or VC_WORKER_HOSTS')
    if env.get('LOCAL_RANK') is not None or env.get('RANK') is not None:
        raise ValueError('Already inside a per-card launcher. Use one entry process per node.')
    for name in ('RUN_NAME', 'MODEL_DIR', 'DATA_ROOT', 'OUTPUT_ROOT'):
        if not env.get(name):
            raise ValueError(f'Missing {name}')
    if 'Qwen2.5-VL' not in env['MODEL_DIR']:
        raise ValueError('Keep Qwen2.5-VL in model directory name: upstream dispatches by path')
    if any(x in env['RUN_NAME'] for x in ('/', '\\', '..')):
        raise ValueError('RUN_NAME must be a single directory name')
    total = nodes * per_node
    micro = int(env.get('MICRO_BATCH', '1'))
    batch = int(env.get('GLOBAL_BATCH', '64'))
    if micro < 1 or batch % (total * micro):
        raise ValueError('GLOBAL_BATCH must be divisible by nodes * cards * MICRO_BATCH')
    accum = batch // (total * micro)
    runtime = Path(env.get('STARVLA_RUNTIME', str(root / 'runtime')))
    ds = {'bf16': {'enabled': True}, 'fp16': {'enabled': False},
          'train_micro_batch_size_per_gpu': micro, 'train_batch_size': batch,
          'gradient_accumulation_steps': accum, 'gradient_clipping': 1.0,
          'zero_optimization': {'stage': 2, 'overlap_comm': False,
                                'contiguous_gradients': True,
                                'reduce_bucket_size': 50000000,
                                'allgather_bucket_size': 50000000},
          'steps_per_print': 10, 'wall_clock_breakdown': False}
    cfg = {'compute_environment': 'LOCAL_MACHINE', 'distributed_type': 'DEEPSPEED',
           'mixed_precision': 'bf16', 'num_machines': nodes, 'num_processes': total,
           'machine_rank': rank, 'main_process_ip': master,
           'main_process_port': int(env.get('MASTER_PORT', '28436')),
           'same_network': True, 'rdzv_backend': 'static',
           'deepspeed_config': {'deepspeed_config_file': str(runtime / 'deepspeed.json'),
                                'deepspeed_multinode_launcher': 'standard',
                                'zero3_init_flag': False}, 'use_cpu': False}
    command = [sys.executable, '-m', 'accelerate.commands.launch',
               '--config_file', str(runtime / 'accelerate.json'),
               str(root / 'starVLA/training/train_starvla.py'),
               '--config_yaml', str(root / 'cann83/train.yaml')]
    return {'nodes': nodes, 'node_rank': rank, 'cards_per_node': per_node,
            'world_size': total, 'global_batch': batch, 'micro_batch': micro,
            'accumulation': accum, 'accelerate': cfg, 'deepspeed': ds,
            'command': command}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    spec = launch_spec(env, root)
    print(json.dumps(spec, indent=2), flush=True)
    if args.dry_run:
        return
    if 'MA_LOCAL_TASK_INDEX' in env and int(env['MA_LOCAL_TASK_INDEX']) != 0:
        raise SystemExit('ModelArts started this per card; configure one entry per node first')
    for name in ('MODEL_DIR', 'DATA_ROOT'):
        if not Path(env[name]).is_dir():
            raise FileNotFoundError(env[name])
    env.update(STARVLA_USE_NPU='1', STARVLA_GRAD_ACCUM=str(spec['accumulation']),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS=env.get('OMP_NUM_THREADS', '1'),
               OPENBLAS_NUM_THREADS=env.get('OPENBLAS_NUM_THREADS', '1'),
               NO_ALBUMENTATIONS_UPDATE='1', ALBUMENTATIONS_OFFLINE='1',
               TORCH_DIST_TIMEOUT_MINUTES='30')
    env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
    runtime = Path(env.get('STARVLA_RUNTIME', str(root / 'runtime')))
    runtime.mkdir(parents=True, exist_ok=True)
    for name, data in [('deepspeed', spec['deepspeed']), ('accelerate', spec['accelerate'])]:
        (runtime / f'{name}.json').write_text(json.dumps(data, indent=2))
    run_dir = Path(env['OUTPUT_ROOT']) / env['RUN_NAME']
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / f'launch-node-{spec["node_rank"]}.json'
    if manifest.exists():
        raise FileExistsError('RUN_NAME already used on this node; choose a new name')
    manifest.write_text(json.dumps({**spec, 'hostname': socket.gethostname()}, indent=2))
    subprocess.run([sys.executable, str(root / 'cann83/preflight.py')],
                   cwd=root, env=env, check=True)
    with (run_dir / f'console-node-{spec["node_rank"]}.log').open('w') as log:
        child = subprocess.Popen(spec['command'], cwd=root, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        for line in child.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = child.wait()
    (run_dir / f'exit-node-{spec["node_rank"]}.json').write_text(json.dumps({'exit_code': code}))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
