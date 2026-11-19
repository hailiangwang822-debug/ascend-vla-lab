"""Full-state checkpoint/resume hooks for the pinned native StarVLA trainer."""
import hashlib
import json
import os
import random
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import torch.distributed as dist


def identity(config, accelerator):
    from omegaconf import OmegaConf
    from importlib import metadata
    p = Path(config.output_dir) / 'config.full.yaml'
    body = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
    for name in ('output_dir', 'run_root_dir', 'run_id', 'wandb_entity', 'wandb_project'):
        body.pop(name, None)
    body['framework']['qwenvl']['base_vlm'] = 'Qwen2.5-VL-3B-Instruct@1b5a7674d40a0ac9acef541a46fed12d710bd36b'
    body['datasets']['vla_data']['data_root_dir'] = 'IPEC-spatial@bf14d6258218d12c2e3c1a3b9922e163cdf6455d'
    for name in ('save_interval', 'logging_frequency'):
        body['trainer'].pop(name, None)
    root = Path(__file__).resolve().parents[1]
    body['adaptation'] = json.loads((root / 'adaptation-manifest.json').read_text())['files']
    body['versions'] = {k: metadata.version(k) for k in ('torch', 'torch-npu', 'accelerate', 'deepspeed', 'transformers')}
    body['world_size'] = accelerator.num_processes
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class DurableTrainerMixin:
    def prepare_training(self):
        super().prepare_training()
        self.consumed_batches = 0
        # Native trainer owns this scheduler directly; register it explicitly.
        self.accelerator.register_for_checkpointing(self.lr_scheduler)
        self._checkpoint_fingerprint = None
        # cfg.full is written only on global rank0 in upstream; materialize it
        # once per node too, before computing the common compatibility identity.
        if self.accelerator.is_local_main_process:
            self.config.save_full_config(Path(self.config.output_dir) / 'config.full.yaml')
        self.accelerator.wait_for_everyone()
        self._checkpoint_fingerprint = identity(self.config, self.accelerator)
        self._last_committed_step = -1
        if os.environ.get('STARVLA_RESUME_DIR'):
            path = Path(os.environ['STARVLA_RESUME_DIR'])
            state = json.loads((path / 'progress.json').read_text())
            if state['fingerprint'] != self._checkpoint_fingerprint:
                raise ValueError('Resume config/source/runtime mismatch; preserve the original training settings')
            if dist.is_initialized():
                low = torch.tensor([state['completed_steps']], device=self.accelerator.device, dtype=torch.int64)
                high = low.clone()
                dist.all_reduce(low, op=dist.ReduceOp.MIN)
                dist.all_reduce(high, op=dist.ReduceOp.MAX)
                if low.item() != high.item():
                    raise ValueError('Nodes restored different steps; retry with RESUME=<explicit step>')
            if int(self.config.datasets.vla_data.num_workers) != 0:
                raise ValueError('Cursor/RNG resume currently requires DATA_WORKERS=0')
            self.accelerator.load_state(str(path))
            # Accelerate 1.11 has no explicit Ascend RNG branch and catches RNG
            # restore exceptions. Restore our explicit per-rank state strictly.
            rng = torch.load(path / f'rank_state_{self.accelerator.process_index}.pt',
                             map_location='cpu', weights_only=False)
            random.setstate(rng['python'])
            import numpy as np
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            if 'npu' in rng:
                torch.npu.set_rng_state(rng['npu'], device=self.accelerator.device)
            self.completed_steps = state['completed_steps']
            self._last_committed_step = self.completed_steps
            self.consumed_batches = state['consumed_batches']
            if self.consumed_batches != self.completed_steps * self.accelerator.gradient_accumulation_steps:
                raise ValueError('Saved update/data cursor mismatch')
            if self.model.global_steps != self.completed_steps:
                raise ValueError('DeepSpeed did not restore the saved optimizer step')
            self.accelerator.print('FULL_STATE_RESUMED', self.completed_steps, 'data_batches', self.consumed_batches)

    def _create_data_iterators(self):
        epoch, offset = divmod(self.consumed_batches, len(self.vla_train_dataloader))
        dl = self.vla_train_dataloader
        dl.set_epoch(epoch)
        dl.dataset.set_epoch(epoch)
        if dl.generator is not None:
            dl.generator.manual_seed(int(self.config.seed) + epoch)
        self.vla_epoch_count = epoch
        active = self.accelerator.skip_first_batches(dl, offset) if offset else dl
        # Accelerate 1.11 constructs the skipped loader with iteration=0.
        # Explicitly restore epoch so its __iter__ cannot reset dataset to epoch0.
        active.set_epoch(epoch)
        self.vla_iter = iter(active)

    def _get_next_batch(self):
        try:
            batch = next(self.vla_iter)
        except StopIteration:
            self._create_data_iterators()
            batch = next(self.vla_iter)
        self.consumed_batches += 1
        return batch

    def _collective_operation(self, function, local_leader=True):
        success = 1
        leader = self.accelerator.is_local_main_process if local_leader else self.accelerator.is_main_process
        if leader:
            try:
                function()
            except Exception as e:
                print('CHECKPOINT_ERROR', type(e).__name__, str(e), flush=True)
                success = 0
        flag = torch.tensor([success], device=self.accelerator.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not flag.item():
            raise RuntimeError('Checkpoint upload/commit failed; no new global resume marker was published')

    def _obs_command(self, command, *args):
        script = Path(__file__).with_name('obs_state.py')
        python = os.environ['STARVLA_BOOTSTRAP_PYTHON']
        subprocess.run([python, '-u', str(script), command, *map(str, args)], check=True)

    def _save_checkpoint(self):
        path = Path(self.checkpoint_dir) / f'step-{self.completed_steps:08d}'
        path.mkdir(parents=True, exist_ok=True)
        import numpy as np
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
        if self.accelerator.device.type == 'npu':
            rng['npu'] = torch.npu.get_rng_state(self.accelerator.device)
        torch.save(rng, path / f'rank_state_{self.accelerator.process_index}.pt')
        self.accelerator.save_state(str(path))  # ALL ranks: model/master/optimizer/RNG
        self.accelerator.wait_for_everyone()
        node = int(os.environ['VC_TASK_INDEX'])
        nodes = int(os.environ['MA_NUM_HOSTS'])
        remote = os.environ['STARVLA_OBS_RUN'] + '/checkpoints/' + path.name

        def publish_node():
            state = {'completed_steps': self.completed_steps, 'consumed_batches': self.consumed_batches,
                     'fingerprint': self._checkpoint_fingerprint, 'world_size': self.accelerator.num_processes}
            (path / 'progress.json').write_text(json.dumps(state, indent=2))
            self.config.save_full_config(path / 'config.full.yaml')
            stats = Path(self.config.output_dir) / f'dataset-preflight-node-{node}.json'
            shutil.copyfile(stats, path / 'dataset_statistics.json')
            self._obs_command('upload', '--local', path, '--remote', remote, '--node', node, '--nodes', nodes)

        self._collective_operation(publish_node)
        self._collective_operation(lambda: self._obs_command('commit', '--local', path.parent / 'receipts',
            '--remote', remote, '--nodes', nodes, '--step', self.completed_steps,
            '--fingerprint', self._checkpoint_fingerprint), local_leader=False)
        if self.accelerator.is_local_main_process:
            (path / '_OBS_COMMITTED.json').write_text(json.dumps({'remote': remote}))
            older = sorted(p for p in path.parent.glob('step-*') if p.is_dir())[:-2]
            for p in older:
                if p.resolve().parent == path.parent.resolve() and (p / '_OBS_COMMITTED.json').is_file():
                    shutil.rmtree(p)  # only older, globally committed LOCAL copies
        self.accelerator.wait_for_everyone()
        self._last_committed_step = self.completed_steps
        self.accelerator.print('DURABLE_CHECKPOINT', self.completed_steps, remote)

    def _finalize_training(self):
        # Last step may already have been checkpointed by the regular save interval.
        if self._last_committed_step != self.completed_steps:
            self._save_checkpoint()
        self._collective_operation(self._export_final, local_leader=False)
        if self.accelerator.is_main_process:
            import wandb
            wandb.finish()
        self.accelerator.wait_for_everyone()

    def _export_final(self):
        folder = Path(self.config.output_dir) / 'final_model'
        folder.mkdir(exist_ok=True)
        weights = self.accelerator.get_state_dict(self.model)  # ZeRO-2, no gather collective
        torch.save({k: v.detach().cpu() for k, v in weights.items()}, folder / 'pytorch_model.pt')
        self.config.save_full_config(folder / 'config.full.yaml')
        shutil.copyfile(Path(self.config.output_dir) / 'dataset_statistics.json', folder / 'dataset_statistics.json')
        remote = os.environ['STARVLA_OBS_RUN'] + '/final_model'
        self._obs_command('export', '--local', folder, '--remote', remote, '--step', self.completed_steps)
        self.accelerator.print('FINAL_MODEL_UPLOADED', remote)
