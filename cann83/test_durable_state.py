"""Local failure-injection/CPU recovery checks; these are not NPU/OBS integration tests."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from obs_state import Store, upload_node, commit_step, restore_node, digest, validate_remote, configured_root
from bootstrap import specification, secondary_entry
from launch import launch_spec

REMOTE = 's3://test-bucket/test-prefix/vla/runs/test'


class FakeFiles:
    def __init__(self, root):
        self.root, self.fail = Path(root), None

    def path(self, value):
        value = str(value)
        return self.root / value[5:] if value.startswith('s3://') else Path(value)

    def exists(self, value):
        return self.path(value).exists()

    def copy(self, src, dst):
        if self.fail and self.fail in str(dst):
            raise OSError('injected transfer failure')
        src, dst = self.path(src), self.path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


class DurableStateTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict('os.environ', {'STARVLA_OBS_ROOT': REMOTE.rsplit('/runs/', 1)[0]})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.files = FakeFiles(self.root / 'objects')
        self.store = Store(self.files)

    def tearDown(self):
        self.tmp.cleanup()

    def node(self, n, step):
        p = self.root / f'local-{step}-{n}'
        (p / 'pytorch_model').mkdir(parents=True)
        (p / 'progress.json').write_text(json.dumps({'completed_steps': step, 'fingerprint': 'abc'}))
        for name in ['custom_checkpoint_0.pkl', 'config.full.yaml', 'dataset_statistics.json']:
            (p / name).write_text(name)
        for rank in range(n * 8, (n + 1) * 8):
            (p / f'random_states_{rank}.pkl').write_text(str(rank))
            (p / f'rank_state_{rank}.pt').write_text(str(rank))
            (p / f'pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt').write_text(str(rank))
        if n == 0:
            (p / 'pytorch_model/mp_rank_00_model_states.pt').write_text('model')
        return p

    def publish(self, step):
        remote = REMOTE + f'/checkpoints/step-{step:08d}'
        for n in range(8):
            upload_node(self.store, self.node(n, step), remote, n, 8)
        commit_step(self.store, remote, 8, step, 'abc', self.root / 'receipts')
        return remote

    def test_partial_cannot_replace_latest(self):
        self.publish(1)
        old = self.files.path(REMOTE + '/LATEST.json').read_bytes()
        remote = REMOTE + '/checkpoints/step-00000100'
        upload_node(self.store, self.node(0, 100), remote, 0, 8)
        with patch('obs_state.time.sleep'), self.assertRaises(FileNotFoundError):
            commit_step(self.store, remote, 8, 100, 'abc', self.root / 'receipts2')
        self.assertEqual(old, self.files.path(REMOTE + '/LATEST.json').read_bytes())
        self.assertFalse(self.store.exists(remote + '/_COMPLETE.json'))

    def test_restore_nonzero_node_model_and_own_shards(self):
        self.publish(100)
        out = self.root / 'restored'
        state = restore_node(self.store, REMOTE, 'latest', 6, 8, out, self.root / 'read')
        self.assertEqual(state['step'], 100)
        self.assertEqual((out / 'pytorch_model/mp_rank_00_model_states.pt').read_text(), 'model')
        self.assertEqual(len(list(out.rglob('*optim_states.pt'))), 8)
        self.assertEqual(len(list(out.glob('random_states_*.pkl'))), 8)
        with self.assertRaises(ValueError):
            restore_node(self.store, REMOTE, 'latest', 0, 4, self.root / 'wrong', self.root / 'read2')

    def test_corrupt_remote_rejected(self):
        remote = self.publish(1)
        self.files.path(remote + '/node-003/random_states_24.pkl').write_text('corrupt')
        with self.assertRaises(ValueError):
            restore_node(self.store, REMOTE, 'latest', 3, 8, self.root / 'bad', self.root / 'read')
        self.assertFalse((self.root / 'bad/_RESTORED.json').exists())

    def test_failed_upload_never_publishes_node(self):
        remote = REMOTE + '/checkpoints/step-00000100'
        self.files.fail = 'optim_states.pt'
        with patch('obs_state.time.sleep'), self.assertRaises(OSError):
            upload_node(self.store, self.node(0, 100), remote, 0, 8)
        self.assertFalse(self.store.exists(remote + '/node-000/_NODE_COMPLETE.json'))

    def test_64_card_entry(self):
        env = dict(STARVLA_OBS_ROOT=REMOTE.rsplit('/runs/', 1)[0], MA_NUM_HOSTS='8', VC_TASK_INDEX='7', VC_WORKER_HOSTS='host0,host1',
                   RUN_NAME='test64', MODEL_DIR='/cache/Qwen2.5-VL-3B-Instruct', DATA_ROOT='/cache/data',
                   OUTPUT_ROOT='/cache/out', STARVLA_RUNTIME='/cache/private/node7')
        self.assertEqual(specification(env)['world_size'], 64)
        s = launch_spec(env, Path('/code/A'))
        self.assertEqual((s['world_size'], s['accumulation']), (64, 1))
        self.assertEqual(s['accelerate']['machine_rank'], 7)
        self.assertEqual(Path(s['accelerate']['deepspeed_config']['deepspeed_config_file']).parent,
                         Path('/cache/private/node7'))
        with self.assertRaises(ValueError):
            specification({**env, 'MA_NUM_HOSTS': '4'})
        self.assertTrue(secondary_entry({'RANK_TABLE_FILE': '/x', 'ASCEND_DEVICE_ID': '1'}))
        with self.assertRaises(ValueError):
            secondary_entry({'MA_RUN_METHOD': 'torchrun'})

    def test_configurable_obs_root(self):
        root = 's3://another-bucket/project/vla'
        env = {'STARVLA_OBS_ROOT': root}
        self.assertEqual(specification(env)['run_remote'], root + '/runs/starvla_spatial_64p_001')
        self.assertEqual(validate_remote(root + '/assets/a', root=root), root + '/assets/a')
        with self.assertRaises(ValueError):
            validate_remote(root + '-other/assets/a', root=root)
        for bad in ['', 'https://bucket/prefix', 's3://bucket', 's3://bucket/a/../b',
                    's3://bucket//b', 's3://user:pass@bucket/a', 's3://bucket/a%20b']:
            with self.assertRaises(ValueError):
                configured_root({'STARVLA_OBS_ROOT': bad})

    def test_path_restriction(self):
        for value in ['s3://other/private', REMOTE + '/../escape', REMOTE + '/%2e%2e']:
            with self.assertRaises(ValueError):
                validate_remote(value)

    def test_cpu_full_state_and_cursor_match_uninterrupted(self):
        import numpy as np
        import torch
        from accelerate import Accelerator
        from accelerate.utils import set_seed, ProjectConfiguration
        from torch.utils.data import Dataset, DataLoader
        from training_checkpoint import DurableTrainerMixin

        class Data(Dataset):
            epoch = 0
            def set_epoch(self, epoch):
                self.epoch = epoch
            def __len__(self):
                return 9
            def __getitem__(self, index):
                return torch.tensor([index / 10, self.epoch / 10], dtype=torch.float32)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Dropout(.2), torch.nn.Linear(8, 1))
                self.register_buffer('updates', torch.tensor(0))
            @property
            def global_steps(self):
                return self.updates.item()
            def forward(self, x):
                return self.net(x)

        class Config(SimpleNamespace):
            def save_full_config(self, path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text('seed: 42')

        class Base:
            def prepare_training(self):
                self.accelerator = Accelerator(cpu=True, project_config=ProjectConfiguration(save_on_each_node=True))
                self.model = Model()
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01)
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)
                dl = DataLoader(Data(), batch_size=2, generator=torch.Generator().manual_seed(42))
                self.model, self.optimizer, self.vla_train_dataloader = self.accelerator.prepare(self.model, self.optimizer, dl)
                self.completed_steps = 0

        class Trainer(DurableTrainerMixin, Base):
            pass

        def fresh(folder, resume=None):
            set_seed(42)
            trainer = Trainer()
            trainer.config = Config(output_dir=str(folder), seed=42,
                                    datasets=SimpleNamespace(vla_data=SimpleNamespace(num_workers=0)))
            with patch.dict('os.environ', {'STARVLA_RESUME_DIR': str(resume)} if resume else {}, clear=True), \
                 patch('training_checkpoint.identity', return_value='cpu-test'):
                trainer.prepare_training()
            trainer._create_data_iterators()
            return trainer

        def run(t, end):
            trace = []
            while t.completed_steps < end:
                batch = t._get_next_batch()
                trace.append(batch.clone())
                # Exercise all three saved RNG families, plus dropout and Adam moments.
                noise = random.random() + np.random.random() + torch.rand(()).item()
                loss = (t.model(batch) - noise).square().mean()
                t.accelerator.backward(loss)
                t.optimizer.step()
                t.lr_scheduler.step()
                t.optimizer.zero_grad()
                t.model.updates += 1
                t.completed_steps += 1
            return trace

        continuous = fresh(self.root / 'continuous')
        expected_trace = run(continuous, 20)
        expected_weights = {k: v.clone() for k, v in continuous.model.state_dict().items()}
        split = fresh(self.root / 'split')
        run(split, 7)
        saved = self.root / 'cpu-checkpoint'
        split.accelerator.save_state(str(saved))
        torch.save({'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()},
                   saved / 'rank_state_0.pt')
        (saved / 'progress.json').write_text(json.dumps({'fingerprint': 'cpu-test',
            'completed_steps': 7, 'consumed_batches': split.consumed_batches}))
        resumed = fresh(self.root / 'resume', saved)
        self.assertEqual(resumed.completed_steps, 7)
        actual_trace = run(resumed, 20)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected_trace[7:], actual_trace)))
        self.assertTrue(all(torch.equal(expected_weights[k], v) for k, v in resumed.model.state_dict().items()))
        self.assertEqual(continuous.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())


if __name__ == '__main__':
    unittest.main(verbosity=2)
