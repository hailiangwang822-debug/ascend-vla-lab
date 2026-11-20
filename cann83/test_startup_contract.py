"""Regression checks for node-level ModelArts startup; no NPU or OBS required."""
from pathlib import Path
import contextlib
import io
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
from launch import launch_spec, validate_topology
from runtime_contract import BASE_PACKAGES, clean_child_environment, validate_base_versions


class StartupContractTests(unittest.TestCase):
    def env(self, node):
        return {
            'STARVLA_OBS_ROOT': 's3://test-bucket/project/vla',
            'MA_NUM_HOSTS': '8', 'VC_TASK_INDEX': str(node),
            'VC_WORKER_HOSTS': ','.join(f'worker-{i}' for i in range(8)),
            'RANK_TABLE_FILE': '/platform/rank_table.json',
            'RUN_NAME': 'startup-regression',
            'MODEL_DIR': '/cache/models/Qwen2.5-VL-3B-Instruct',
            'DATA_ROOT': '/cache/data', 'OUTPUT_ROOT': '/cache/out',
        }

    def test_reported_rtf_without_device_id_launches_node_zero_and_five(self):
        for node in (0, 5):
            with self.subTest(node=node):
                env = self.env(node)
                self.assertNotIn('ASCEND_DEVICE_ID', env)
                self.assertFalse(bootstrap.secondary_entry(env))
                spec = bootstrap.specification(env)
                launch = launch_spec(clean_child_environment(env), Path('/code/A'))
                self.assertEqual(spec['node'], node)
                self.assertEqual(launch['accelerate']['machine_rank'], node)
                self.assertEqual(launch['accelerate']['main_process_ip'], 'worker-0')
                self.assertEqual(launch['accelerate']['num_machines'], 8)
                self.assertEqual(launch['accelerate']['num_processes'], 64)
                self.assertEqual(launch['cards_per_node'], 8)
                self.assertEqual(launch['deepspeed']['train_batch_size'], 64)
                self.assertEqual(launch['deepspeed']['gradient_accumulation_steps'], 1)
                self.assertEqual(launch['accelerate']['deepspeed_config']['deepspeed_multinode_launcher'], 'standard')

    def test_reported_environment_reaches_execute_through_real_cli_entry(self):
        # Windows lacks fcntl; stub only the OS locking boundary. Exercise the
        # actual argument parsing, specification and entry logic from the log.
        for node in (0, 5):
            with self.subTest(node=node), tempfile.TemporaryDirectory() as folder:
                env = {**self.env(node), 'STARVLA_CACHE': folder}
                lock_api = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
                with patch.dict('os.environ', env, clear=True), \
                     patch.dict('sys.modules', {'fcntl': lock_api}), \
                     patch.object(sys, 'argv', ['bootstrap.py', 'train']), \
                     patch('bootstrap.execute') as execute, \
                     contextlib.redirect_stdout(io.StringIO()):
                    bootstrap.main()
                    execute.assert_called_once()
                    self.assertEqual(execute.call_args.args[0]['node'], node)

    def test_duplicate_launcher_fails_before_execute(self):
        with tempfile.TemporaryDirectory() as folder:
            env = {**self.env(5), 'STARVLA_CACHE': folder}
            def busy(*args):
                raise BlockingIOError('lock busy')
            lock_api = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=busy)
            with patch.dict('os.environ', env, clear=True), \
                 patch.dict('sys.modules', {'fcntl': lock_api}), \
                 patch.object(sys, 'argv', ['bootstrap.py', 'train']), \
                 patch('bootstrap.execute') as execute, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'Another launcher is active'):
                    bootstrap.main()
                execute.assert_not_called()

    def test_platform_metadata_never_silently_drops_worker_node(self):
        for node in range(8):
            with self.subTest(node=node):
                env = {**self.env(node), 'ASCEND_DEVICE_ID': str(node),
                       'MA_LOCAL_TASK_INDEX': str(node)}
                self.assertFalse(bootstrap.secondary_entry(env))
                child = clean_child_environment(env)
                self.assertEqual(launch_spec(child, Path('/code/A'))['node_rank'], node)

    def test_clean_child_preserves_platform_topology_and_device_visibility(self):
        env = {**self.env(5), 'ASCEND_RT_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
               'ASCEND_DEVICE_ID': '5', 'MA_LOCAL_TASK_INDEX': '5',
               'RANK_ID': '40', 'RANK_SIZE': '64', 'DEVICE_ID': '5',
               'RANK': '40', 'WORLD_SIZE': '64', 'LOCAL_WORLD_SIZE': '8'}
        original = env.copy()
        child = clean_child_environment(env)
        self.assertEqual(env, original)
        for key in ('MA_NUM_HOSTS', 'VC_TASK_INDEX', 'VC_WORKER_HOSTS', 'ASCEND_RT_VISIBLE_DEVICES'):
            self.assertEqual(child[key], env[key])
        for key in ('RANK_TABLE_FILE', 'RANK_ID', 'RANK_SIZE', 'DEVICE_ID',
                    'ASCEND_DEVICE_ID', 'RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MA_LOCAL_TASK_INDEX'):
            self.assertNotIn(key, child)

    def test_nested_launcher_rejected_before_metadata_cleanup(self):
        for extra in ({'LOCAL_RANK': '0'}, {'LOCAL_RANK': '7'},
                      {'MA_RUN_METHOD': 'torchrun'}, {'MA_RUN_METHOD': 'msrun'},
                      {'MA_RUN_METHOD': 'TorchRun'}):
            with self.subTest(extra=extra):
                env = {**self.env(5), **extra}
                with self.assertRaisesRegex(ValueError, 'one entry per node'):
                    bootstrap.secondary_entry(env)
                with self.assertRaisesRegex(ValueError, 'one entry per node'):
                    clean_child_environment(env)

    def test_stale_accelerate_settings_do_not_override_this_job(self):
        env = {**self.env(0), 'ACCELERATE_CONFIG_DS_FIELDS': 'mixed_precision,zero_stage',
               'ACCELERATE_USE_FSDP': 'true', 'ACCELERATE_MIXED_PRECISION': 'fp16',
               'ACCELERATE_DEEPSPEED_ZERO_STAGE': '3', 'HCCL_CONNECT_TIMEOUT': '7200'}
        child = clean_child_environment(env)
        self.assertFalse(any(k.startswith('ACCELERATE_') for k in child))
        self.assertEqual(child['HCCL_CONNECT_TIMEOUT'], '7200')
        self.assertEqual(env['ACCELERATE_MIXED_PRECISION'], 'fp16')

    def test_base_versions_accept_local_build_suffix(self):
        versions = {**BASE_PACKAGES, 'torch': '2.7.1+cpu'}
        validate_base_versions(versions, (3, 11, 10), 'aarch64')

    def test_wrong_image_fails_before_obs_or_asset_transfer(self):
        env = self.env(5)
        with patch.dict('os.environ', env, clear=True), \
             patch('bootstrap.check_base_runtime', side_effect=RuntimeError('wrong image')), \
             patch('bootstrap.Store') as store, \
             patch('bootstrap.prepare_assets') as prepare:
            with self.assertRaisesRegex(RuntimeError, 'wrong image'):
                bootstrap.execute(bootstrap.specification(env))
            store.assert_not_called()
            prepare.assert_not_called()

    def test_incompatible_runtime_rejected(self):
        cases = [({**BASE_PACKAGES, 'torch': '2.9.0'}, (3, 11, 10), 'aarch64'),
                 ({**BASE_PACKAGES, 'torch-npu': 'missing'}, (3, 11, 10), 'aarch64'),
                 (BASE_PACKAGES, (3, 12, 0), 'aarch64'),
                 (BASE_PACKAGES, (3, 11, 10), 'x86_64')]
        for versions, python, arch in cases:
            with self.subTest(versions=versions, python=python, arch=arch):
                with self.assertRaises(RuntimeError):
                    validate_base_versions(versions, python, arch)

    def test_invalid_rank_port_and_batch_fail_before_workers(self):
        env = self.env(5)
        for extra in ({'VC_TASK_INDEX': '-1'}, {'VC_TASK_INDEX': '8'},
                      {'MASTER_PORT': '0'}, {'MASTER_PORT': '65536'},
                      {'GLOBAL_BATCH': '0'}, {'GLOBAL_BATCH': '63'}):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    launch_spec(clean_child_environment({**env, **extra}), Path('/code/A'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
