"""Regression of the observed image labels, including all three real guards.

OS/NPU/cloud boundaries are stubbed; this does not validate actual NPU kernels.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
from obs_state import digest
import runtime_contract as runtime


# Exact package labels from the failed submitted job, not the preferred pins.
ACTUAL_JOB = {'torch': '2.7.1', 'torch-npu': '2.7.1',
              'torchvision': '0.22.1', 'accelerate': '1.11.0',
              'deepspeed': '0.18.4'}
ISOLATED_JOB = {**ACTUAL_JOB, 'numpy': '1.26.4',
                'transformers': '4.57.1', 'pydantic': '2.10.6',
                'numpydantic': '1.6.9'}


class CloudBoundaryReached(Exception):
    pass


class RuntimeVersionTests(unittest.TestCase):
    @contextlib.contextmanager
    def image_runtime(self, versions):
        # Only redirect the fixed CANN filename. All source manifest and file
        # hashing operations continue to use a real temporary filesystem.
        cann = SimpleNamespace(is_file=lambda: True,
                               read_text=lambda: 'toolkit_running_version=[8.3.0.1.200:8.3.RC1]')
        real_path = Path
        def path(value):
            if str(value) == '/usr/local/Ascend/ascend-toolkit/latest/version.cfg':
                return cann
            return real_path(value)
        with patch.object(runtime.metadata, 'version', side_effect=versions.__getitem__), \
             patch.object(runtime.platform, 'machine', return_value='aarch64'), \
             patch.object(runtime.sys, 'version_info', (3, 11, 10)), \
             patch.object(runtime, 'Path', side_effect=path):
            yield

    def make_manifest(self, root):
        source = root / 'test-source.py'
        source.write_text('verified source\n')
        (root / 'adaptation-manifest.json').write_text(json.dumps({
            'release': 'test-v0.1.2',
            'files': [{'path': source.name, 'sha256': digest(source)}]}))
        return source

    def test_base_policy_accepts_both_observed_builds_and_local_labels(self):
        for label in ('2.7.1', '2.7.1.post2', '2.7.1+vendor', '2.7.1.post2+vendor'):
            with self.subTest(label=label):
                runtime.validate_base_versions({**ACTUAL_JOB, 'torch-npu': label,
                                                'torch': '2.7.1+cpu'}, (3, 11, 10), 'aarch64')

    def test_unobserved_npu_versions_rejected_in_base_and_isolated_policy(self):
        for label in ('2.6.0', '2.8.0', '2.9.0', '2.7.0', '2.7.1.post3',
                      '2.7.1.post99', '2.7.1.dev0', 'missing', None):
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, 'torch-npu'):
                    runtime.validate_base_versions({**ACTUAL_JOB, 'torch-npu': label}, (3, 11, 10), 'aarch64')
                with self.assertRaisesRegex(RuntimeError, 'torch-npu'):
                    runtime.validate_package_versions({**ISOLATED_JOB, 'torch-npu': label}, runtime.ISOLATED_PACKAGES)
        with self.assertRaisesRegex(RuntimeError, 'torch-npu'):
            runtime.validate_base_versions({k: v for k, v in ACTUAL_JOB.items() if k != 'torch-npu'}, (3, 11, 10), 'aarch64')

    def test_other_package_pins_and_platform_remain_strict(self):
        for name, wrong in [('torch', '2.7.1.post2'), ('torchvision', '0.24.0'),
                            ('accelerate', '1.12.0'), ('deepspeed', '0.18.5'),
                            ('transformers', '5.8.0')]:
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, name):
                runtime.validate_package_versions({**ISOLATED_JOB, name: wrong}, runtime.ISOLATED_PACKAGES)
        for python, arch in [((3, 12, 0), 'aarch64'), ((3, 11, 10), 'x86_64')]:
            with self.subTest(python=python, arch=arch), self.assertRaises(RuntimeError):
                runtime.validate_base_versions(ACTUAL_JOB, python, arch)

    def test_logged_runtime_reaches_real_source_manifest_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.make_manifest(root)
            output = io.StringIO()
            with self.image_runtime(ACTUAL_JOB), contextlib.redirect_stdout(output):
                runtime.check_base_runtime(root)
                self.assertIn('BASE_RUNTIME_PASSED test-v0.1.2', output.getvalue())
                source.write_text('corrupted source\n')
                with self.assertRaisesRegex(RuntimeError, 'Mixed or modified code package'):
                    runtime.check_base_runtime(root)

    def test_logged_runtime_reaches_obs_boundary_through_real_bootstrap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_manifest(root)
            env = {'STARVLA_OBS_ROOT': 's3://test-bucket/project/vla',
                   'MA_NUM_HOSTS': '8', 'VC_TASK_INDEX': '5',
                   'VC_WORKER_HOSTS': ','.join('worker-' + str(i) for i in range(8)),
                   'RANK_TABLE_FILE': '/user/config', 'RUN_NAME': 'actual-log-regression',
                   'STARVLA_CACHE': str(root / 'cache')}
            with patch.dict(os.environ, env, clear=True), \
                 patch.object(bootstrap, 'ROOT', root), self.image_runtime(ACTUAL_JOB), \
                 patch.object(bootstrap, 'Store', side_effect=CloudBoundaryReached) as store, \
                 patch.object(bootstrap, 'prepare_assets') as transfer, \
                 contextlib.redirect_stdout(io.StringIO()):
                # check_base_runtime is intentionally NOT mocked.
                with self.assertRaises(CloudBoundaryReached):
                    bootstrap.execute(bootstrap.specification(env))
                store.assert_called_once_with()
                transfer.assert_not_called()

    @contextlib.contextmanager
    def preflight_runtime(self, metadata_npu, module_npu, model_dir):
        versions = {**ISOLATED_JOB, 'torch-npu': metadata_npu}
        modules = {}
        for name, package in [('torch', 'torch'), ('torch_npu', 'torch-npu'),
                              ('transformers', 'transformers'), ('accelerate', 'accelerate'),
                              ('deepspeed', 'deepspeed')]:
            module = ModuleType(name)
            module.__version__ = module_npu if name == 'torch_npu' else versions[package]
            module.__file__ = '/test-image/' + name + '/__init__.py'
            modules[name] = module
        modules['torch'].npu = SimpleNamespace(is_available=Mock(return_value=True),
                                              device_count=Mock(return_value=8))
        modules['fcntl'] = SimpleNamespace(LOCK_EX=1, flock=lambda *args: None)
        spec = importlib.util.spec_from_file_location('_version_preflight', Path(__file__).with_name('preflight.py'))
        preflight = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(preflight)
            with patch.object(preflight.metadata, 'version', side_effect=versions.__getitem__), \
                 patch.object(preflight.platform, 'machine', return_value='aarch64'), \
                 patch.object(preflight.importlib, 'import_module'), \
                 patch.object(preflight, 'validate_package_versions', wraps=runtime.validate_package_versions) as guard, \
                 patch.dict(os.environ, {'MODEL_DIR': str(model_dir)}, clear=True), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                yield preflight, guard, modules['torch'].npu, output

    def test_real_preflight_metadata_and_module_guards_accept_observed_cross_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            for metadata_label in ('2.7.1', '2.7.1.post2', '2.7.1+vendor'):
                for module_label in ('2.7.1', '2.7.1.post2', '2.7.1.post2+vendor'):
                    with self.subTest(metadata=metadata_label, module=module_label), \
                         self.preflight_runtime(metadata_label, module_label, folder) as (preflight, guard, npu, output):
                        # Reaching the next real model-file read proves BOTH
                        # guard layers passed; later heavy data checks are not mocked as successes.
                        with self.assertRaisesRegex(FileNotFoundError, 'model.safetensors.index.json'):
                            preflight.main()
                        self.assertEqual(guard.call_count, 6)
                        self.assertEqual(guard.call_args_list[0].args[0]['torch-npu'], metadata_label)
                        self.assertEqual(guard.call_args_list[2].args[0], {'torch-npu': module_label})
                        self.assertTrue(npu.is_available.called)
                        if metadata_label.split('+')[0] != module_label.split('+')[0]:
                            self.assertIn('RUNTIME_VERSION_LABEL_DIFF torch-npu', output.getvalue())

    def test_real_preflight_rejects_bad_metadata_before_npu_check(self):
        for label in ('2.9.0', '2.7.1.post99', '2.7.1.dev0', 'missing'):
            with self.subTest(label=label), self.preflight_runtime(label, '2.7.1', '/unused') as (preflight, guard, npu, output):
                with self.assertRaisesRegex(RuntimeError, 'torch-npu'):
                    preflight.main()
                self.assertEqual(guard.call_count, 1)
                npu.is_available.assert_not_called()

    def test_real_preflight_rejects_bad_imported_module_after_metadata_passes(self):
        for label in ('2.8.0', '2.7.1.post99', '2.7.1.dev0', 'missing'):
            with self.subTest(label=label), self.preflight_runtime('2.7.1', label, '/unused') as (preflight, guard, npu, output):
                with self.assertRaisesRegex(RuntimeError, 'torch-npu'):
                    preflight.main()
                self.assertEqual(guard.call_count, 3)
                self.assertTrue(npu.is_available.called)


if __name__ == '__main__':
    unittest.main(verbosity=2)
