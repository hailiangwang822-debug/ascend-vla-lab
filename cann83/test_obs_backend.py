"""OBS interpreter routing and subprocess protocol without a cloud account."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from obs_backend import InterpreterBackend, PREFIX, select_backend
from obs_state import Store


class ObsBackendTests(unittest.TestCase):
    def test_helper_restores_platform_python_import_environment(self):
        backend = object.__new__(InterpreterBackend)
        backend.python = '/platform/python'
        result = subprocess.CompletedProcess([], 0, PREFIX + '{"ok":true,"value":true}\n', '')
        env = {'PYTHONPATH': '/training/project', 'PYTHONNOUSERSITE': '1',
               'STARVLA_OBS_PYTHONPATH': '/platform/modules', 'STARVLA_OBS_PYTHONNOUSERSITE': ''}
        with patch.dict(os.environ, env), patch('obs_backend.subprocess.run', return_value=result) as call:
            self.assertTrue(backend.exists('/data'))
            helper = call.call_args.kwargs['env']
            self.assertEqual(helper['PYTHONPATH'], '/platform/modules')
            self.assertNotIn('PYTHONNOUSERSITE', helper)

    def test_real_helper_handles_logging_and_literal_paths(self):
        # A fake moxing module exercises the real subprocess, JSON protocol,
        # Unicode/space paths and noisy third-party stdout, without OBS access.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'moxing.py').write_text(
                "print('MoXing startup log')\n"
                "import os, shutil\n"
                "class file:\n"
                "    exists = staticmethod(os.path.exists)\n"
                "    copy = staticmethod(shutil.copyfile)\n", encoding='utf-8')
            source = root / 'input unicode 数据 $literal.txt'
            target = root / 'output with spaces.txt'
            source.write_text('preserved contents', encoding='utf-8')
            with patch.dict(os.environ, {'PYTHONPATH': str(root)}):
                backend = InterpreterBackend(sys.executable)
                self.assertEqual(backend.probe['python'], sys.executable)
                self.assertTrue(backend.exists(source))
                self.assertFalse(backend.exists(target))
                backend.copy(source, target)
                self.assertEqual(target.read_text(encoding='utf-8'), 'preserved contents')
                with self.assertRaisesRegex(RuntimeError, 'FileNotFoundError'):
                    backend.copy(root / 'missing', target)

    def test_helper_failure_does_not_succeed_on_exit_code_alone(self):
        backend = object.__new__(InterpreterBackend)
        backend.python = sys.executable
        bad = [subprocess.CompletedProcess([], 0, 'only logging\n', ''),
               subprocess.CompletedProcess([], 1, PREFIX + '{"ok":true,"value":true}\n', ''),
               subprocess.CompletedProcess([], 0, PREFIX + '{"ok":false,"error":"denied"}\n', '')]
        for result in bad:
            with self.subTest(result=result), patch('obs_backend.subprocess.run', return_value=result):
                with self.assertRaises(RuntimeError):
                    backend.exists('/anything')

    def test_preferred_interpreter_failure_falls_back_to_current_moxing(self):
        file_api = SimpleNamespace(copy=lambda *a: None, exists=lambda p: True)
        fake_mox = SimpleNamespace(file=file_api, __file__='/current/moxing/__init__.py')
        with patch.dict(os.environ, {'STARVLA_OBS_PYTHON': '/unavailable/platform/python'}), \
             patch('obs_backend.InterpreterBackend', side_effect=OSError('missing')), \
             patch('obs_backend.importlib.import_module', return_value=fake_mox), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(select_backend(), file_api)

    def test_no_moxing_fails_at_store_construction(self):
        with patch.dict(os.environ, {'STARVLA_OBS_PYTHON': ''}), \
             patch('obs_backend.importlib.import_module', side_effect=ImportError('moxing missing')):
            with self.assertRaisesRegex(RuntimeError, 'STARVLA_OBS_PYTHON'):
                Store()

    def test_explicit_test_backend_does_not_import_or_spawn(self):
        backend = object()
        with patch('obs_state.select_backend') as select:
            self.assertIs(Store(backend).backend, backend)
            select.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
