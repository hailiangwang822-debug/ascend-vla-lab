"""MoXing bridge for the platform Python, independent of the training venv.

The helper uses only Python 3.9-compatible standard-library code before loading
MoXing. A platform interpreter does not need the training requirements installed.
"""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


PREFIX = 'STARVLA_OBS_RESULT '


class InterpreterBackend:
    def __init__(self, python):
        self.python = str(python)
        self.probe = self._call('probe')

    def _call(self, operation, **values):
        request = {'operation': operation, **values}
        helper_env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        for key in ('PYTHONPATH', 'PYTHONNOUSERSITE'):
            captured = 'STARVLA_OBS_' + key
            if captured in helper_env:
                if helper_env[captured]:
                    helper_env[key] = helper_env[captured]
                else:
                    helper_env.pop(key, None)
        result = subprocess.run(
            [self.python, '-u', str(Path(__file__).resolve()), '--helper'],
            input=json.dumps(request), text=True, encoding='utf-8',
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, env=helper_env,
        )
        records = [line[len(PREFIX):] for line in result.stdout.splitlines()
                   if line.startswith(PREFIX)]
        if not records:
            raise RuntimeError('MoXing helper returned no result (exit %s, Python %s)' %
                               (result.returncode, self.python))
        try:
            response = json.loads(records[-1])
        except (ValueError, TypeError) as exc:
            raise RuntimeError('Invalid MoXing helper result') from exc
        if result.returncode != 0 or not response.get('ok'):
            raise RuntimeError('MoXing %s failed in %s: %s: %s' %
                               (operation, self.python, response.get('error_type', 'unknown'),
                                response.get('error', 'nonzero exit')))
        return response['value']

    def exists(self, path):
        value = self._call('exists', path=str(path))
        if not isinstance(value, bool):
            raise RuntimeError('MoXing exists returned a non-boolean result')
        return value

    def copy(self, src, dst):
        self._call('copy', src=str(src), dst=str(dst))


def select_backend():
    preferred = os.environ.get('STARVLA_OBS_PYTHON', '')
    preferred_error = None
    if preferred and Path(preferred).resolve() != Path(sys.executable).resolve():
        try:
            backend = InterpreterBackend(preferred)
        except Exception as exc:
            preferred_error = exc
            print('OBS_PLATFORM_PYTHON_UNAVAILABLE; checking training interpreter', flush=True)
        else:
            print('OBS_BACKEND', json.dumps(backend.probe), flush=True)
            return backend
    try:
        mox = importlib.import_module('moxing')
        backend = mox.file
        if not callable(backend.copy) or not callable(backend.exists):
            raise RuntimeError('MoXing file backend is incomplete')
    except Exception as exc:
        message = ('MoXing is unavailable in the selected platform and training interpreters. '
                   'Set STARVLA_OBS_PYTHON to the original ModelArts Python that imports moxing. '
                   'Training interpreter: ' + sys.executable)
        if preferred_error is not None:
            message += '; platform probe: ' + str(preferred_error)
        raise RuntimeError(message) from exc
    print('OBS_BACKEND', json.dumps({'python': sys.executable, 'moxing': mox.__file__}), flush=True)
    return backend


def helper_main():
    try:
        request = json.load(sys.stdin)
        operation = request['operation']
        if operation not in ('probe', 'exists', 'copy'):
            raise ValueError('Unsupported MoXing operation')
        mox = importlib.import_module('moxing')
        if operation == 'probe':
            # Fail early even when importing the module did not expose file I/O.
            if not callable(mox.file.copy) or not callable(mox.file.exists):
                raise RuntimeError('MoXing file backend is incomplete')
            value = {'python': sys.executable, 'moxing': mox.__file__}
        elif operation == 'exists':
            value = bool(mox.file.exists(request['path']))
        else:
            mox.file.copy(request['src'], request['dst'])
            value = None
        response = {'ok': True, 'value': value}
        code = 0
    except Exception as exc:
        response = {'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)}
        code = 1
    print(PREFIX + json.dumps(response, ensure_ascii=True), flush=True)
    return code


if __name__ == '__main__':
    if sys.argv[1:] != ['--helper']:
        raise SystemExit('This module is an internal OBS helper')
    raise SystemExit(helper_main())
