"""Checks for the existing ModelArts image; never install or select a device."""
from importlib import metadata
import json
from pathlib import Path
import platform
import sys


BASE_PACKAGES = {'torch': '2.7.1', 'torch-npu': '2.7.1.post2',
                 'torchvision': '0.22.1', 'accelerate': '1.11.0',
                 'deepspeed': '0.18.4'}
# The inspected running VLM image reported post2; the actual submitted VLA
# container reports 2.7.1. Admit both observed builds to FUNCTIONAL validation.
# This is not a claim that their kernels or checkpoint runtimes are identical.
VERSION_VARIANTS = {'torch-npu': ('2.7.1', '2.7.1.post2')}
ISOLATED_PACKAGES = {**BASE_PACKAGES, 'numpy': '1.26.4',
                     'transformers': '4.57.1', 'pydantic': '2.10.6',
                     'numpydantic': '1.6.9'}


def validate_package_versions(versions, expected):
    for name, preferred in expected.items():
        actual = str(versions.get(name, 'missing'))
        allowed = VERSION_VARIANTS.get(name, (preferred,))
        if actual.split('+', 1)[0] not in allowed:
            raise RuntimeError(f'{name}: accepted {" or ".join(allowed)}, found {actual}; inspect the actual image before changing platform packages.')


def validate_node_entry(env):
    if env.get('MA_RUN_METHOD', '').lower() in ('torchrun', 'msrun') or 'LOCAL_RANK' in env:
        raise ValueError('Use the same custom bash launch mode as the old job: one entry per node. Do not nest torchrun/msrun.')
    # RANK_TABLE_FILE, ASCEND_DEVICE_ID and MA_LOCAL_TASK_INDEX may be inherited
    # platform metadata. They do not establish a local worker-launch contract.
    # A node-local flock guards against two simultaneous launchers instead.


def clean_child_environment(env):
    validate_node_entry(env)
    child = dict(env)
    for name in ('RANK_TABLE_FILE', 'RANK_ID', 'RANK_SIZE', 'DEVICE_ID',
                 'ASCEND_DEVICE_ID', 'RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                 'MA_LOCAL_TASK_INDEX'):
        child.pop(name, None)
    return child


def validate_base_versions(versions, python_version, arch):
    if tuple(python_version[:2]) != (3, 11) or arch != 'aarch64':
        raise RuntimeError('Offline wheels require aarch64 Python 3.11; check CONDA_ENV and BOOTSTRAP_PYTHON above.')
    validate_package_versions(versions, BASE_PACKAGES)


def check_base_runtime(root):
    versions = {}
    for name in BASE_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = 'missing'
    print('BASE_RUNTIME', json.dumps({'python': sys.version, 'executable': sys.executable,
          'arch': platform.machine(), 'versions': versions}), flush=True)
    cann = Path('/usr/local/Ascend/ascend-toolkit/latest/version.cfg')
    cann_text = cann.read_text() if cann.is_file() else '<missing>'
    print('CANN_RUNTIME', str(cann), cann_text, flush=True)
    validate_base_versions(versions, sys.version_info, platform.machine())
    if '8.3.RC1' not in cann_text:
        raise RuntimeError('Expected CANN 8.3.RC1 in ascend-toolkit/latest/version.cfg')
    from obs_state import digest
    manifest = json.loads((root / 'adaptation-manifest.json').read_text())
    for row in manifest['files']:
        path = root / row['path']
        if not path.is_file() or digest(path) != row['sha256']:
            raise RuntimeError('Mixed or modified code package: ' + row['path'] + '; upload the entire corrected A directory.')
    print('BASE_RUNTIME_PASSED', manifest['release'], flush=True)
