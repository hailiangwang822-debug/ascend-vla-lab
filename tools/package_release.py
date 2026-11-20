"""Build an offline release using verified assets already downloaded locally.

Usage: python tools/package_release.py --assets-dir /path/to/asset-staging
The staging directory must contain starvla-spatial.tar, starvla-wheels.tar,
and models/Qwen2.5-VL-3B-Instruct/ with the upstream small config files.
No model weights are bundled. Run from a Git checkout with all release source
files staged or committed; ignored build outputs are excluded.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'v0.1.2'
NOTICE = ('Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, '
          'Copyright (c) Alibaba Cloud. All Rights Reserved.\n')
ATTRIBUTION = '''LIBERO Spatial offline data archive

Original benchmark and authors: https://github.com/Lifelong-Robot-Learning/LIBERO
LeRobot conversion: IPEC-COMMUNITY
https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
Revision: bf14d6258218d12c2e3c1a3b9922e163cdf6455d
License: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/
Changes by Ascend VLA Lab: packaging selected Spatial files into a tar archive;
underlying data and metadata unchanged. No new demonstrations are claimed.
'''


def info(path):
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'filename': path.name, 'bytes': path.stat().st_size, 'sha256': sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path)
    parser.add_argument('--code-only', action='store_true', help='Build the small source update; reuse v0.1.0 assets')
    parser.add_argument('--output', type=Path, default=ROOT / 'release')
    parser.add_argument('--git', default='git')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.code_only:
        if args.assets_dir is None:
            parser.error('--assets-dir is required unless --code-only is selected')
        bundle = json.loads((ROOT / 'cann83/bundle-manifest.json').read_text())
        for row in bundle['archives']:
            path = args.assets_dir / row['filename']
            if info(path) != row:
                raise ValueError('Archive size/hash mismatch: ' + path.name)
            shutil.copyfile(path, args.output / path.name)

    manifest_path = ROOT / 'adaptation-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    paths = {r['path'] for r in manifest['files']} | {
        'run_cloud.sh', 'cann83/requirements.lock', 'cann83/bundle-manifest.json',
        'cann83/assets-manifest.json', 'cann83/runtime_contract.py', 'cann83/obs_backend.py'}
    manifest['files'] = [{'path': p, 'sha256': info(ROOT / p)['sha256']} for p in sorted(paths)]
    manifest['release'] = VERSION
    manifest['validation'] = 'CPU recovery tests passed; actual 64-card NPU/OBS validation pending'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')

    paths = subprocess.check_output([args.git, 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                                    cwd=ROOT).decode('utf-8').split('\0')
    paths = sorted({p for p in paths if p})
    code = args.output / f'ascend-vla-{VERSION}-code.zip'
    with zipfile.ZipFile(code, 'w', zipfile.ZIP_DEFLATED) as z:
        for relative in paths:
            p = ROOT / relative
            if p.suffix == '.py':
                ast.parse(p.read_text(encoding='utf-8'), filename=relative)
            if p.stat().st_size >= 100 * 1024**2:
                raise ValueError('Large asset accidentally added to source: ' + relative)
            z.write(p, 'A/' + relative)

    if args.code_only:
        with zipfile.ZipFile(code) as z:
            if z.testzip() is not None:
                raise ValueError('Corrupt code ZIP')
        report = {'version': VERSION, 'npu_training_tested': False,
                  'reuse_assets_from': 'v0.1.0', 'files': [info(code)]}
        (args.output / 'code-manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))
        return

    configs = args.output / 'qwen25-vl-3b-configs.zip'
    assets = json.loads((ROOT / 'cann83/assets-manifest.json').read_text())
    with zipfile.ZipFile(configs, 'w', zipfile.ZIP_DEFLATED) as z:
        for row in assets['files']:
            if row['group'] != 'model' or row['path'].endswith('.safetensors'):
                continue
            p = args.assets_dir / row['path']
            checked = info(p)
            if checked['bytes'] != row['bytes'] or checked['sha256'] != row['hash']:
                raise ValueError('Config mismatch: ' + row['path'])
            z.write(p, row['path'].split('/', 1)[1])
        z.writestr('Qwen2.5-VL-3B-Instruct/Notice', NOTICE)

    shutil.copyfile(ROOT / 'tools/download_internvl_shard1.py', args.output / 'download_internvl_shard1.py')
    shutil.copyfile(ROOT / 'docs/ascend/YELLOW_ZONE_RUNBOOK_CN.md', args.output / '64卡上传与启动说明.md')
    shutil.copyfile(ROOT / 'THIRD_PARTY_NOTICES.md', args.output / 'THIRD_PARTY_NOTICES.md')
    (args.output / 'DATASET_ATTRIBUTION.txt').write_text(ATTRIBUTION, encoding='utf-8')
    names = [code.name, configs.name, 'starvla-spatial.tar', 'starvla-wheels.tar',
             'download_internvl_shard1.py', '64卡上传与启动说明.md',
             'THIRD_PARTY_NOTICES.md', 'DATASET_ATTRIBUTION.txt']
    report = {'version': VERSION, 'npu_training_tested': False,
              'files': [info(args.output / name) for name in names]}
    (args.output / 'delivery-manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    names += ['delivery-manifest.json']
    transfer = args.output / f'ascend-vla-{VERSION}-transfer.zip'
    with zipfile.ZipFile(transfer, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name in names:
            z.write(args.output / name, 'ascend-vla-ready/' + name)
    for archive in (code, configs, transfer):
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError('Corrupt zip: ' + archive.name)
    report['transfer_bundle'] = info(transfer)
    (args.output / 'release-manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (args.output / 'SHA256SUMS.txt').write_text('\n'.join(
        info(args.output / name)['sha256'] + '  ' + name
        for name in [transfer.name, code.name, 'release-manifest.json']) + '\n', encoding='utf-8')
    print(json.dumps(report['transfer_bundle'], indent=2))


if __name__ == '__main__':
    main()
