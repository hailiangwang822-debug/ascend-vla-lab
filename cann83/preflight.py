"""No installation and no NPU allocation: check the isolated environment and real input data."""
from importlib import metadata
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import fcntl


def main():
    expected = {'torch': '2.7.1', 'torch-npu': '2.7.1.post2', 'torchvision': '0.22.1',
                'accelerate': '1.11.0', 'deepspeed': '0.18.4', 'numpy': '1.26.4',
                'transformers': '4.57.1', 'pydantic': '2.10.6', 'numpydantic': '1.6.9'}
    found = {k: metadata.version(k) for k in expected}
    print(json.dumps({'arch': platform.machine(), 'versions': found}, indent=2), flush=True)
    for k, v in expected.items():
        if found[k].split('+', 1)[0] != v:
            raise RuntimeError(f'{k}: expected {v}, got {found[k]}')
    import torch
    import torch_npu
    if platform.machine() != 'aarch64' or not torch.npu.is_available() or torch.npu.device_count() != 8:
        raise RuntimeError('Expected eight visible Ascend devices on each aarch64 training node')
    print('Node NPU count:', torch.npu.device_count(), flush=True)
    for name in ['av', 'cv2', 'pyarrow', 'pandas', 'qwen_vl_utils',
                 'starVLA.dataloader.lerobot_datasets',
                 'starVLA.model.framework.VLM4A.QwenOFT']:
        importlib.import_module(name)
    model = Path(os.environ['MODEL_DIR'])
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    for name in set(index['weight_map'].values()):
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)
    root = Path(__file__).resolve().parents[1]
    dataset = Path(os.environ['DATA_ROOT']) / 'libero_spatial_no_noops_1.0.0_lerobot'
    info = json.loads((dataset / 'meta/info.json').read_text())
    if info['total_episodes'] != 432:
        raise ValueError('Expected complete 432-episode spatial dataset')
    modality = dataset / 'meta/modality.json'
    reference = root / 'examples/LIBERO/train_files/modality.json'
    if modality.exists() and modality.read_bytes() != reference.read_bytes():
        raise ValueError('Existing modality differs from the upstream LIBERO mapping; review before replacement')
    if not modality.exists():
        shutil.copyfile(reference, modality)
    from download_assets import valid
    asset_manifest = json.loads((root / 'cann83/assets-manifest.json').read_text())
    for row in asset_manifest['files']:
        relative = row['path'].split('/', 2)[-1] if row['group'] == 'model' else row['path'].split('/', 1)[-1]
        p = model / relative if row['group'] == 'model' else Path(os.environ['DATA_ROOT']) / relative
        if not valid(p, row):
            raise ValueError('Input asset missing or checksum mismatch: ' + str(p))
    import pandas as pd
    import av
    p = dataset / 'data/chunk-000/episode_000000.parquet'
    frame = pd.read_parquet(p)
    assert len(frame) > 8 and 'action' in frame.columns
    for key, spec in info['features'].items():
        if spec['dtype'] == 'video':
            p = dataset / info['video_path'].format(episode_chunk=0, episode_index=0, video_key=key)
            with av.open(str(p)) as container:
                image = next(container.decode(video=0)).to_ndarray(format='rgb24')
            assert image.shape == (256, 256, 3)
    from omegaconf import OmegaConf
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    # Build caches before distributed startup: upstream assumes rank 0's cache
    # is visible to every rank, which is false for node-local /cache.
    # The lock also protects shared SFS datasets if nodes use the same mount.
    with (dataset / 'meta/.starvla-prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cfg = OmegaConf.load(root / 'cann83/train.yaml')
        ds = get_vla_dataset(cfg.datasets.vla_data)
        sample = ds[0]
        import numpy as np
        assert sample['action'].shape == (8, 7)
        assert np.isfinite(sample['action']).all()
        assert len(sample['image']) == 2 and sample['lang']
        ds.save_dataset_statistics(Path(os.environ['OUTPUT_ROOT']) / os.environ['RUN_NAME'] / f'dataset-preflight-node-{os.environ.get("VC_TASK_INDEX", "0")}.json')
        print('Training reader passed:', len(ds), 'steps; first sample:', sample['lang'], flush=True)
    print('Preflight passed: full input checksums, imports, two cameras and normalized action sample', flush=True)


if __name__ == '__main__':
    main()
