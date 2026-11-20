"""No installation and no NPU allocation: check the isolated environment and real input data."""
from importlib import metadata
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import fcntl
from runtime_contract import ISOLATED_PACKAGES, validate_package_versions


def main():
    expected = ISOLATED_PACKAGES
    found = {k: metadata.version(k) for k in expected}
    print(json.dumps({'arch': platform.machine(), 'versions': found}, indent=2), flush=True)
    validate_package_versions(found, expected)
    import torch
    import torch_npu
    if platform.machine() != 'aarch64' or not torch.npu.is_available() or torch.npu.device_count() != 8:
        raise RuntimeError('Expected eight visible Ascend devices on each aarch64 training node')
    print('Node NPU count:', torch.npu.device_count(), flush=True)
    print('VISIBLE_DEVICES', os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<all allocated>'), flush=True)
    for name in ['av', 'cv2', 'pyarrow', 'pandas', 'qwen_vl_utils',
                 'starVLA.dataloader.lerobot_datasets',
                 'starVLA.model.framework.VLM4A.QwenOFT']:
        importlib.import_module(name)
    import transformers
    import accelerate
    import deepspeed
    for name, module in [('torch', torch), ('torch-npu', torch_npu),
                         ('transformers', transformers), ('accelerate', accelerate),
                         ('deepspeed', deepspeed)]:
        print('RUNTIME_MODULE', name, module.__version__, module.__file__, flush=True)
        validate_package_versions({name: module.__version__}, {name: expected[name]})
        if module.__version__.split('+', 1)[0] != found[name].split('+', 1)[0]:
            print('RUNTIME_VERSION_LABEL_DIFF', name, 'metadata=' + found[name],
                  'module=' + module.__version__, 'both admitted to functional validation', flush=True)
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
        # Exercise the real tokenizer/image processor offline, without loading
        # the 3B weights or allocating an NPU context.
        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info
        processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
        processor.tokenizer.padding_side = 'left'
        token_ids = processor.tokenizer('🔍', add_special_tokens=False)['input_ids']
        if len(token_ids) != 1:
            raise ValueError('Action placeholder must tokenize to one ID')
        prompt = sample['lang'] + ' Please predict the next 8 robot actions: <action>' + '🔍' * 8 + '<action>.'
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': im} for im in sample['image']] + [{'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        encoded = processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt')
        if int((encoded['input_ids'] == token_ids[0]).sum()) != 8 or len(encoded['image_grid_thw']) != 2:
            raise ValueError('Real two-camera/action-token processor contract failed')
        print('PROCESSOR_PASSED two_cameras=2 action_tokens=8', flush=True)
        ds.save_dataset_statistics(Path(os.environ['OUTPUT_ROOT']) / os.environ['RUN_NAME'] / f'dataset-preflight-node-{os.environ.get("VC_TASK_INDEX", "0")}.json')
        print('Training reader passed:', len(ds), 'steps; first sample:', sample['lang'], flush=True)
    print('Preflight passed: full input checksums, imports, two cameras and normalized action sample', flush=True)


if __name__ == '__main__':
    main()
