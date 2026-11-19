# Third-party sources and attribution

## StarVLA

Source: https://github.com/starVLA/starVLA

Pinned commit: `261ba5dd4313bbb910a9caeea07433c3d7b1aa34`.
Original source archive SHA256: `f6e38637964f0c12c6d98f939569ed5c8738692c2d24e8f780aa78690edd9fda`.

Copyright (c) StarVLA Team. The original LICENSE, CITATION.cff, README copy and Git ancestry are preserved. This repository adds Ascend/ModelArts integration; it does not claim authorship of the StarVLA framework or QwenOFT method. See `adaptation-manifest.json` and Git diff from the pinned commit for modifications.

## Qwen2.5-VL-3B-Instruct

Source: https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct

Pinned ModelScope revision: `1b5a7674d40a0ac9acef541a46fed12d710bd36b`.

The upstream 3B distribution includes the **Qwen RESEARCH LICENSE AGREEMENT**, release date September 19, 2024. Its research/non-commercial terms and commercial licensing conditions apply independently of this repository's code license. Read the supplied model LICENSE before using the model. We distribute the original small configuration/tokenizer files and LICENSE; model weights are downloaded directly from the official repository.

Required notice:

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.

## LIBERO Spatial data

Original benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

LeRobot conversion: https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot

Pinned revision: `bf14d6258218d12c2e3c1a3b9922e163cdf6455d`.

LIBERO data is released under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ . Credit the LIBERO authors and IPEC-COMMUNITY conversion maintainers. The release redistributes the selected Spatial conversion as a tar archive with its data and metadata unchanged; no new demonstrations are claimed. A `DATASET_ATTRIBUTION.txt` accompanies the archive.

## Offline Python wheels

The release contains unmodified CPython3.11/aarch64 or universal wheel distributions. Each retains its own embedded license and metadata. `wheelhouse/manifest.json` inside `starvla-wheels.tar` records package names, versions, source URLs and SHA256. These dependencies are not relicensed by this project.

## Requests downloader

`tools/download_internvl_shard1.py` is the resumable downloader provided in the project workflow. Its original filename and default example refer to InternVL; the documented Qwen commands supply URL, output path, size and checksum explicitly.
