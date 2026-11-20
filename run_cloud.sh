#!/usr/bin/env bash
set -eo pipefail
mode="${1:-preview}"
case "${mode}" in preview|train) ;; *) echo 'Usage: bash run_cloud.sh preview|train'; exit 2 ;; esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${mode}" == train ]]; then
    # The old successful package used the platform Python for MoXing before
    # activating the training conda environment. Preserve that interpreter.
    export STARVLA_OBS_PYTHON="${STARVLA_OBS_PYTHON:-${BOOTSTRAP_PYTHON:-$(command -v python)}}"
    export STARVLA_OBS_PYTHONPATH="${PYTHONPATH:-}"
    export STARVLA_OBS_PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-}"
    source "${CONDA_SH:-/home/ma-user/anaconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-PyTorch-2.7.1}"
    if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
        source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
    else
        echo 'CANN set_env.sh missing'; exit 2
    fi
    # Match the library paths used by the previous successful VLM job.
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/openblas/lib:/usr/local/lib:/usr/lib64:/usr/lib:/usr/lib/aarch64_64-linux-gnu"
    export PYTHONNOUSERSITE=1
fi
exec python -u "${script_dir}/cann83/bootstrap.py" "${mode}"
