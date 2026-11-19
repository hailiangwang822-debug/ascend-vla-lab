#!/usr/bin/env bash
set -eo pipefail
mode="${1:-preview}"
case "${mode}" in preview|train) ;; *) echo 'Usage: bash run_cloud.sh preview|train'; exit 2 ;; esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${mode}" == train ]]; then
    source "${CONDA_SH:-/home/ma-user/anaconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-PyTorch-2.7.1}"
    if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
        source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
    else
        echo 'CANN set_env.sh missing'; exit 2
    fi
fi
exec python -u "${script_dir}/cann83/bootstrap.py" "${mode}"
