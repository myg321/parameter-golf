#!/bin/bash
set -euo pipefail

export SEED=${SEED:-1}
export BASE_MODEL=${BASE_MODEL:-/storage/yiguang/all_models/Qwen3-VL-2B-Instruct}
# export BASE_MODEL=${BASE_MODEL:-/storage/data/shiyd2023/LLM_model/Qwen3-VL-2B-Instruct}
export ADAPTER=${1:-${ADAPTER:-./weights}}
export MERGED_MODEL=${2:-${MERGED_MODEL:-./outputs/textvqa_qwen3vl_lora_seed${SEED}/merged}}
export MERGE_DTYPE=${MERGE_DTYPE:-float16}

python merge_lora.py \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --output "${MERGED_MODEL}" \
  --dtype "${MERGE_DTYPE}"
