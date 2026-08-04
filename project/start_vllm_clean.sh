#!/usr/bin/env bash
source /opt/conda/etc/profile.d/conda.sh
conda activate vllm_clean
export LD_LIBRARY_PATH=/opt/conda/envs/vllm_clean/lib:$LD_LIBRARY_PATH
export VLLM_DISABLE_COMPILE_CACHE=1
CUDA_VISIBLE_DEVICES=0,1 vllm serve /model-storage/model/Qwen3.5-35B-A3B-FP8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --enforce-eager \
  --disable-custom-all-reduce \
  --max-logprobs 20