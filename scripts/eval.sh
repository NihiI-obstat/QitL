#!/usr/bin/env bash
# Evaluate LLaVA-Video-7B-Qwen2 with QitL on video benchmarks (lmms-eval, 64 frames).
#
#   RETENTION_RATIO=0.25 TASKS=mvbench bash scripts/eval.sh
#   RETENTION_RATIO=1.0  TASKS=videomme bash scripts/eval.sh            # uncompressed model
#   CUDA_VISIBLE_DEVICES=0,1,2,3 RETENTION_RATIO=0.1 TASKS=mlvu_dev bash scripts/eval.sh
#
# RETENTION_RATIO  fraction of visual patch tokens kept in every frame (default 0.25); 1.0 runs the original model
# TASKS            comma-separated: mvbench, videomme, longvideobench_val_v, mlvu_dev
# MODEL_PATH       local checkpoint or Hugging Face id (default lmms-lab/LLaVA-Video-7B-Qwen2)
# OUTPUT_DIR       where results and per-sample logs go (default results/<tasks>_r<ratio>)
# LIMIT            evaluate only the first LIMIT questions of every (sub)task, for a quick check
# CUDA_VISIBLE_DEVICES  one evaluation process per listed GPU; questions are split across them
# LLAVA_VID_LOADER_WORKERS  CPU threads that decode upcoming videos while the GPU runs (default 8)
# LLAVA_VID_FRAME_CACHE     optional directory for decoded frames, shared by runs on the same benchmark
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RETENTION_RATIO="${RETENTION_RATIO:-0.25}"
TASKS="${TASKS:-mvbench}"
MODEL_PATH="${MODEL_PATH:-lmms-lab/LLaVA-Video-7B-Qwen2}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/${TASKS//,/_}_r${RETENTION_RATIO}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
export PYTHONPATH="${ROOT}/LLaVA-NeXT:${ROOT}/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"
export LLAVA_VID_LOADER_WORKERS="${LLAVA_VID_LOADER_WORKERS:-8}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"  # keeps memory flat while many threads decode long videos
export TOKENIZERS_PARALLELISM=false

# Official LLaVA-Video evaluation settings: 64 evenly spaced frames, 2x2 average pooling, a newline token per row.
MODEL_ARGS="pretrained=${MODEL_PATH},conv_template=qwen_1_5,max_frames_num=64,force_sample=True"
MODEL_ARGS+=",mm_spatial_pool_mode=average,mm_spatial_pool_stride=2,mm_newline_position=grid"
MODEL_ARGS+=",video_decode_backend=decord,attn_implementation=sdpa,retention_ratio=${RETENTION_RATIO}"

ARGS=(--model llava_vid --model_args "${MODEL_ARGS}" --tasks "${TASKS}" --batch_size 1 --log_samples --output_path "${OUTPUT_DIR}")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "${LIMIT}")

echo "Tasks: ${TASKS}  retention ratio: ${RETENTION_RATIO}  GPUs: ${CUDA_VISIBLE_DEVICES}  output: ${OUTPUT_DIR}"
cd "${ROOT}/lmms-eval"
exec python -m accelerate.commands.launch --num_processes "${#GPUS[@]}" --main_process_port "$((RANDOM % 10000 + 20000))" \
    -m lmms_eval "${ARGS[@]}" "$@"
