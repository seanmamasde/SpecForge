#!/usr/bin/env bash
set -euo pipefail

PORT=8000
MODEL_NAME='meta-llama/Llama-4-Maverick-17B-128E'
EXTRA_ARGS=()

while getopts ":eh-:" opt; do
  case "$opt" in
    e)  EXTRA_ARGS+=( \
          --speculative-algorithm EAGLE3 \
          --speculative-draft-model-path nvidia/Llama-4-Maverick-17B-128E-Eagle3 \
          # --speculative-draft-model-path /home/seanma0627/src/SpecForge/outputs/llama4-maverick-17B-128E-eagle3/epoch_1 \
          --speculative-num-steps 2 \
          --speculative-eagle-topk 24 \
          --speculative-num-draft-tokens 128 \
        ) ;;
    h|\?) echo "Usage: $0 [-e|--eagle]" ; exit 0 ;;
    -)  case "$OPTARG" in
          eagle)  EXTRA_ARGS=( \
                    # --speculative-draft-model-path /home/seanma0627/src/SpecForge/outputs/llama4-maverick-17B-128E-eagle3/epoch_1 \
                    --speculative-algorithm EAGLE3 \
                    --speculative-draft-model-path nvidia/Llama-4-Maverick-17B-128E-Eagle3 \
                    --speculative-num-steps 2 \
                    --speculative-eagle-topk 24 \
                    --speculative-num-draft-tokens 128 \
                  ) ;;
          help)  echo "Usage: $0 [-e|--eagle]" ; exit 0 ;;
          *)     echo "Unknown option --$OPTARG" >&2; exit 1 ;;
        esac ;;
  esac
done
shift $((OPTIND-1))

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="/dev/shm/inductor_cache_${USER}_${PORT}"
export XDG_CACHE_HOME="/dev/shm/pycache_${USER}_${PORT}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
singularity exec --nv \
  --bind "$HOME:$HOME" \
  /home/seanma0627/src/sglang.sif \
  python3 -m sglang.launch_server \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --model "meta-llama/Llama-4-Maverick-17B-128E" \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --mem-fraction-static 0.7 \
    --tp 8 \
    --max-total-tokens 131072 \
    --cuda-graph-max-bs 1 \
    "${EXTRA_ARGS[@]}"

    # --disable-custom-all-reduce \
    # --disable-cuda-graph \
    # --enable-mscclpp \
    # --cpu-offload-gb 80 \

# baseline
python3 -m sglang.launch_server --host 0.0.0.0 --port 8000 --model "meta-llama/Llama-4-Maverick-17B-128E" --dtype bfloat16 --attention-backend fa3 --mem-fraction-static 0.7 --tp 8
# nvidia one
python3 -m sglang.launch_server --host 0.0.0.0 --port 8000 --model "meta-llama/Llama-4-Maverick-17B-128E" --dtype bfloat16 --attention-backend fa3 --mem-fraction-static 0.7 --tp 8 --speculative-algorithm EAGLE3 --speculative-draft-model-path nvidia/Llama-4-Maverick-17B-128E-Eagle3 --speculative-num-steps 3 --speculative-eagle-topk 8 --speculative-num-draft-tokens 10
# my shit
python3 -m sglang.launch_server --host 0.0.0.0 --port 8000 --model "meta-llama/Llama-4-Maverick-17B-128E" --dtype bfloat16 --attention-backend fa3 --mem-fraction-static 0.7 --tp 8 --speculative-algorithm EAGLE3 --speculative-draft-model-path "/home/seanma0627/src/SpecForge/outputs/llama4-maverick-17B-128E-eagle3/epoch_1" --speculative-num-steps 3 --speculative-eagle-topk 8 --speculative-num-draft-tokens 10
