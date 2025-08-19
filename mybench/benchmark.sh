#!/usr/bin/env bash
set -euo pipefail

PORT=8000
ENDPOINT="/v1/chat/completions"

while getopts ":e-:h" opt; do
  case "${opt}" in
    e) PORT=8001 ;;
    -)
      case "${OPTARG}" in
        eagle) PORT=8001 ;;
        *) echo "Unknown option --${OPTARG}" >&2; exit 1 ;;
      esac ;;
    h|\?) echo "Usage: $0 [-e|--eagle]"; exit 0 ;;
  esac
done
shift $((OPTIND - 1))


MODEL="/home/seanma0627/.cache/huggingface/hub/models--meta-llama--Llama-4-Maverick-17B-128E/snapshots/10751cb97a4d7c90f7ed89196b98eb8220cfa1c2"
REPO=/home/seanma0627/src/eagle3
DATA="/home/seanma0627/.cache/huggingface/hub/datasets--Aeala--ShareGPT_Vicuna_unfiltered/snapshots/8b0048ad6ae8c22f46a78c15559dec98feef5539/ShareGPT_V4.3_unfiltered_cleaned_split.json"
BASE_URL="http://127.0.0.1:${PORT}"

python3 /home/seanma0627/src/vllm/benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --base-url "${BASE_URL}" \
  --endpoint "${ENDPOINT}" \
  --model "${MODEL}" \
  --tokenizer "${MODEL}" \
  --dataset-name sharegpt \
  --dataset-path "${DATA}" \
  --num-prompts 1

# testing commands
# vllm
python3 /home/seanma0627/src/vllm/benchmarks/benchmark_serving.py --backend openai-chat --base-url 'http://127.0.0.1:8000' --endpoint /v1/chat/completions --model '../cache/meta-llama/Llama-4-Maverick-17B-128E' --tokenizer '../cache/meta-llama/Llama-4-Maverick-17B-128E' --dataset-name sharegpt --dataset-path "/home/seanma0627/.cache/huggingface/hub/datasets--Aeala--ShareGPT_Vicuna_unfiltered/snapshots/8b0048ad6ae8c22f46a78c15559dec98feef5539/ShareGPT_V4.3_unfiltered_cleaned_split.json" --num-prompts 1
# sglang
python scripts/playground/bench_speculative.py --model-path ../cache/meta-llama/Llama-4-Maverick-17B-128E --speculative-draft-model-path nvidia/Llama-4-Maverick-17B-128E-Eagle3 --steps 3 4 5 --topk 8 --num_draft_tokens 10 --batch-size 1 2 4 --trust-remote-code
