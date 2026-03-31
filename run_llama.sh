#!/bin/bash
# Run Llama 3.2 1B inference on IRON NPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS="/home/jiajli/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/model.safetensors"
TOKENIZER="/home/jiajli/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/original/tokenizer.model"

# source "$SCRIPT_DIR/ironenv/bin/activate"
# source /opt/xilinx/xrt/setup.sh 2>/dev/null

NUM_TOKENS="100"
PROMPT_LEN="2048"

set -x 
python "$SCRIPT_DIR/iron/applications/llama_3.2_1b/inference.py" \
    "$WEIGHTS" \
    "$TOKENIZER" \
    --num_tokens ${NUM_TOKENS} \
    --prompt_len ${PROMPT_LEN} \
    -vv
    # "$@"
