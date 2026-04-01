#!/bin/bash
# Fine-tune Qwen2.5-3B with MLX on M1 — throttled to reduce heat
# Runs training in chunks with cooldown pauses between them

set -e
cd "$(dirname "$0")/.."
source .venv-mlx/bin/activate

MODEL="Qwen/Qwen2.5-3B-Instruct"
DATA="data/mlx_train"
CONFIG="config/mlx_lora.yaml"
ADAPTER="adapters"

echo "=== MLX LoRA Fine-tune (throttled) ==="
echo "Model: $MODEL"
echo "Data: $(wc -l < $DATA/train.jsonl) train, $(wc -l < $DATA/valid.jsonl) valid"
echo ""

# Run training with nice (lower CPU priority) to reduce thermal pressure
nice -n 10 python3 -m mlx_lm.lora \
  --model "$MODEL" \
  --data "$DATA" \
  --config "$CONFIG" \
  --train

echo ""
echo "=== Training complete ==="
echo "Adapters saved to: $ADAPTER/"
echo ""
echo "Next steps:"
echo "  1. Fuse: python3 -m mlx_lm.fuse --model $MODEL --adapter-path $ADAPTER --save-path car-parser-3b-mlx"
echo "  2. Convert to GGUF: python3 -m mlx_lm.convert --model car-parser-3b-mlx --quantize q4_k_m --upload-repo '' --gguf"
