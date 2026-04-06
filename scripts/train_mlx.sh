#!/bin/bash
# Fine-tune Qwen2.5-3B with MLX on M1 8GB — thermal-friendly
# Trains in 500-iter chunks with 2-min cooldown between them
# Total: ~5200 iters (3 epochs), ~7-8 hours
#
# NOTE: mlx_lm resets iter counter on resume, so each chunk always
# runs exactly CHUNK iterations. Progress tracked via .chunk_progress file.

set -e
cd "$(dirname "$0")/.."
source .venv-mlx/bin/activate

MODEL="Qwen/Qwen2.5-3B-Instruct"
DATA="data/mlx_train"
CONFIG="config/mlx_lora.yaml"
ADAPTER="adapters"
PROGRESS="$ADAPTER/.chunk_progress"

CHUNK=500         # iterations per chunk (each chunk = CHUNK new iters)
TOTAL=5200        # total iterations (~3 epochs over 1710 examples)
COOLDOWN=120      # seconds between chunks (2 min cooldown)

TOTAL_CHUNKS=$(( (TOTAL + CHUNK - 1) / CHUNK ))

echo "=== MLX LoRA Fine-tune (thermal-friendly) ==="
echo "Model: $MODEL"
echo "Data: $(wc -l < $DATA/train.jsonl) train, $(wc -l < $DATA/valid.jsonl) valid"
echo "Plan: $TOTAL iters in $TOTAL_CHUNKS chunks of $CHUNK, ${COOLDOWN}s cooldown"
echo "LoRA: rank=16, layers=16, modules=q/k/v/o, grad_checkpoint=on"
echo "Started: $(date '+%Y-%m-%d %H:%M')"
echo ""

# Detect resume point
DONE_CHUNKS=0
if [ -f "$PROGRESS" ]; then
    DONE_CHUNKS=$(cat "$PROGRESS")
    echo "Resuming after chunk $DONE_CHUNKS/$TOTAL_CHUNKS ($(( DONE_CHUNKS * CHUNK )) iters done)"
else
    if [ -d "$ADAPTER" ] && [ ! -f "$ADAPTER/adapters.safetensors" ]; then
        rm -rf "$ADAPTER"
    fi
    echo "Fresh start"
fi
mkdir -p "$ADAPTER"
echo ""

for ((chunk=DONE_CHUNKS+1; chunk<=TOTAL_CHUNKS; chunk++)); do
    ITERS=$CHUNK
    # Last chunk may be shorter
    REMAINING=$(( TOTAL - (chunk - 1) * CHUNK ))
    if [ $REMAINING -lt $CHUNK ]; then ITERS=$REMAINING; fi

    DONE_ITERS=$(( (chunk - 1) * CHUNK ))
    TARGET_ITERS=$(( DONE_ITERS + ITERS ))

    RESUME=""
    if [ -f "$ADAPTER/adapters.safetensors" ]; then
        RESUME="--resume-adapter-file $ADAPTER/adapters.safetensors"
    fi

    echo "[$(date '+%H:%M')] Chunk $chunk/$TOTAL_CHUNKS — total iters $DONE_ITERS → $TARGET_ITERS"

    nice -n 15 python3 -m mlx_lm.lora \
        --model "$MODEL" \
        --data "$DATA" \
        --config "$CONFIG" \
        --iters $ITERS \
        $RESUME \
        --train

    echo "$chunk" > "$PROGRESS"

    if [ $chunk -lt $TOTAL_CHUNKS ]; then
        echo "[$(date '+%H:%M')] Cooldown ${COOLDOWN}s..."
        sleep $COOLDOWN
    fi
done

echo ""
echo "=== Training complete — $TOTAL iterations ==="
echo "Finished: $(date '+%Y-%m-%d %H:%M')"
echo "Adapters: $ADAPTER/"
echo ""
echo "Next: bash scripts/deploy_model.sh"
