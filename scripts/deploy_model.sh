#!/bin/bash
# Fuse adapters → GGUF Q4_K_M → deploy to Ollama on remote server
set -e
cd "$(dirname "$0")/.."
source .venv-mlx/bin/activate

MODEL="Qwen/Qwen2.5-3B-Instruct"
ADAPTER="adapters"
FUSED="car-parser-3b-fused"
SERVER="anastasia@192.168.1.74"

echo "=== 1. Fuse adapters ==="
rm -rf "$FUSED"
python3 -m mlx_lm.fuse \
    --model "$MODEL" \
    --adapter-path "$ADAPTER" \
    --save-path "$FUSED"

echo ""
echo "=== 2. Convert to GGUF F16 ==="
# Patch convert script for gguf 0.18 compatibility
python3 -c "
code = open('/opt/homebrew/bin/convert_hf_to_gguf.py').read()
code = code.replace('gguf.MODEL_ARCH.MISTRAL4', 'gguf.MODEL_ARCH.MISTRAL3')
code = code.replace('gguf.MODEL_TENSOR_ARCH.MISTRAL4', 'gguf.MODEL_TENSOR_ARCH.MISTRAL3')
open('/tmp/convert_hf_to_gguf_patched.py','w').write(code)
"
python3 /tmp/convert_hf_to_gguf_patched.py "$FUSED" \
    --outfile car-parser-3b-f16.gguf --outtype f16

echo ""
echo "=== 3. Quantize to Q4_K_M ==="
llama-quantize car-parser-3b-f16.gguf car-parser-3b-q4km.gguf Q4_K_M
rm car-parser-3b-f16.gguf

echo ""
echo "=== 4. Deploy to server ==="
scp car-parser-3b-q4km.gguf Modelfile "$SERVER":~/olx-car-parser/
ssh "$SERVER" "cd ~/olx-car-parser && /Applications/Ollama.app/Contents/Resources/ollama create car-parser-3b -f Modelfile"

echo ""
echo "=== Done! ==="
echo "Test: ssh $SERVER 'curl -s http://localhost:11434/api/chat -d ...'"
