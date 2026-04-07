#!/bin/bash
# Setup Ollama on macOS server for local LLM enrichment.
# Run once on the server: bash scripts/setup-ollama.sh [model]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('${SCRIPT_DIR}/../config/settings.yaml'))['llm']['ollama_model'])" 2>/dev/null || echo "qwen2.5:1.5b")
MODEL="${1:-$DEFAULT_MODEL}"
OLLAMA_BIN="/Applications/Ollama.app/Contents/Resources/ollama"
PLIST_NAME="com.olx-car-parser.ollama"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"

echo "=== Ollama Setup for OLX Car Parser ==="

# 1. Install Ollama if not present
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "Downloading Ollama..."
    curl -fsSL https://ollama.com/download/Ollama-darwin.zip -o /tmp/Ollama-darwin.zip
    unzip -o /tmp/Ollama-darwin.zip -d /tmp/Ollama
    cp -R /tmp/Ollama/Ollama.app /Applications/Ollama.app
    rm -rf /tmp/Ollama /tmp/Ollama-darwin.zip
    echo "Installed Ollama to /Applications/Ollama.app"
else
    echo "Ollama already installed: $($OLLAMA_BIN --version 2>&1 | tail -1)"
fi

# 2. Start Ollama temporarily to pull the model
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama server..."
    nohup "$OLLAMA_BIN" serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi

# 3. Pull the model
echo "Pulling model: ${MODEL} ..."
"$OLLAMA_BIN" pull "$MODEL"

# 4. Create LaunchAgent for auto-start on boot
echo "Creating LaunchAgent: ${PLIST_PATH}"
mkdir -p "$(dirname "$PLIST_PATH")"
cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${OLLAMA_BIN}</string>
        <string>serve</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>OLLAMA_NUM_PARALLEL</key>
        <string>3</string>
        <key>OLLAMA_FLASH_ATTENTION</key>
        <string>1</string>
    </dict>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/ollama.out.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/ollama.err.log</string>
</dict>
</plist>
EOF

# 5. Kill GUI Ollama (it ignores plist env), reload via LaunchAgent
pkill -9 -f "Ollama" 2>/dev/null || true
sleep 2
launchctl bootout gui/$(id -u) "$PLIST_PATH" 2>/dev/null || true
sleep 1
launchctl bootstrap gui/$(id -u) "$PLIST_PATH"

# 6. Verify
sleep 2
if curl -sf http://localhost:11434/api/tags | python3 -c "import sys,json; m=json.load(sys.stdin)['models']; print('Models:', [x['name'] for x in m])" 2>/dev/null; then
    echo ""
    echo "=== SUCCESS ==="
    echo "Ollama running with model: ${MODEL}"
    echo "API: http://localhost:11434"
    echo "Auto-starts on boot via LaunchAgent"
else
    echo ""
    echo "=== WARNING ==="
    echo "Check: curl http://localhost:11434/api/tags"
fi
