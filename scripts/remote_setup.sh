#!/usr/bin/env bash
# Runs ON the remote server (macOS) — sets up venv, cron, launchd dashboard.
set -euo pipefail

# Ensure Homebrew is in PATH
eval "$(/opt/homebrew/bin/brew shellenv zsh 2>/dev/null)" || true

PROJECT_DIR="$HOME/olx-car-parser"
VENV="$PROJECT_DIR/.venv"
LOG="$PROJECT_DIR/data/scrape.log"
PYTHON="/opt/homebrew/bin/python3.11"

echo "==> Setting up olx-car-parser on $(hostname) ..."

# --- Ensure data dirs exist ---
mkdir -p "$PROJECT_DIR/data/raw" "$PROJECT_DIR/data/processed"

# --- Python venv with 3.11 ---
if [ ! -d "$VENV" ]; then
    echo "Creating virtualenv with Python 3.11..."
    "$PYTHON" -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install -r "$PROJECT_DIR/requirements.txt" -q
pip install -e "$PROJECT_DIR" -q
echo "Dependencies installed."

# --- Cron: scrape every 6 hours + upload DB to GitHub Releases ---
CRON_LINE="0 */6 * * * export PATH=/opt/homebrew/bin:\$PATH && source \$HOME/.zshrc 2>/dev/null; cd $PROJECT_DIR && PYTHONPATH=$PROJECT_DIR $VENV/bin/python -m src.cli scrape >> $LOG 2>&1 && bash $PROJECT_DIR/scripts/upload_db.sh >> $LOG 2>&1"
# Remove old entry, add new one
( crontab -l 2>/dev/null | grep -v "olx-car-parser" || true ; echo "$CRON_LINE" ) | crontab -
echo "Cron job set: scrape every 6 hours + upload DB."

# --- GITHUB_TOKEN for DB upload ---
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo ""
  echo "WARNING: GITHUB_TOKEN is not set."
  echo "  To enable DB upload to GitHub Releases, add to ~/.zshrc or ~/.bashrc:"
  echo "  export GITHUB_TOKEN=\"ghp_your_token_here\""
  echo "  Then re-run this setup."
fi

# --- LaunchAgent for Streamlit dashboard ---
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST="$PLIST_DIR/com.olx-car-parser.dashboard.plist"
mkdir -p "$PLIST_DIR"

cat > "$PLIST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.olx-car-parser.dashboard</string>
    <key>ProgramArguments</key>
    <array>
        <string>${VENV}/bin/streamlit</string>
        <string>run</string>
        <string>src/dashboard/app.py</string>
        <string>--server.port</string>
        <string>8501</string>
        <string>--server.address</string>
        <string>0.0.0.0</string>
        <string>--server.headless</string>
        <string>true</string>
        <string>--browser.gatherUsageStats</string>
        <string>false</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${PROJECT_DIR}/data/dashboard.log</string>
    <key>StandardErrorPath</key>
    <string>${PROJECT_DIR}/data/dashboard.log</string>
</dict>
</plist>
EOF

# Reload launchd agent
launchctl bootout gui/$(id -u) "$PLIST" 2>/dev/null || true
launchctl bootstrap gui/$(id -u) "$PLIST"
echo "Dashboard service started on port 8501."

# --- Run initial scrape if DB is empty ---
if [ ! -f "$PROJECT_DIR/data/olx_cars.db" ]; then
    echo "Running initial scrape..."
    "$VENV/bin/python" -m src.cli scrape >> "$LOG" 2>&1 || true
fi

IP=$(ipconfig getifaddr en0 2>/dev/null || echo "this-server")
echo ""
echo "====================================="
echo " Setup complete!"
echo " LAN Dashboard:    http://${IP}:8501"
echo " Public Dashboard: https://share.streamlit.io (connect repo)"
echo " Scraping:         every 6h via cron"
echo " DB upload:        after each scrape (needs GITHUB_TOKEN)"
echo " Logs:             $LOG"
echo "====================================="
