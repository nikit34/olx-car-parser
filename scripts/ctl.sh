#!/usr/bin/env bash
# OLX Car Parser — Remote Control Panel
# Usage: ./scripts/ctl.sh <command>
set -euo pipefail

HOST="192.168.1.77"
USER="anastasia"
REMOTE="olx-car-parser"
SSH="ssh ${USER}@${HOST}"
VENV="~/${REMOTE}/.venv"
LOG="~/${REMOTE}/data/scrape.log"
PLIST_LABEL="com.olx-car-parser.dashboard"
PLIST_PATH="\$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

usage() {
    echo -e "${BOLD}OLX Car Parser — Remote Control${NC}"
    echo ""
    echo -e "Usage: ${CYAN}$0 <command>${NC}"
    echo ""
    echo "  status      Cron, dashboard, last scrape results"
    echo "  logs        Tail scraper log (live)"
    echo "  log         Show last 50 lines of log"
    echo "  scrape      Trigger scrape now"
    echo "  dashboard   Open dashboard in browser"
    echo "  deploy      Sync code & restart services"
    echo "  start       Start dashboard service"
    echo "  stop        Stop dashboard service"
    echo "  restart     Restart dashboard service"
    echo "  ssh         Open SSH session"
    echo "  db-pull     Download database to local data/"
    echo "  db-push     Upload local database to server"
    echo ""
}

case "${1:-help}" in

  status)
    echo -e "${BOLD}=== Server Status ===${NC}"
    $SSH bash -s << 'REMOTE_SCRIPT'
        PLIST_LABEL="com.olx-car-parser.dashboard"

        echo ""
        echo "--- Cron Schedule ---"
        crontab -l 2>/dev/null | grep olx || echo "(no cron jobs)"

        echo ""
        echo "--- Dashboard Service ---"
        if launchctl list "$PLIST_LABEL" &>/dev/null; then
            PID=$(launchctl list "$PLIST_LABEL" 2>/dev/null | grep PID | awk '{print $NF}')
            if [ -n "$PID" ] && [ "$PID" != "-" ]; then
                echo "Status: RUNNING (PID $PID)"
            else
                echo "Status: LOADED (not running)"
            fi
        else
            echo "Status: NOT LOADED"
        fi

        echo ""
        echo "--- Disk ---"
        DB="$HOME/olx-car-parser/data/olx_cars.db"
        if [ -f "$DB" ]; then
            echo "Database: $(du -h "$DB" | cut -f1)"
        else
            echo "Database: not found"
        fi
        df -h "$HOME" 2>/dev/null | tail -1 | awk '{print "Free space: " $4}'

        echo ""
        echo "--- Last Scrape ---"
        LOG="$HOME/olx-car-parser/data/scrape.log"
        if [ -f "$LOG" ]; then
            tail -8 "$LOG"
        else
            echo "(no logs yet)"
        fi
REMOTE_SCRIPT
    ;;

  logs)
    echo -e "${CYAN}Tailing scraper log (Ctrl+C to stop)...${NC}"
    $SSH "tail -f ${LOG}"
    ;;

  log)
    $SSH "tail -50 ${LOG} 2>/dev/null || echo '(no logs yet)'"
    ;;

  scrape)
    echo -e "${CYAN}Triggering scrape...${NC}"
    $SSH "cd ~/${REMOTE} && PYTHONPATH=~/${REMOTE} ${VENV}/bin/python -m src.cli scrape 2>&1 | tee -a ${LOG}"
    ;;

  alerts)
    echo -e "${CYAN}Sending deal alerts...${NC}"
    $SSH "cd ~/${REMOTE} && PYTHONPATH=~/${REMOTE} ${VENV}/bin/python -m src.cli alerts 2>&1 | tee -a ${LOG}"
    ;;

  dashboard)
    URL="http://${HOST}:8501"
    echo -e "${GREEN}Opening ${URL}${NC}"
    open "$URL" 2>/dev/null || xdg-open "$URL" 2>/dev/null || echo "$URL"
    ;;

  deploy)
    echo -e "${CYAN}Deploying...${NC}"
    PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    rsync -avz --delete \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.venv' \
        --exclude 'data/olx_cars.db' \
        --exclude 'data/*.log' \
        --exclude '.env' \
        --exclude '.DS_Store' \
        --exclude '.claude' \
        --exclude '.cursor' \
        -e "ssh" \
        "${PROJ_ROOT}/" \
        "${USER}@${HOST}:~/${REMOTE}/"
    $SSH "cd ~/${REMOTE} && ${VENV}/bin/pip install -r requirements.txt -q && ${VENV}/bin/pip install -e . -q && launchctl bootout gui/\$(id -u) ~/Library/LaunchAgents/${PLIST_LABEL}.plist 2>/dev/null; launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/${PLIST_LABEL}.plist"
    echo -e "${GREEN}Deployed & dashboard restarted.${NC}"
    ;;

  start)
    $SSH "launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/${PLIST_LABEL}.plist 2>/dev/null || launchctl kickstart gui/\$(id -u)/${PLIST_LABEL}"
    echo -e "${GREEN}Dashboard started.${NC}"
    ;;

  stop)
    $SSH "launchctl bootout gui/\$(id -u)/${PLIST_LABEL} 2>/dev/null || true"
    echo -e "${RED}Dashboard stopped.${NC}"
    ;;

  restart)
    $SSH "launchctl bootout gui/\$(id -u) ~/Library/LaunchAgents/${PLIST_LABEL}.plist 2>/dev/null; launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/${PLIST_LABEL}.plist"
    echo -e "${GREEN}Dashboard restarted.${NC}"
    ;;

  ssh)
    $SSH
    ;;

  db-pull)
    echo -e "${CYAN}Downloading database...${NC}"
    PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    scp "${USER}@${HOST}:~/${REMOTE}/data/olx_cars.db" "${PROJ_ROOT}/data/olx_cars.db"
    echo -e "${GREEN}Database saved to data/olx_cars.db${NC}"
    ;;

  db-push)
    echo -e "${CYAN}Uploading database...${NC}"
    PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    scp "${PROJ_ROOT}/data/olx_cars.db" "${USER}@${HOST}:~/${REMOTE}/data/olx_cars.db"
    echo -e "${GREEN}Database uploaded.${NC}"
    ;;

  help|*)
    usage
    ;;
esac
