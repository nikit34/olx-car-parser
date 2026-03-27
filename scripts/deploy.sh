#!/usr/bin/env bash
# Deploy olx-car-parser to remote server and run setup.
# Usage: ./scripts/deploy.sh
set -euo pipefail

HOST="192.168.1.77"
USER="anastasia"
REMOTE_DIR="olx-car-parser"
SSH_CMD="ssh ${USER}@${HOST}"

echo "==> Syncing project to ${USER}@${HOST}:~/${REMOTE_DIR} ..."

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
    "$(cd "$(dirname "$0")/.." && pwd)/" \
    "${USER}@${HOST}:~/${REMOTE_DIR}/"

echo "==> Running remote setup..."
$SSH_CMD "bash ~/${REMOTE_DIR}/scripts/remote_setup.sh"

echo "==> Done! Dashboard: http://${HOST}:8501"
