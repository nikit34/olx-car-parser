#!/bin/bash
set -euo pipefail

BACKUP_HOST="${BACKUP_HOST:-permi@192.168.1.69}"
BACKUP_DIR="${BACKUP_DIR:-C:/olx-backup}"
BACKUP_NAME="${BACKUP_NAME:-olx_cars.dump}"
MIN_BYTES="${MIN_BYTES:-20000000}"
PGDUMP="${PGDUMP:-/opt/homebrew/opt/postgresql@17/bin/pg_dump}"

: "${OLX_DB_URL:?OLX_DB_URL is not set}"

log() { printf '%s %s\n' "$(date -u '+%F %T')" "$*"; }

local_dump="$(mktemp -t olx_cars).dump"
trap 'rm -f "$local_dump"' EXIT

log "dumping"
"$PGDUMP" -Fc --no-owner --no-privileges -f "$local_dump" \
    "$(printf '%s' "$OLX_DB_URL" | sed 's|^postgresql+psycopg://|postgresql://|')"

size=$(stat -f %z "$local_dump")
if [ "$size" -lt "$MIN_BYTES" ]; then
    log "ABORT: dump is $size bytes, below the $MIN_BYTES floor"
    exit 1
fi
log "dump ok: $size bytes"

ssh -o BatchMode=yes -o ConnectTimeout=10 "$BACKUP_HOST" \
    "if not exist \"${BACKUP_DIR//\//\\}\" mkdir \"${BACKUP_DIR//\//\\}\""

log "shipping to $BACKUP_HOST:$BACKUP_DIR"
scp -q -o BatchMode=yes -o ConnectTimeout=10 \
    "$local_dump" "$BACKUP_HOST:$BACKUP_DIR/$BACKUP_NAME.incoming"

remote_size=$(ssh -o BatchMode=yes "$BACKUP_HOST" \
    "powershell -NoProfile -Command \"(Get-Item '$BACKUP_DIR/$BACKUP_NAME.incoming').Length\"" \
    | tr -d '\r')
if [ "$remote_size" != "$size" ]; then
    log "ABORT: shipped $remote_size bytes, expected $size"
    ssh -o BatchMode=yes "$BACKUP_HOST" \
        "powershell -NoProfile -Command \"Remove-Item -Force '$BACKUP_DIR/$BACKUP_NAME.incoming'\"" || true
    exit 1
fi

ssh -o BatchMode=yes "$BACKUP_HOST" \
    "powershell -NoProfile -Command \"Move-Item -Force '$BACKUP_DIR/$BACKUP_NAME.incoming' '$BACKUP_DIR/$BACKUP_NAME'\""

marker="$(cd "$(dirname "$0")/.." && pwd)/data/.last_backup"
date -u '+%F %T' > "$marker"

log "replaced $BACKUP_DIR/$BACKUP_NAME ($size bytes)"
