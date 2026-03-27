#!/usr/bin/env bash
# Upload olx_cars.db to GitHub Releases (tag: latest-data).
# Requires: GITHUB_TOKEN env var with repo scope.
# Usage: ./scripts/upload_db.sh
set -euo pipefail

REPO="nikit34/olx-car-parser"
TAG="latest-data"
DB_FILE="${1:-data/olx_cars.db}"
ASSET_NAME="olx_cars.db"
API="https://api.github.com"
UPLOAD="https://uploads.github.com"

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "Error: GITHUB_TOKEN is not set." >&2
  exit 1
fi

if [ ! -f "$DB_FILE" ]; then
  echo "Error: $DB_FILE not found." >&2
  exit 1
fi

# --- Flush WAL to main DB file so the upload contains all data ---
echo "Checkpointing WAL..."
sqlite3 "$DB_FILE" "PRAGMA wal_checkpoint(TRUNCATE);" 2>/dev/null || \
  python3 -c "import sqlite3; sqlite3.connect('$DB_FILE').execute('PRAGMA wal_checkpoint(TRUNCATE)')"

AUTH="Authorization: Bearer ${GITHUB_TOKEN}"
ACCEPT="Accept: application/vnd.github+json"

# --- Get or create the release ---
RELEASE_JSON=$(curl -sf -H "$AUTH" -H "$ACCEPT" \
  "$API/repos/$REPO/releases/tags/$TAG" 2>/dev/null || echo "")

if [ -z "$RELEASE_JSON" ]; then
  echo "Creating release '$TAG'..."
  RELEASE_JSON=$(curl -sf -X POST -H "$AUTH" -H "$ACCEPT" \
    -d "{\"tag_name\":\"$TAG\",\"name\":\"Latest Data\",\"body\":\"Auto-updated database\",\"draft\":false,\"prerelease\":false}" \
    "$API/repos/$REPO/releases")
fi

RELEASE_ID=$(echo "$RELEASE_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "Release ID: $RELEASE_ID"

# --- Delete existing asset if present ---
ASSETS=$(curl -sf -H "$AUTH" -H "$ACCEPT" \
  "$API/repos/$REPO/releases/$RELEASE_ID/assets")

OLD_ID=$(echo "$ASSETS" | python3 -c "
import sys, json
for a in json.load(sys.stdin):
    if a['name'] == '$ASSET_NAME':
        print(a['id']); break
" 2>/dev/null || echo "")

if [ -n "$OLD_ID" ]; then
  echo "Deleting old asset ($OLD_ID)..."
  curl -sf -X DELETE -H "$AUTH" -H "$ACCEPT" \
    "$API/repos/$REPO/releases/assets/$OLD_ID" >/dev/null
fi

# --- Upload new asset ---
SIZE=$(stat -f%z "$DB_FILE" 2>/dev/null || stat -c%s "$DB_FILE")
echo "Uploading $DB_FILE ($SIZE bytes)..."

curl -sf -X POST -H "$AUTH" -H "$ACCEPT" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@$DB_FILE" \
  "$UPLOAD/repos/$REPO/releases/$RELEASE_ID/assets?name=$ASSET_NAME" >/dev/null

echo "Done. DB published to https://github.com/$REPO/releases/tag/$TAG"
