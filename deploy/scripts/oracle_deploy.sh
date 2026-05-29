#!/usr/bin/env bash
# Oracle VM end-to-end deploy script (idempotent, runs from any pwd inside repo).
# Builds docker images, brings up stack, runs smoke test.
#
# Pre-req on VM:
#   - oracle_provision.sh sudah dijalankan (docker + tesseract + ufw)
#   - repo sudah di-clone ke /opt/patih (atau pwd)
#   - .env sudah di-fill (API keys + LANGFUSE_*)
#   - data_indexes.tar.gz sudah di-SCP ke project root, ATAU
#     PDF Permensos_Nomor_8_Tahun_2023.pdf sudah di data/raw/
#
# Usage:
#   cd /opt/patih
#   bash deploy/scripts/oracle_deploy.sh [--rebuild-index]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

REBUILD_INDEX=0
for arg in "$@"; do
  case "$arg" in
    --rebuild-index) REBUILD_INDEX=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

echo "=== Patih deploy starting at $(date -u +%FT%TZ) ==="
echo "REPO_ROOT: $REPO_ROOT"

# --- 1. Sanity: .env present ---
if [ ! -f .env ]; then
  echo "FATAL: .env not found. Copy from .env.example and fill in keys." >&2
  exit 1
fi
# Require at minimum GEMINI_API_KEY
if ! grep -q '^GEMINI_API_KEY=.\+' .env; then
  echo "FATAL: GEMINI_API_KEY empty in .env" >&2
  exit 1
fi

# --- 2. Indexes: extract or rebuild ---
if [ "$REBUILD_INDEX" = "1" ] || [ ! -d data/chroma ] || [ -z "$(ls -A data/chroma 2>/dev/null)" ]; then
  if [ -f deploy/artifacts/data_indexes.tar.gz ] && [ "$REBUILD_INDEX" = "0" ]; then
    echo "--- Extracting pre-built indexes from deploy/artifacts/data_indexes.tar.gz ---"
    tar xzf deploy/artifacts/data_indexes.tar.gz
  elif [ -f data_indexes.tar.gz ] && [ "$REBUILD_INDEX" = "0" ]; then
    echo "--- Extracting pre-built indexes from data_indexes.tar.gz ---"
    tar xzf data_indexes.tar.gz
  else
    echo "--- Rebuilding indexes from PDF (slower path) ---"
    if [ ! -f data/raw/Permensos_Nomor_8_Tahun_2023.pdf ]; then
      echo "FATAL: no indexes tarball and no PDF in data/raw/" >&2
      exit 1
    fi
    # Run via a one-shot container so we use the same image envs.
    docker compose -f deploy/docker-compose.yml run --rm chatbot \
      python -m app.ingest.cli ingest \
        --pdf data/raw/Permensos_Nomor_8_Tahun_2023.pdf \
        --doc-id permensos-8-2023
  fi
else
  echo "--- data/chroma exists, skipping ingest (use --rebuild-index to force) ---"
fi

# --- 3. Build + start stack ---
echo "--- docker compose build + up -d ---"
docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml up -d

# --- 4. Wait for chatbot /health ---
echo "--- Waiting for chatbot /health (max 90s) ---"
deadline=$(( $(date +%s) + 90 ))
until curl -fsSL http://localhost:8000/health >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "FATAL: chatbot did not come up within 90s. Last 50 log lines:" >&2
    docker compose -f deploy/docker-compose.yml logs --tail=50 chatbot >&2
    exit 1
  fi
  sleep 3
done
echo "chatbot /health OK"

# --- 5. Smoke test ---
echo "--- Running smoke test ---"
bash deploy/scripts/smoke_test.sh

echo "=== Deploy complete at $(date -u +%FT%TZ) ==="
echo
echo "Services:"
docker compose -f deploy/docker-compose.yml ps
echo
echo "Next: install cloudflared tunnel — see HANDOFF_STEP8.md section 4"
