#!/bin/bash
# Daily backup: Chroma + registry/parsed + Langfuse Postgres → OCI Object Storage
# See build-spec Section 7.4
# Cron: 0 2 * * * /opt/chatbot-permensos/deploy/scripts/backup_chroma.sh

set -euo pipefail

DATE=$(date +%Y%m%d)
BACKUP_DIR=/tmp/chatbot_backup
APP_DIR=/opt/chatbot-permensos

mkdir -p "$BACKUP_DIR"

tar czf "$BACKUP_DIR/chroma_${DATE}.tar.gz" -C "$APP_DIR" data/chroma
tar czf "$BACKUP_DIR/registry_${DATE}.tar.gz" -C "$APP_DIR" data/registry data/parsed

docker compose -f "$APP_DIR/deploy/docker-compose.yml" exec -T langfuse-db \
    pg_dump -U langfuse langfuse | gzip > "$BACKUP_DIR/langfuse_${DATE}.sql.gz"

# Upload to OCI Object Storage (requires oci CLI configured)
oci os object bulk-upload \
    --bucket-name chatbot-backup \
    --src-dir "$BACKUP_DIR" \
    --overwrite

rm -rf "$BACKUP_DIR"
# Retention handled via OCI lifecycle rule (7-day retention configured in OCI console)

echo "Backup complete: $DATE"
