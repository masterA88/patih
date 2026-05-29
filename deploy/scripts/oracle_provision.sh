#!/bin/bash
# Oracle Cloud Always Free ARM A1 — initial provisioning script
# See build-spec Section 7.1 and 7.2
# Step 8 — run once on fresh Ubuntu 24.04 ARM VM

set -euo pipefail

echo "=== Oracle ARM A1 provisioning for chatbot-permensos ==="

# OS dependencies
sudo apt-get update && sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev build-essential \
    tesseract-ocr tesseract-ocr-ind libleptonica-dev poppler-utils \
    git curl jq docker.io docker-compose-v2 sqlite3 ufw

# Firewall
sudo ufw allow OpenSSH
sudo ufw allow from 172.20.0.0/16 to any port 8000  # Cloudflared subnet
sudo ufw --force enable

# Clone repo (replace with actual repo URL)
# git clone <repo-url> /opt/chatbot-permensos
# cd /opt/chatbot-permensos

echo "TODO: clone repo, install Python env, run quantize_e5.py, configure .env"
echo "See build-spec Section 7.2 for full runbook"
