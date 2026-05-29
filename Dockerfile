# HuggingFace Spaces — Docker SDK entrypoint
# Multi-arch (amd64 + arm64) — HF free CPU is amd64.
#
# Differences vs deploy/Dockerfile.app (Oracle ARM):
#   - No langfuse stack (single-container HF, langfuse → cloud free tier OR skip)
#   - Run as non-root `user` per HF Spaces convention
#   - Writes go to /data (HF persistent if enabled) or /tmp
#   - Port 8000 (set via `app_port: 8000` in README frontmatter)
#   - Indexes baked into image (data/ is COPYed; small ~365KB)

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    build-essential \
    tesseract-ocr tesseract-ocr-ind \
    libleptonica-dev \
    poppler-utils \
    git curl jq \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: non-root user with uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Install Python deps as the user (poetry writes to ~/.local).
COPY --chown=user pyproject.toml poetry.lock* ./
RUN python3.11 -m venv .venv \
 && .venv/bin/pip install --upgrade pip \
 && .venv/bin/pip install poetry==1.8.3 \
 && .venv/bin/poetry config virtualenvs.in-project true \
 && .venv/bin/poetry install --no-root --no-dev

# Copy app code + configs + pre-built indexes.
COPY --chown=user app/ ./app/
COPY --chown=user configs/ ./configs/
COPY --chown=user data/ ./data/
COPY --chown=user chainlit.md ./
COPY --chown=user .chainlit/ ./.chainlit/

ENV PATH=/home/user/app/.venv/bin:$PATH
ENV PYTHONPATH=/home/user/app
# Direct SQLite writes (conversations.db) to /tmp on free tier.
# Override via Space Secret CHATBOT_DATA_DIR=/data when persistent storage enabled.
ENV CHATBOT_CONVERSATIONS_DB=/tmp/conversations.db

EXPOSE 8000

CMD ["chainlit", "run", "app/ui/chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
