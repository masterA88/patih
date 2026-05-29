# ARM-compatible Dockerfile for chatbot-permensos
# Target: Oracle Cloud Always Free ARM Ampere A1 (linux/arm64)
# Base: Ubuntu 24.04 with Python 3.11
# Step 8 — finalize before Oracle deploy

FROM ubuntu:24.04

# System deps — see build-spec Section 2.2
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    build-essential \
    tesseract-ocr tesseract-ocr-ind \
    libleptonica-dev \
    poppler-utils \
    git curl jq \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY pyproject.toml poetry.lock* ./
RUN python3.11 -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install poetry==1.8.3 && \
    .venv/bin/poetry install --no-root --no-dev

# Copy app code
COPY app/ ./app/
COPY configs/ ./configs/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8000

CMD ["chainlit", "run", "app/ui/chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
