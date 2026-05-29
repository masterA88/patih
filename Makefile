# Makefile — chatbot-permensos task entry points
# See build-spec Section 1 (line 41) and Section 13 for step context
# Requires: Poetry installed and `poetry install` already run

.PHONY: install lint test clean ingest index eval serve

# ─── Setup ───────────────────────────────────────────────────────────────────

install:
	poetry install

# ─── Code quality ────────────────────────────────────────────────────────────

lint:
	poetry run ruff check app/ tests/
	poetry run black --check app/ tests/

format:
	poetry run black app/ tests/
	poetry run ruff check --fix app/ tests/

# ─── Tests ───────────────────────────────────────────────────────────────────

test:
	poetry run pytest tests/unit/ -v

test-integration:
	poetry run pytest tests/integration/ -v

test-all:
	poetry run pytest tests/ -v --tb=short

# ─── Pipeline entry points ───────────────────────────────────────────────────

ingest:
	@echo "Step 2 not yet implemented — run after build-spec Step 2 complete"
	@echo "Target command: poetry run python -m app.ingest.cli ingest --pdf data/raw/Permensos_Nomor_8_Tahun_2023.pdf --doc-id permensos-8-2023"

index:
	@echo "Step 3 not yet implemented — run after build-spec Step 3 complete"
	@echo "Target command: poetry run python -m app.retrieval.pipeline index"

eval:
	@echo "Step 7 not yet implemented — run after build-spec Step 7 complete"
	@echo "Target command: poetry run python -m app.eval.cli run --test-set data/test/permensos8_50q.jsonl"

serve:
	@echo "Step 6 not yet implemented — run after build-spec Step 6 complete"
	@echo "Target command: poetry run chainlit run app/ui/chainlit_app.py -w"

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
