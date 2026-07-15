# =============================================================================
# Industrial Knowledge Copilot — Makefile
# =============================================================================
# This is the entry point for running the project locally.
#
# Architecture reminder:
#   - ChromaDB runs in Docker (managed by docker compose)
#   - MLX LLM + embeddings run NATIVELY on the macOS host (Metal/MPS)
#   - FastAPI + Streamlit run on the host (so they can talk to MLX)
#
# Typical first run:
#   make setup            # create venv, install deps
#   make pull-models      # download Mistral 7B + bge-small (one-time, ~5 GB)
#   make data             # download NASA CMAPSS (requires NASA PCoE account)
#   make chroma-up        # start ChromaDB
#   make ingest           # build the vector index
#   make api              # start the API on :8000
#   make ui               # start the Streamlit UI on :8501
#   make eval             # run RAGAS evaluation, snapshot scores
# =============================================================================

# Detect platform
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
else
    PLATFORM := linux
endif

# Detect Apple Silicon
ifeq ($(PLATFORM),macos)
    ARCH := $(shell uname -m)
    ifeq ($(ARCH),arm64)
        IS_APPLE_SILICON := 1
    else
        IS_APPLE_SILICON := 0
    endif
else
    IS_APPLE_SILICON := 0
endif

# Python (use python3 explicitly to avoid pyenv shadowing)
PYTHON ?= python3
VENV := .venv
VENV_BIN := $(VENV)/bin

# Load .env if present (export all vars)
ifneq (,$(wildcard .env))
    include .env
    export
endif

.DEFAULT_GOAL := help

# ----- Help -----------------------------------------------------------------
.PHONY: help
help: ## Show this help message
	@echo "Industrial Knowledge Copilot — Makefile"
	@echo ""
	@echo "Platform: $(PLATFORM) | Apple Silicon: $(IS_APPLE_SILICON)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ----- Setup ----------------------------------------------------------------
.PHONY: setup
setup: ## Create venv, install runtime + dev dependencies
	@echo ">> Creating venv at $(VENV)..."
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@echo ">> Upgrading pip..."
	$(VENV_BIN)/pip install --upgrade pip wheel setuptools
	@echo ">> Installing dependencies (this may take a few minutes)..."
	$(VENV_BIN)/pip install -r requirements-dev.txt
	@echo ">> Done. Activate with: source $(VENV_BIN)/activate"

.PHONY: preflight
preflight: ## Verify the host is Apple Silicon (MLX will not work otherwise)
	@echo ">> Pre-flight checks"
	@echo "   Platform: $(PLATFORM)"
	@echo "   Arch:     $(ARCH)"
	@echo "   Apple Silicon: $(IS_APPLE_SILICON)"
	@if [ "$(IS_APPLE_SILICON)" != "1" ]; then \
	    echo ""; \
	    echo "   WARNING: MLX requires Apple Silicon (M1/M2/M3/M4/M5)."; \
	    echo "   This host is $(PLATFORM)/$(ARCH)."; \
	    echo "   The LLM/embedding modules will refuse to load (intentional)."; \
	    echo "   Use a different backend or run on Apple Silicon hardware."; \
	    echo ""; \
	fi
	@$(PYTHON) -c "import sys; assert sys.version_info >= (3, 12), 'Python 3.12+ required'" \
	    && echo "   Python:    $(shell $(PYTHON) --version) OK" \
	    || (echo "   Python 3.12+ required"; exit 1)
	@command -v docker >/dev/null && echo "   Docker:    OK" \
	    || (echo "   Docker not found (needed for ChromaDB)"; exit 1)

# ----- Models ---------------------------------------------------------------
.PHONY: pull-models
pull-models: ## Download Mistral 7B (MLX) + bge-small embeddings into HF cache
	@echo ">> Downloading MLX models into HuggingFace cache..."
	@echo "   Target LLM:    $(MLX_MODEL_REPO)"
	@echo "   Target embed:  $(MLX_EMBED_REPO)"
	@echo "   This is a one-time download of ~5 GB. Be patient."
	@if [ "$(IS_APPLE_SILICON)" != "1" ]; then \
	    echo "   ERROR: MLX is Apple-Silicon only. Aborting."; exit 1; \
	fi
	$(VENV_BIN)/python -c "from huggingface_hub import snapshot_download; snapshot_download('$(MLX_MODEL_REPO)'); snapshot_download('$(MLX_EMBED_REPO)')"
	@echo ">> Models downloaded."

# ----- Data -----------------------------------------------------------------
.PHONY: data
data: ## Download NASA CMAPSS dataset (requires free NASA PCoE account)
	@echo ">> NASA CMAPSS download instructions"
	@echo ""
	@echo "   1. Register a free account at: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/"
	@echo "   2. Download the 'Turbofan Engine Degradation Simulation' dataset (CMAPSS)"
	@echo "   3. Unzip into: data/raw/cmapss/"
	@echo "   4. Expected files: train_FD001.txt, test_FD001.txt, RUL_FD001.txt (× FD002, FD003, FD004)"
	@echo "   5. Also drop the readme.txt (technical documentation for RAG ingestion)"
	@echo ""
	@echo "   If you don't have access yet, request it now — NASA approval is usually < 24h."
	@test -d data/raw/cmapss && ls data/raw/cmapss/ | head -5
	@echo ""
	@echo ">> After downloading, run: make ingest"

# ----- ChromaDB (Docker) ----------------------------------------------------
.PHONY: chroma-up
chroma-up: ## Start ChromaDB in Docker (detached)
	docker compose up -d chroma
	@echo ">> Waiting for ChromaDB to be ready..."
	@for i in $$(seq 1 30); do \
	    if curl -fsS http://localhost:8001/api/v1/heartbeat >/dev/null 2>&1; then \
	        echo "   ChromaDB ready on :8001"; break; \
	    fi; \
	    sleep 1; \
	done

.PHONY: chroma-down
chroma-down: ## Stop ChromaDB
	docker compose down

.PHONY: chroma-logs
chroma-logs: ## Tail ChromaDB logs
	docker compose logs -f chroma

.PHONY: chroma-reset
chroma-reset: ## Delete persisted ChromaDB index (irreversible)
	@echo ">> This will DELETE the persisted vector index."
	@read -p "   Are you sure? [y/N] " r && [ "$$r" = "y" ] || (echo "   Aborted."; exit 1)
	docker compose down
	docker volume rm ikc-chroma-data
	@echo ">> ChromaDB volume removed."

# ----- Pipeline -------------------------------------------------------------
.PHONY: ingest
ingest: ## Run the full ingestion pipeline (loaders -> chunker -> embeddings -> Chroma)
	@echo ">> Running ingestion pipeline..."
	@test -d data/raw/cmapss || (echo "   ERROR: data/raw/cmapss/ missing. Run 'make data' first."; exit 1)
	$(VENV_BIN)/python -m src.ingestion.pipeline

.PHONY: api
api: ## Start the FastAPI server on :8000
	$(VENV_BIN)/uvicorn src.api.main:app --host $(API_HOST) --port $(API_PORT) --reload

.PHONY: ui
ui: ## Start the Streamlit UI on :8501
	$(VENV_BIN)/streamlit run src/ui/streamlit_app.py --server.port $(UI_PORT) --server.address 0.0.0.0

# ----- Evaluation -----------------------------------------------------------
.PHONY: eval
eval: ## Run the RAGAS evaluation suite and snapshot the scores
	@echo ">> Running RAGAS evaluation..."
	$(VENV_BIN)/python -m src.eval.ragas_runner
	@echo ">> Reports in reports/"

.PHONY: eval-dataset
eval-dataset: ## Regenerate the evaluation dataset from CMAPSS
	$(VENV_BIN)/python -m src.eval.dataset

# ----- Quality --------------------------------------------------------------
.PHONY: lint
lint: ## Run ruff linter
	$(VENV_BIN)/ruff check src tests

.PHONY: format
format: ## Auto-format with ruff
	$(VENV_BIN)/ruff check --fix src tests
	$(VENV_BIN)/ruff format src tests

.PHONY: test
test: ## Run unit tests (skips integration tests)
	$(VENV_BIN)/pytest

.PHONY: test-integration
test-integration: ## Run integration tests (requires live services)
	$(VENV_BIN)/pytest -m integration

.PHONY: test-cov
test-cov: ## Run unit tests with coverage report
	$(VENV_BIN)/pytest --cov=src --cov-report=term-missing

# ----- Cleanup --------------------------------------------------------------
.PHONY: clean
clean: ## Remove caches and build artifacts (keeps data and models)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist *.egg-info .coverage htmlcov

.PHONY: nuke
nuke: clean chroma-reset ## Full reset (cache + chroma data + models cache)
	@echo ">> This will DELETE the ChromaDB volume and the HuggingFace model cache."
	@read -p "   Are you sure? [y/N] " r && [ "$$r" = "y" ] || (echo "   Aborted."; exit 1)
	rm -rf $(HOME)/.cache/huggingface/hub/models--mlx-community--*
	@echo ">> Full reset done."
