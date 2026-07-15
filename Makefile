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

# MLX model repos + API/UI host:port — read from src.config (single source
# of truth, honors .env overrides). Empty if the venv isn't set up yet
# (fine: these targets all require `make setup` first anyway).
MLX_MODEL_REPO := $(shell $(VENV_BIN)/python -c "from src.config import settings; print(settings.mlx_model_repo)" 2>/dev/null)
MLX_EMBED_REPO := $(shell $(VENV_BIN)/python -c "from src.config import settings; print(settings.mlx_embed_repo)" 2>/dev/null)
API_HOST := $(shell $(VENV_BIN)/python -c "from src.config import settings; print(settings.api_host)" 2>/dev/null)
API_PORT := $(shell $(VENV_BIN)/python -c "from src.config import settings; print(settings.api_port)" 2>/dev/null)
UI_PORT := $(shell $(VENV_BIN)/python -c "from src.config import settings; print(settings.ui_port)" 2>/dev/null)

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
# NASA CMAPSS is now hosted in open access on data.nasa.gov (no PCoE account
# required as of 2026). The PCoE page still exists but its direct download
# was retired; data.nasa.gov is the canonical mirror.
CMAPSS_URL := https://data.nasa.gov/docs/legacy/CMAPSSData.zip
CMAPSS_ZIP := data/raw/cmapss/CMAPSSData.zip

.PHONY: data
data: ## Download NASA CMAPSS dataset (direct from data.nasa.gov, no account needed)
	@echo ">> Downloading NASA CMAPSS from data.nasa.gov..."
	@echo "   URL:  $(CMAPSS_URL)"
	@echo "   Size: ~12 MB compressed, ~45 MB uncompressed"
	@mkdir -p data/raw/cmapss
	@if [ ! -f $(CMAPSS_ZIP) ]; then \
	    curl -sL --fail -o $(CMAPSS_ZIP) $(CMAPSS_URL) || \
	        (echo "   ERROR: download failed. Check your network or grab the file manually:" \
	         "         $(CMAPSS_URL)"; exit 1); \
	else \
	    echo "   Already downloaded, skipping."; \
	fi
	@echo ">> Unzipping into data/raw/cmapss/..."
	@unzip -o -q $(CMAPSS_ZIP) -d data/raw/cmapss/
	@rm $(CMAPSS_ZIP)
	@echo ">> Files now in data/raw/cmapss/:"
	@ls data/raw/cmapss/ | sed 's/^/   - /'
	@echo ""
	@echo ">> Dataset ready. Next: 'make chroma-up && make ingest'."

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
	# `streamlit run` puts the script's own dir on sys.path, not the
	# project root, so `from src.config import settings` fails with
	# ModuleNotFoundError unless PYTHONPATH is set explicitly here.
	PYTHONPATH=. $(VENV_BIN)/streamlit run src/ui/streamlit_app.py --server.port $(UI_PORT) --server.address 0.0.0.0

# ----- Evaluation -----------------------------------------------------------
.PHONY: eval
eval: ## Run the RAGAS evaluation suite and snapshot the scores
	@echo ">> Running RAGAS evaluation..."
	$(VENV_BIN)/python -m src.eval.ragas_runner
	@echo ">> Reports in reports/"

.PHONY: eval-dataset
eval-dataset: ## Regenerate the evaluation dataset from CMAPSS
	$(VENV_BIN)/python -m src.eval.dataset

# ----- Collection management (manual — never auto-reset) --------------------
# See scripts/manage_collection.py. All destructive operations require
# explicit human confirmation. The ingest pipeline does NOT reset anything.
.PHONY: collection-list
collection-list: ## List all Chroma collections with dim and chunk count
	$(VENV_BIN)/python scripts/manage_collection.py list

.PHONY: collection-info
collection-info: ## Show details for one collection (COLLECTION=name)
	$(VENV_BIN)/python scripts/manage_collection.py info $(COLLECTION)

.PHONY: collection-new
collection-new: ## Create a new empty collection (NAME=name, never overwrites)
	$(VENV_BIN)/python scripts/manage_collection.py new $(NAME)

.PHONY: collection-drop
collection-drop: ## Drop a collection (IRREVERSIBLE — requires confirm via stdin)
	@echo "⚠️  About to drop $(NAME) (IRREVERSIBLE)."
	@echo "   Source data (CMAPSS + PDFs) is NOT affected — you can rebuild with 'make ingest'."
	@read -p "   Type the collection name to confirm: " confirm && \
	    [ "$$confirm" = "$(NAME)" ] && \
	    $(VENV_BIN)/python scripts/manage_collection.py drop $(NAME) --yes || \
	    (echo "   Aborted (input did not match)."; exit 1)

.PHONY: collection-use
collection-use: ## Set the active collection in .env (NAME=name)
	$(VENV_BIN)/python scripts/manage_collection.py use $(NAME)

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
