.DEFAULT_GOAL := help
SHELL := /bin/bash

# ── Variables ─────────────────────────────────────────────────────────
PYTHON   ?= python3
VENV_DIR ?= .venv

# ── Environment ───────────────────────────────────────────────────────
.PHONY: venv install install-dev

venv:                          ## Create virtual environment
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

install: venv                  ## Install runtime dependencies
	$(VENV_DIR)/bin/pip install -e .

install-dev: venv              ## Install dev + all optional deps
	$(VENV_DIR)/bin/pip install -e ".[dev,kfp,serving,docs]"
	$(VENV_DIR)/bin/pre-commit install

# ── Quality ───────────────────────────────────────────────────────────
.PHONY: lint format typecheck test

lint:                          ## Run linter (ruff)
	ruff check src/ tests/

format:                        ## Auto-format code
	ruff format src/ tests/

typecheck:                     ## Run mypy
	mypy src/

test:                          ## Run test suite
	pytest

# ── Pipeline ──────────────────────────────────────────────────────────
.PHONY: compile-pipeline

compile-pipeline:              ## Compile KFP pipeline to YAML
	$(PYTHON) -m pipelines.rag_pipeline --compile

# ── Docs ──────────────────────────────────────────────────────────────
.PHONY: docs docs-serve

docs:                          ## Build documentation
	mkdocs build

docs-serve:                    ## Serve documentation locally
	mkdocs serve

# ── Clean ─────────────────────────────────────────────────────────────
.PHONY: clean

clean:                         ## Remove build artefacts
	rm -rf dist/ build/ *.egg-info .mypy_cache .pytest_cache .ruff_cache

# ── Help ──────────────────────────────────────────────────────────────
.PHONY: help

help:                          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
