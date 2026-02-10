# Contributing to Agentic RAG on Kubeflow

Thank you for your interest in contributing! This document explains how to
get involved, what we expect from contributions, and how the review process
works.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Layout](#project-layout)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Review Process](#review-process)

---

## Code of Conduct

This project follows the
[Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
By participating you agree to abide by its terms.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/agentic-rag-kubeflow.git
   cd agentic-rag-kubeflow
   ```
3. **Create a branch** for your change:
   ```bash
   git checkout -b feat/my-new-feature
   ```

---

## Development Setup

```bash
# Create venv and install all dev dependencies
make install-dev

# Install pre-commit hooks (runs linter + formatter on every commit)
pre-commit install

# Copy environment variables
cp .env.example .env
```

---

## Project Layout

| Directory              | Purpose                                        |
|------------------------|------------------------------------------------|
| `src/agentic_rag/`    | Core library — agent, ingestion, retrieval, serving |
| `pipelines/`           | Kubeflow Pipeline components and definitions   |
| `infra/`               | Kubernetes / KServe / Terraform manifests       |
| `tests/`               | Unit and integration tests                     |
| `docs/`                | Documentation sources                          |

> **Key principle:** Agent logic (`src/agentic_rag/agent/`) must remain
> independent of infrastructure code. Never import from `pipelines/` or
> `infra/` inside the agent module.

---

## How to Contribute

### Reporting Bugs

Open a [GitHub Issue](https://github.com/tilakgupta/agentic-rag-kubeflow/issues/new)
with:

- A clear title and description.
- Steps to reproduce.
- Expected vs. actual behaviour.
- Python version and OS.

### Suggesting Features

Open an issue with the **`enhancement`** label. Describe the motivation,
proposed solution, and any alternatives considered.

### Submitting Pull Requests

1. Ensure your branch is up to date with `main`.
2. Write or update tests for your changes.
3. Run the full quality suite:
   ```bash
   make lint && make typecheck && make test
   ```
4. Push your branch and open a PR against `main`.
5. Fill in the PR template — link related issues.

---

## Coding Standards

- **Formatter / Linter:** [Ruff](https://docs.astral.sh/ruff/) (configured
  in `pyproject.toml`).
- **Type checking:** [Mypy](https://mypy-lang.org/) in strict mode.
- **Line length:** 100 characters.
- **Docstrings:** NumPy style.
- **Imports:** Sorted by `isort` via Ruff — first-party under `agentic_rag`.

All checks run automatically via pre-commit hooks. CI will reject PRs that
fail linting or type checks.

---

## Testing

```bash
make test            # Run full test suite with coverage
pytest tests/unit    # Run only unit tests
pytest tests/integration  # Run integration tests (needs ChromaDB)
```

- **Unit tests** should not require external services.
- **Integration tests** may connect to ChromaDB or an LLM API — mark them
  with `@pytest.mark.integration`.
- Aim for ≥ 80 % coverage on new code.

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

<optional body>
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`

**Examples:**

```
feat(agent): add web-search tool to LangGraph graph
fix(ingestion): handle empty PDF pages gracefully
docs(readme): update architecture diagram
test(retrieval): add unit tests for reranker
```

---

## Review Process

1. At least **one approving review** is required before merging.
2. CI must pass (lint + typecheck + tests).
3. Keep PRs focused — one logical change per PR.
4. Respond to review feedback promptly.

Thank you for helping make this project better! 🎉
