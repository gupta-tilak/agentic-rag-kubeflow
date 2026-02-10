# 🧠 Agentic RAG on Kubeflow

> Production-grade **Retrieval-Augmented Generation** with autonomous agents,
> orchestrated on **Kubeflow Pipelines** and served via **KServe**.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![KFP](https://img.shields.io/badge/Kubeflow_Pipelines-v2-orange)](https://www.kubeflow.org/docs/components/pipelines/)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
  - [Running the Agent Locally](#running-the-agent-locally)
- [Kubeflow Pipeline](#kubeflow-pipeline)
- [Deployment](#deployment)
  - [Docker](#docker)
  - [KServe](#kserve)
- [Infrastructure](#infrastructure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements an **agentic RAG** system where a LangGraph-powered
autonomous agent decides _when_ and _how_ to retrieve external knowledge
before generating an answer. The entire workflow is productionised on
Kubeflow:

| Concern       | Technology             |
|---------------|------------------------|
| Agent logic   | LangGraph / LangChain  |
| Orchestration | Kubeflow Pipelines v2  |
| Serving       | KServe + FastAPI       |
| Vector store  | ChromaDB               |
| Embeddings    | sentence-transformers  |
| IaC (optional)| Terraform              |

---

## Architecture

```
┌────────────┐    ┌─────────────┐    ┌────────────┐
│  Documents │───▶│  Ingestion  │───▶│  ChromaDB  │
└────────────┘    │  (chunking  │    │  (vectors) │
                  │  + embed)   │    └─────┬──────┘
                  └─────────────┘          │
                                           ▼
┌────────────┐    ┌─────────────┐    ┌────────────┐
│   User     │───▶│   Agent     │───▶│ Retrieval  │
│   Query    │    │ (LangGraph) │◀───│            │
└────────────┘    └──────┬──────┘    └────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   KServe /  │
                  │   FastAPI   │
                  └─────────────┘
```

---

## Repository Structure

```
agentic-rag-kubeflow/
├── src/agentic_rag/            # Core Python library
│   ├── ingestion/              #   Document loading, chunking, embedding
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── retrieval/              #   Vector search & re-ranking
│   │   ├── retriever.py
│   │   └── reranker.py
│   ├── agent/                  #   LangGraph agent (infra-independent)
│   │   ├── state.py            #     Agent state definition
│   │   ├── nodes.py            #     Graph node functions
│   │   ├── graph.py            #     Graph wiring
│   │   ├── prompts.py          #     Prompt templates
│   │   ├── llm.py              #     LLM provider config
│   │   └── tools.py            #     Tools exposed to the agent
│   ├── serving/                #   HTTP & KServe runtime
│   │   ├── app.py              #     FastAPI application
│   │   └── kserve_runtime.py   #     KServe model class
│   └── config.py               #   Centralised settings (pydantic)
│
├── pipelines/                  # Kubeflow Pipelines (isolated)
│   ├── components/             #   KFP v2 component definitions
│   │   ├── ingest.py
│   │   └── evaluate.py
│   ├── rag_pipeline.py         #   Pipeline definition
│   └── compiled/               #   Compiled YAML artefacts
│
├── infra/                      # Infrastructure (optional)
│   ├── kserve/                 #   KServe manifests
│   │   ├── inference-service.yaml
│   │   └── secrets.yaml
│   ├── k8s/                    #   Plain Kubernetes manifests
│   │   ├── namespace.yaml
│   │   └── chroma.yaml
│   └── terraform/              #   Terraform IaC (optional)
│       ├── main.tf
│       └── terraform.tfvars.example
│
├── tests/                      # Test suite
│   ├── unit/
│   └── integration/
│
├── data/sample/                # Sample documents for quick-start
├── docs/                       # Project documentation
│
├── pyproject.toml              # PEP 621 project metadata
├── Makefile                    # Developer shortcuts
├── Dockerfile                  # Container image
├── .env.example                # Environment variable template
├── .pre-commit-config.yaml     # Code quality hooks
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker (for containerised deployment)
- A running ChromaDB instance (or `docker run -p 8000:8000 chromadb/chroma`)
- _(Optional)_ A Kubeflow cluster for pipeline execution
- _(Optional)_ Terraform ≥ 1.5 for infrastructure provisioning

### Local Development

```bash
# Clone the repository
git clone https://github.com/tilakgupta/agentic-rag-kubeflow.git
cd agentic-rag-kubeflow

# Set up environment
make install-dev

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Verify installation
make test
```

### Running the Agent Locally

```bash
# Start ChromaDB
docker run -d -p 8000:8000 chromadb/chroma

# Ingest sample documents
python -c "
from agentic_rag.ingestion.loader import load_directory
from agentic_rag.ingestion.chunker import chunk_documents
from agentic_rag.ingestion.embedder import embed_and_store

docs = load_directory('data/sample')
chunks = chunk_documents(docs)
embed_and_store(chunks)
print(f'Stored {len(chunks)} chunks')
"

# Launch the API server
uvicorn agentic_rag.serving.app:app --reload --port 8080

# Query the agent
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic RAG?"}'
```

---

## Kubeflow Pipeline

```bash
# Compile the pipeline to YAML
make compile-pipeline

# Upload to Kubeflow (requires KFP SDK + cluster access)
python -c "
import kfp
client = kfp.Client(host='http://localhost:8888')
client.upload_pipeline('pipelines/compiled/rag_pipeline.yaml', pipeline_name='agentic-rag')
"
```

---

## Deployment

### Docker

```bash
docker build -t agentic-rag-kubeflow:latest .
docker run -p 8080:8080 --env-file .env agentic-rag-kubeflow:latest
```

### KServe

```bash
# Create secrets
kubectl apply -f infra/kserve/secrets.yaml

# Deploy the InferenceService
kubectl apply -f infra/kserve/inference-service.yaml

# Deploy ChromaDB (if not already running)
kubectl apply -f infra/k8s/chroma.yaml
```

---

## Infrastructure

Infrastructure provisioning is **optional**. The project runs on any
Kubernetes cluster with Kubeflow and KServe installed.

For GKE provisioning via Terraform:

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars
terraform init && terraform apply
```

---

## Testing

```bash
make test          # Unit + integration tests
make lint          # Ruff linter
make typecheck     # Mypy type checking
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
