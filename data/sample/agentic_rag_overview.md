# What is Agentic RAG?

Agentic RAG (Retrieval-Augmented Generation) is an advanced pattern where an
autonomous **agent** decides when and how to retrieve external knowledge before
generating an answer.

Unlike traditional RAG — where every query triggers a fixed retrieval step —
an agentic system can:

- Decide whether retrieval is even needed
- Reformulate queries for better recall
- Perform multiple retrieval rounds
- Combine retrieved evidence with reasoning

## Why Kubeflow?

Kubeflow provides production-grade orchestration for ML workflows:

- **Kubeflow Pipelines** for reproducible, versioned data and model pipelines
- **KServe** for scalable model serving with canary rollouts
- **Kubernetes-native** — runs on any conformant cluster

## Key Technologies

| Component         | Role                          |
|-------------------|-------------------------------|
| LangGraph         | Agent state machine           |
| LangChain         | LLM + retriever abstractions  |
| ChromaDB          | Vector storage                |
| sentence-transformers | Text embeddings            |
| Kubeflow Pipelines| Workflow orchestration        |
| KServe            | Model serving                 |
