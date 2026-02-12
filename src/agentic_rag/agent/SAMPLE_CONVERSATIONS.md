# Sample Conversations — Agentic RAG

These examples illustrate how the agent reasons through different query
types.  Each shows the **reasoning trace**, **tool calls**, and **cited
answer** the agent would produce.

---

## 1. Simple Factual Question

**User**: _What is Kubeflow?_

### Reasoning Trace

```
Step 1 [analyze_query]: Intent: conceptual. Key concepts: ['Kubeflow', 'ML platform'].
  → Action: Plan 1 retrieval step(s): ['vector_search']

Step 2 [execute_tools]: Executing 1 planned tool call(s).
  → Action: Run tools
  → Observation: vector_search('Kubeflow overview') → 5 docs

Step 3 [grade_results]: Verdict: sufficient. Missing: nothing.
  → Action: Proceed to synthesis

Step 4 [synthesize]: Generating cited answer from gathered evidence.
  → Observation: Answer length: 487 chars, 5 citation(s).
```

### Tool Calls

| # | Tool           | Query               | Docs |
|---|----------------|---------------------|------|
| 1 | vector_search  | Kubeflow overview   | 5    |

### Answer

> Kubeflow is an open-source machine learning platform designed to make
> deployments of ML workflows on Kubernetes simple, portable, and
> scalable [1]. It provides components for model training, serving, and
> pipeline orchestration [2]. Kubeflow Pipelines enable reproducible,
> end-to-end ML workflows [3].
>
> **Sources**
> [1] docs/kubeflow-intro.md §1
> [2] docs/architecture.md §3
> [3] docs/pipelines-overview.md §2

---

## 2. Specific Document Question (Multi-tool)

**User**: _What does the agentic_rag_overview.md file say about the retrieval strategy?_

### Reasoning Trace

```
Step 1 [analyze_query]: Intent: factual. Key concepts: ['agentic_rag_overview.md', 'retrieval strategy'].
  → Action: Plan 2 retrieval step(s): ['document_lookup', 'keyword_search']

Step 2 [execute_tools]: Executing 2 planned tool call(s).
  → Action: Run tools
  → Observation: document_lookup('agentic_rag_overview.md') → 8 docs; keyword_search('retrieval strategy') → 4 docs

Step 3 [grade_results]: Verdict: sufficient. Missing: nothing.
  → Action: Proceed to synthesis

Step 4 [synthesize]: Generating cited answer from gathered evidence.
  → Observation: Answer length: 623 chars, 6 citation(s).
```

### Tool Calls

| # | Tool             | Query / Source           | Docs |
|---|------------------|--------------------------|------|
| 1 | document_lookup  | agentic_rag_overview.md  | 8    |
| 2 | keyword_search   | retrieval strategy       | 4    |

### Answer

> The agentic RAG overview document describes a **two-phase retrieval
> strategy**: an initial semantic search followed by an optional
> re-ranking step [1]. The semantic search uses dense embeddings from
> `sentence-transformers/all-MiniLM-L6-v2` [2], and the re-ranking
> step applies a cross-encoder to improve precision [3]. The document
> also notes that metadata filters can restrict retrieval to specific
> data sources [4].
>
> **Sources**
> [1] agentic_rag_overview.md §2
> [2] agentic_rag_overview.md §3
> [3] agentic_rag_overview.md §5
> [4] agentic_rag_overview.md §6

---

## 3. Comparative Question (Retry Loop)

**User**: _How does the Kubeflow pipeline approach differ from a simple Python script for ML training?_

### Reasoning Trace

```
Step 1 [analyze_query]: Intent: comparative. Key concepts: ['Kubeflow pipelines', 'Python script', 'ML training'].
  → Action: Plan 2 retrieval step(s): ['vector_search', 'vector_search']

Step 2 [execute_tools]: Executing 2 planned tool call(s).
  → Action: Run tools
  → Observation: vector_search('Kubeflow pipeline ML training benefits') → 3 docs; vector_search('simple Python script ML training limitations') → 2 docs

Step 3 [grade_results]: Verdict: insufficient. Missing: Need concrete comparison points between pipeline orchestration and scripts.
  → Action: Refine and retry

Step 4 [execute_tools]: Executing 1 planned tool call(s).
  → Action: Run tools
  → Observation: vector_search('Kubeflow pipeline vs script reproducibility scalability') → 4 docs

Step 5 [grade_results]: Verdict: sufficient. Missing: nothing.
  → Action: Proceed to synthesis

Step 6 [synthesize]: Generating cited answer from gathered evidence.
  → Observation: Answer length: 812 chars, 7 citation(s).
```

### Tool Calls

| # | Tool           | Query                                                       | Docs |
|---|----------------|-------------------------------------------------------------|------|
| 1 | vector_search  | Kubeflow pipeline ML training benefits                      | 3    |
| 2 | vector_search  | simple Python script ML training limitations                | 2    |
| 3 | vector_search  | Kubeflow pipeline vs script reproducibility scalability     | 4    |

### Answer

> Kubeflow Pipelines and simple Python scripts differ across several
> dimensions:
>
> **Reproducibility** — Kubeflow Pipelines capture each step as a
> containerised component with pinned dependencies, ensuring
> reproducible runs [1][2]. A standalone script relies on the local
> environment, which can drift over time [3].
>
> **Scalability** — Pipelines execute steps in parallel on Kubernetes
> and can distribute training across multiple nodes [4]. Scripts run
> sequentially on a single machine unless manually parallelised.
>
> **Auditability** — Every pipeline run is versioned with artifacts,
> logs, and metrics stored centrally [5], whereas script outputs are
> typically scattered across local file systems [6].
>
> **Sources**
> [1] docs/pipelines-overview.md §2
> [2] docs/architecture.md §5
> [3] docs/best-practices.md §1
> [4] docs/pipelines-overview.md §4
> [5] docs/pipelines-overview.md §6
> [6] docs/best-practices.md §3

---

## 4. Out-of-scope Question (Web Search Stub)

**User**: _What is the latest LangGraph release version?_

### Reasoning Trace

```
Step 1 [analyze_query]: Intent: factual. Key concepts: ['LangGraph', 'latest release', 'version'].
  → Action: Plan 2 retrieval step(s): ['vector_search', 'web_search']

Step 2 [execute_tools]: Executing 2 planned tool call(s).
  → Action: Run tools
  → Observation: vector_search('LangGraph release version') → 1 docs; web_search('latest LangGraph release version') → 1 docs

Step 3 [grade_results]: Verdict: sufficient. Missing: nothing.
  → Action: Proceed to synthesis

Step 4 [synthesize]: Generating cited answer from gathered evidence.
  → Observation: Answer length: 295 chars, 2 citation(s).
```

### Answer

> The knowledge base mentions that the project depends on
> `langgraph>=0.1,<1` [1], but does not include the latest release
> number. A live web search was not available during this session [2].
> For the most current version, check the
> [LangGraph PyPI page](https://pypi.org/project/langgraph/).
>
> **Sources**
> [1] pyproject.toml §0
> [2] web_search_stub
