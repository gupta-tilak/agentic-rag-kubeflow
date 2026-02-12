# Agentic RAG — Architecture & Design

## Overview

This agent implements **true** agentic Retrieval-Augmented Generation: the
LLM reasons about _which tools to use_, queries _multiple data sources_,
evaluates the evidence, and synthesises a cited answer — all within a
LangGraph state machine that is fully testable without Kubeflow.

---

## Graph Topology

```
                  ┌──────────┐
                  │  START    │
                  └────┬─────┘
                       │
                       ▼
               ┌───────────────┐
               │ analyze_query  │  Interpret intent, extract key
               │                │  concepts, plan which tools to call
               └───────┬───────┘
                       │
                       ▼
               ┌───────────────┐
           ┌──►│ execute_tools  │  Call vector_search, keyword_search,
           │   │                │  document_lookup, web_search …
           │   └───────┬───────┘
           │           │
           │           ▼
           │   ┌───────────────┐
           │   │ grade_results  │  LLM judges whether evidence is
           │   │                │  sufficient or needs refinement
           │   └───────┬───────┘
           │           │
           │     ┌─────┴──────┐
           │     │            │
           │  insufficient  sufficient
           │     │            │
           │     ▼            ▼
           └─────┘    ┌───────────────┐
                      │  synthesize    │  Generate cited answer,
                      │                │  extract structured citations
                      └───────┬───────┘
                              │
                              ▼
                          ┌───────┐
                          │  END  │
                          └───────┘
```

## State Schema

| Field              | Type                      | Purpose                                      |
|--------------------|---------------------------|----------------------------------------------|
| `messages`         | `list[BaseMessage]`       | Conversation history (LangGraph reducer)      |
| `query`            | `str`                     | User's original question                     |
| `query_analysis`   | `dict`                    | Structured intent / concepts / tool plan      |
| `retrieval_plan`   | `list[dict]`              | Ordered tool calls to execute                 |
| `documents`        | `list[Document]`          | All retrieved context (deduplicated)          |
| `reasoning_trace`  | `list[ReasoningStep]`     | Visible multi-step reasoning log              |
| `tool_calls_made`  | `list[ToolCall]`          | Audit log of every tool invocation            |
| `citations`        | `list[SourceCitation]`    | Structured source citations                   |
| `answer`           | `str`                     | Final synthesised answer                      |
| `iteration`        | `int`                     | Current retrieval loop count                  |
| `max_iterations`   | `int`                     | Safety cap (default 3)                        |
| `needs_more_info`  | `bool`                    | Flag: loop back for more retrieval?           |

## Node Descriptions

### 1. `analyze_query`
- **Input**: `query`
- **Output**: `query_analysis`, `retrieval_plan`, `reasoning_trace`
- Calls the LLM with a JSON-output prompt to classify intent, extract
  key concepts, and choose which retrieval tools to invoke.

### 2. `execute_tools`
- **Input**: `retrieval_plan`, `documents`
- **Output**: `documents`, `tool_calls_made`, `reasoning_trace`, `iteration`
- Iterates over the retrieval plan, calls each tool, deduplicates results,
  and logs every invocation in `tool_calls_made`.

### 3. `grade_results`
- **Input**: `query`, `documents`, `iteration`, `max_iterations`
- **Output**: `needs_more_info`, `retrieval_plan`, `reasoning_trace`
- Uses the LLM as a relevance judge. If evidence is insufficient and the
  iteration cap hasn't been reached, sets `needs_more_info = True` and
  populates a refined `retrieval_plan`.

### 4. `synthesize`
- **Input**: `query`, `documents`, `reasoning_trace`
- **Output**: `answer`, `citations`, `messages`, `reasoning_trace`
- Generates a final answer with inline `[1]`, `[2]` citations and a
  **Sources** section. Extracts structured `SourceCitation` objects.

## Available Tools

| Tool              | When to use                                        |
|-------------------|----------------------------------------------------|
| `vector_search`   | Open-ended semantic queries                        |
| `keyword_search`  | Queries mentioning specific sources or topics      |
| `document_lookup` | Retrieve all chunks from a specific document       |
| `web_search`      | External/recent information (stub — wire in prod)  |

## Testability

Every node is a pure function of `AgentState` + LLM calls. Tests inject
mock LLMs and fake tool registries — **no Kubeflow, no Chroma, no OpenAI
key needed**.

```python
from agentic_rag.agent.graph import build_graph, create_initial_state

graph = build_graph()
state = create_initial_state("How does KServe autoscaling work?")
result = graph.invoke(state)

print(result["answer"])
print(result["citations"])
print(result["reasoning_trace"])
```
