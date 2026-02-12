"""Prompt templates for the agentic RAG workflow.

Every node that calls the LLM uses a dedicated prompt from this module.
Keeping prompts in one place makes them easy to audit, version, and A/B
test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage

# ── 1. Query analysis ─────────────────────────────────────────────────

QUERY_ANALYSIS_SYSTEM = """\
You are an expert query analyst for a Retrieval-Augmented Generation system.

Your job is to analyse the user's question and produce a JSON object with
exactly these keys:

  "intent"          – one of: "factual", "conceptual", "procedural",
                       "comparative", "exploratory", "meta"
  "key_concepts"    – list of 2-5 noun phrases that capture the core topics
  "suggested_tools" – ordered list of tool names to call
                       (choose from: vector_search, keyword_search,
                        document_lookup, web_search)
  "tool_queries"    – for each tool in suggested_tools, a concise search
                       query string tailored to that tool
  "reasoning"       – one sentence explaining your analysis

Guidelines:
- If the user mentions a specific document or source, include
  "keyword_search" or "document_lookup".
- Default to "vector_search" for open-ended questions.
- Use "web_search" only when the knowledge base is unlikely to contain
  the answer (recent events, external projects, etc.).
- Always include at least one tool.

Respond with **only** valid JSON — no markdown fences, no commentary.
"""


def build_query_analysis_prompt(query: str) -> list[BaseMessage]:
    """Build the prompt for the ``analyze_query`` node."""
    return [
        SystemMessage(content=QUERY_ANALYSIS_SYSTEM),
        HumanMessage(content=f"User question: {query}"),
    ]


# ── 2. Document grading ───────────────────────────────────────────────

GRADING_SYSTEM = """\
You are a relevance judge for a RAG system.

Given a user question and a set of retrieved documents, evaluate whether
the documents contain **enough information** to produce a good answer.

Respond with a JSON object:

  "verdict"  – "sufficient" or "insufficient"
  "missing"  – a brief description of what information is still needed
               (empty string when verdict is "sufficient")
  "refined_query" – if insufficient, suggest a better search query;
                     otherwise empty string

Respond with **only** valid JSON.
"""


def build_grading_prompt(query: str, documents: list[Document]) -> list[BaseMessage]:
    """Build the prompt for the ``grade_results`` node."""
    context = _format_documents_for_prompt(documents)
    return [
        SystemMessage(content=GRADING_SYSTEM),
        HumanMessage(
            content=(
                f"Question: {query}\n\n"
                f"Retrieved documents:\n{context}\n\n"
                "Are these documents sufficient to answer the question?"
            )
        ),
    ]


# ── 3. Synthesis with citations ───────────────────────────────────────

SYNTHESIS_SYSTEM = """\
You are a precise, helpful assistant. Answer the user's question using
**only** the provided context documents.

Rules:
1. Cite every factual claim with a bracketed reference like [1], [2], etc.
2. At the end of your answer, include a "Sources" section listing each
   reference number with its source identifier.
3. If the context is insufficient, say so honestly and explain what is
   missing — do NOT fabricate information.
4. Be concise but thorough.
5. When the context contains conflicting information, acknowledge the
   conflict and present both sides with their respective citations.

Example format:

Kubeflow Pipelines enable reproducible ML workflows [1]. They use
Argo Workflows under the hood [2].

**Sources**
[1] docs/kubeflow-overview.md §3
[2] docs/architecture.md §7
"""


def build_synthesis_prompt(
    query: str,
    documents: list[Document],
    reasoning_trace: str = "",
) -> list[BaseMessage]:
    """Build the prompt for the ``synthesize`` node.

    Parameters
    ----------
    query:
        The user's original question.
    documents:
        All gathered context documents with citation metadata.
    reasoning_trace:
        Human-readable summary of the reasoning steps taken so far,
        so the synthesiser can mention the agent's research process
        when appropriate.
    """
    numbered_context = _format_documents_numbered(documents)
    parts = [f"Question: {query}\n"]
    if reasoning_trace:
        parts.append(f"Research process:\n{reasoning_trace}\n")
    parts.append(f"Context documents:\n{numbered_context}\n")
    parts.append(
        "Provide a detailed answer with citations. "
        "End with a **Sources** section."
    )
    return [
        SystemMessage(content=SYNTHESIS_SYSTEM),
        HumanMessage(content="\n".join(parts)),
    ]


# ── 4. Backward-compatible simple prompt ──────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions using the provided context.
If the context does not contain enough information, say so honestly.
Always cite which part of the context supports your answer.
"""


def build_rag_prompt(query: str, documents: list[Document]) -> list[BaseMessage]:
    """Assemble the prompt messages for a retrieval-augmented generation call.

    This is the **legacy** prompt kept for backward compatibility with
    callers that have not migrated to the full agentic workflow.

    Parameters
    ----------
    query:
        The user question.
    documents:
        Retrieved context chunks.

    Returns
    -------
    list[BaseMessage]
        A list of LangChain message objects ready for ``.invoke()``.
    """
    context = "\n\n---\n\n".join(doc.page_content for doc in documents)
    user_msg = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Provide a detailed answer based on the context above."
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]


# ── Helpers ────────────────────────────────────────────────────────────


def _format_documents_for_prompt(documents: list[Document]) -> str:
    """Plain listing of documents with source metadata."""
    parts: list[str] = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get("chunk_index", "?")
        parts.append(f"[Doc {i}] (source={source}, chunk={chunk})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_documents_numbered(documents: list[Document]) -> str:
    """Numbered listing suitable for citation references [1], [2], …"""
    parts: list[str] = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get("chunk_index", "?")
        score = doc.metadata.get("score")
        score_str = f", score={score:.3f}" if score is not None else ""
        parts.append(
            f"[{i}] source={source} §{chunk}{score_str}\n{doc.page_content}"
        )
    return "\n\n".join(parts)
