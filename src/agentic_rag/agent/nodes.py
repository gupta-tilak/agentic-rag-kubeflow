"""Graph nodes — each function is one step in the agentic RAG workflow.

Node contract
-------------
* Accepts the full :class:`AgentState` dict.
* Returns a *partial* dict with **only the keys that changed**.
* Must be a pure function of its input (+ LLM calls); no hidden global
  state so that every node is independently testable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from agentic_rag.agent.llm import get_llm
from agentic_rag.agent.prompts import (
    build_grading_prompt,
    build_query_analysis_prompt,
    build_synthesis_prompt,
)
from agentic_rag.agent.state import (
    AgentState,
    ReasoningStep,
    SourceCitation,
    ToolCall,
)
from agentic_rag.agent.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


# ── 1. ANALYZE QUERY ──────────────────────────────────────────────────


def analyze_query(state: AgentState) -> dict[str, Any]:
    """Interpret user intent, extract key concepts, suggest tools.

    This node calls the LLM with a structured JSON-output prompt.
    It produces:
    * ``query_analysis`` — the parsed JSON dict
    * ``retrieval_plan`` — list of ``{"tool", "query", "reason"}``
    * a :class:`ReasoningStep` appended to ``reasoning_trace``
    """
    query = state["query"]
    llm = get_llm(temperature=0.0)
    prompt = build_query_analysis_prompt(query)
    response = llm.invoke(prompt)

    # Parse the structured JSON response
    analysis = _safe_parse_json(response.content, fallback_query=query)

    # Build a retrieval plan from the analysis
    tools = analysis.get("suggested_tools", ["vector_search"])
    queries = analysis.get("tool_queries", [query] * len(tools))
    # Pad queries in case the LLM returned fewer than tools
    while len(queries) < len(tools):
        queries.append(query)

    plan = [
        {"tool": t, "query": q, "reason": analysis.get("reasoning", "")}
        for t, q in zip(tools, queries)
    ]

    step = ReasoningStep(
        node="analyze_query",
        thought=f"Intent: {analysis.get('intent', 'unknown')}. "
        f"Key concepts: {analysis.get('key_concepts', [])}.",
        action=f"Plan {len(plan)} retrieval step(s): {[p['tool'] for p in plan]}",
    )

    return {
        "query_analysis": analysis,
        "retrieval_plan": plan,
        "reasoning_trace": [step],
    }


# ── 2. EXECUTE TOOLS ──────────────────────────────────────────────────


def execute_tools(state: AgentState) -> dict[str, Any]:
    """Execute every tool call listed in the retrieval plan.

    Collects documents from all tools and deduplicates by page_content
    hash so repeated chunks don't pollute the context window.
    """
    plan = state.get("retrieval_plan", [])
    existing_docs: list[Document] = list(state.get("documents", []))
    seen_contents: set[int] = {hash(d.page_content) for d in existing_docs}

    new_docs: list[Document] = []
    tool_log: list[ToolCall] = []
    observations: list[str] = []

    for step in plan:
        tool_name = step.get("tool", "vector_search")
        tool_query = step.get("query", state["query"])
        tool_fn = TOOL_REGISTRY.get(tool_name)

        if tool_fn is None:
            logger.warning("Unknown tool %r in plan — skipping", tool_name)
            observations.append(f"⚠ tool {tool_name!r} not found")
            continue

        try:
            kwargs: dict[str, Any] = {"query": tool_query}
            # document_lookup uses "source" instead of "query"
            if tool_name == "document_lookup":
                kwargs = {"source": tool_query}
            elif tool_name == "keyword_search" and "source_filter" in step:
                kwargs["source_filter"] = step["source_filter"]

            result_docs: list[Document] = tool_fn.invoke(kwargs)

            for doc in result_docs:
                h = hash(doc.page_content)
                if h not in seen_contents:
                    seen_contents.add(h)
                    new_docs.append(doc)

            tool_log.append(
                ToolCall(
                    tool_name=tool_name,
                    tool_input=kwargs,
                    result_summary=f"{len(result_docs)} document(s)",
                    documents_returned=len(result_docs),
                )
            )
            observations.append(
                f"{tool_name}({tool_query!r}) → {len(result_docs)} docs"
            )
        except Exception as exc:
            logger.exception("Tool %s failed", tool_name)
            observations.append(f"⚠ {tool_name} error: {exc}")
            tool_log.append(
                ToolCall(
                    tool_name=tool_name,
                    tool_input={"query": tool_query},
                    result_summary=f"ERROR: {exc}",
                    documents_returned=0,
                )
            )

    step = ReasoningStep(
        node="execute_tools",
        thought=f"Executing {len(plan)} planned tool call(s).",
        action="Run tools",
        observation="; ".join(observations),
    )

    all_docs = existing_docs + new_docs
    return {
        "documents": all_docs,
        "tool_calls_made": tool_log,
        "reasoning_trace": [step],
        "iteration": state.get("iteration", 0) + 1,
    }


# ── 3. GRADE RESULTS ──────────────────────────────────────────────────


def grade_results(state: AgentState) -> dict[str, Any]:
    """Evaluate whether gathered documents are sufficient to answer.

    Uses the LLM as a judge.  When evidence is insufficient *and* we
    haven't hit the iteration cap, the node sets ``needs_more_info``
    and populates a new ``retrieval_plan`` with a refined query.
    """
    documents = state.get("documents", [])
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    # Fast-path: no documents at all
    if not documents:
        return {
            "needs_more_info": iteration < max_iter,
            "retrieval_plan": [{"tool": "vector_search", "query": state["query"]}],
            "reasoning_trace": [
                ReasoningStep(
                    node="grade_results",
                    thought="No documents retrieved yet.",
                    action="Retry with original query",
                )
            ],
        }

    llm = get_llm(temperature=0.0)
    prompt = build_grading_prompt(state["query"], documents)
    response = llm.invoke(prompt)
    grading = _safe_parse_json(
        response.content,
        fallback_query=state["query"],
        default_verdict="sufficient",
    )

    verdict = grading.get("verdict", "sufficient")
    needs_more = verdict != "sufficient" and iteration < max_iter

    new_plan: list[dict[str, Any]] = []
    if needs_more:
        refined = grading.get("refined_query", state["query"])
        new_plan = [{"tool": "vector_search", "query": refined, "reason": grading.get("missing", "")}]

    step = ReasoningStep(
        node="grade_results",
        thought=f"Verdict: {verdict}. Missing: {grading.get('missing', 'nothing')}.",
        action="Refine and retry" if needs_more else "Proceed to synthesis",
    )

    return {
        "needs_more_info": needs_more,
        "retrieval_plan": new_plan,
        "reasoning_trace": [step],
    }


# ── 4. SYNTHESIZE ─────────────────────────────────────────────────────


def synthesize(state: AgentState) -> dict[str, Any]:
    """Generate the final answer with inline citations.

    Post-processes the LLM output to extract structured
    :class:`SourceCitation` objects.
    """
    documents = state.get("documents", [])
    trace_text = _format_reasoning_trace(state.get("reasoning_trace", []))

    llm = get_llm(temperature=0.0)
    prompt = build_synthesis_prompt(state["query"], documents, reasoning_trace=trace_text)
    response = llm.invoke(prompt)
    answer = response.content

    # Build structured citations from the documents that were provided
    citations: list[SourceCitation] = []
    for i, doc in enumerate(documents, 1):
        citations.append(
            SourceCitation(
                citation_id=f"[{i}]",
                source=doc.metadata.get("source", "unknown"),
                chunk_index=doc.metadata.get("chunk_index"),
                score=doc.metadata.get("score"),
                excerpt=doc.page_content[:200],
            )
        )

    step = ReasoningStep(
        node="synthesize",
        thought="Generating cited answer from gathered evidence.",
        action="Produce final answer",
        observation=f"Answer length: {len(answer)} chars, {len(citations)} citation(s).",
    )

    return {
        "answer": answer,
        "citations": citations,
        "messages": [AIMessage(content=answer)],
        "reasoning_trace": [step],
    }


# ── 5. ROUTING (conditional edge) ─────────────────────────────────────


def should_continue(state: AgentState) -> str:
    """Conditional edge after ``grade_results``.

    Returns
    -------
    str
        ``"execute_tools"`` when more retrieval is needed,
        ``"synthesize"`` when we have enough evidence.
    """
    if state.get("needs_more_info", False):
        return "execute_tools"
    return "synthesize"


# ── Backward-compatible node wrappers ──────────────────────────────────


def retrieve(state: AgentState) -> dict[str, Any]:
    """**Legacy** retrieve node — wraps the new ``execute_tools``.

    Kept so that callers importing ``retrieve`` from the old API
    continue to work.
    """
    # Synthesize a minimal plan and delegate
    plan_state: AgentState = {  # type: ignore[typeddict-item]
        **state,
        "retrieval_plan": [{"tool": "vector_search", "query": state["query"]}],
        "documents": state.get("documents", []),
        "iteration": state.get("iteration", 0),
    }
    return execute_tools(plan_state)


def generate(state: AgentState) -> dict[str, Any]:
    """**Legacy** generate node — wraps the new ``synthesize``."""
    return synthesize(state)


def grade_documents(state: AgentState) -> str:
    """**Legacy** conditional edge — wraps the new ``should_continue``."""
    if not state.get("documents"):
        return "retrieve"
    return "generate"


# ── Internal helpers ───────────────────────────────────────────────────


def _safe_parse_json(
    text: str,
    *,
    fallback_query: str = "",
    default_verdict: str = "sufficient",
) -> dict[str, Any]:
    """Best-effort JSON parsing with graceful fallback.

    LLMs occasionally return JSON wrapped in markdown fences or with
    trailing commentary.  This helper strips common wrappers before
    parsing.
    """
    cleaned = text.strip()
    # Strip ```json … ``` wrappers
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n") if "\n" in cleaned else 3
        cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM JSON, using fallback: %.200s", text)
        return {
            "intent": "unknown",
            "key_concepts": [],
            "suggested_tools": ["vector_search"],
            "tool_queries": [fallback_query],
            "reasoning": "Failed to parse LLM response — defaulting to vector search.",
            "verdict": default_verdict,
            "missing": "",
            "refined_query": fallback_query,
        }


def _format_reasoning_trace(trace: list[ReasoningStep]) -> str:
    """Render the reasoning trace as human-readable text."""
    if not trace:
        return ""
    lines: list[str] = []
    for i, step in enumerate(trace, 1):
        lines.append(f"Step {i} [{step.node}]: {step.thought}")
        if step.action:
            lines.append(f"  → Action: {step.action}")
        if step.observation:
            lines.append(f"  → Observation: {step.observation}")
    return "\n".join(lines)
