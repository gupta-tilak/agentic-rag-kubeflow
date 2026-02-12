"""Unit tests for the agentic RAG workflow.

All tests run **without** Kubeflow, Chroma, or OpenAI by injecting
lightweight fakes / mocks.  The test suite validates:

- State schema construction
- Prompt construction (analysis, grading, synthesis, legacy)
- Individual node logic (analyze_query, execute_tools, grade_results, synthesize)
- Conditional routing (should_continue)
- Graph compilation and end-to-end invocation
- Tool registry and tool behaviour
- Reasoning trace visibility
- Citation extraction
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from agentic_rag.agent.graph import build_graph, create_initial_state
from agentic_rag.agent.nodes import (
    _format_reasoning_trace,
    _safe_parse_json,
    analyze_query,
    execute_tools,
    grade_results,
    should_continue,
    synthesize,
)
from agentic_rag.agent.prompts import (
    build_grading_prompt,
    build_query_analysis_prompt,
    build_rag_prompt,
    build_synthesis_prompt,
)
from agentic_rag.agent.state import (
    AgentState,
    ReasoningStep,
    SourceCitation,
    ToolCall,
)


# ── Fixtures & helpers ─────────────────────────────────────────────────


def _make_docs(n: int = 3) -> list[Document]:
    """Create *n* fake documents with metadata."""
    return [
        Document(
            page_content=f"Content of document {i}.",
            metadata={
                "source": f"docs/test-{i}.md",
                "chunk_index": i,
                "score": 0.9 - i * 0.1,
                "citation_id": f"cit-{i}",
            },
        )
        for i in range(1, n + 1)
    ]


def _make_state(
    query: str = "What is Kubeflow?",
    documents: list[Document] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Return a minimal valid state dict for testing."""
    base = create_initial_state(query)
    if documents is not None:
        base["documents"] = documents
    base.update(overrides)
    return base


def _fake_llm_response(content: str) -> MagicMock:
    """Create a mock LLM response with the given content."""
    resp = MagicMock()
    resp.content = content
    return resp


# ═══════════════════════════════════════════════════════════════════════
# State schema
# ═══════════════════════════════════════════════════════════════════════


class TestStateSchema:
    """Verify state dataclasses and initial state factory."""

    def test_create_initial_state_has_all_keys(self) -> None:
        state = create_initial_state("hello")
        expected_keys = {
            "query",
            "messages",
            "query_analysis",
            "retrieval_plan",
            "documents",
            "reasoning_trace",
            "tool_calls_made",
            "citations",
            "answer",
            "iteration",
            "max_iterations",
            "needs_more_info",
        }
        assert set(state.keys()) == expected_keys

    def test_initial_state_defaults(self) -> None:
        state = create_initial_state("q", max_iterations=5)
        assert state["query"] == "q"
        assert state["max_iterations"] == 5
        assert state["iteration"] == 0
        assert state["needs_more_info"] is False

    def test_reasoning_step_fields(self) -> None:
        step = ReasoningStep(node="test", thought="thinking", action="do", observation="saw")
        assert step.node == "test"
        assert step.thought == "thinking"

    def test_tool_call_defaults(self) -> None:
        tc = ToolCall(tool_name="vector_search")
        assert tc.tool_input == {}
        assert tc.documents_returned == 0

    def test_source_citation_fields(self) -> None:
        c = SourceCitation(citation_id="[1]", source="docs/foo.md", score=0.95)
        assert c.citation_id == "[1]"
        assert c.score == 0.95


# ═══════════════════════════════════════════════════════════════════════
# Prompt construction
# ═══════════════════════════════════════════════════════════════════════


class TestPrompts:
    """Verify each prompt builder returns well-formed messages."""

    def test_build_rag_prompt_returns_two_messages(self) -> None:
        """Legacy prompt: system + human."""
        docs = [Document(page_content="Kubeflow is an ML platform.")]
        messages = build_rag_prompt("What is Kubeflow?", docs)
        assert len(messages) == 2

    def test_build_rag_prompt_includes_context(self) -> None:
        content = "LangGraph enables agentic workflows."
        docs = [Document(page_content=content)]
        messages = build_rag_prompt("Tell me about LangGraph", docs)
        assert content in messages[-1].content

    def test_build_rag_prompt_includes_query(self) -> None:
        query = "How does retrieval work?"
        docs = [Document(page_content="Retrieval uses vector search.")]
        messages = build_rag_prompt(query, docs)
        assert query in messages[-1].content

    def test_query_analysis_prompt_mentions_tools(self) -> None:
        msgs = build_query_analysis_prompt("How does KServe work?")
        system = msgs[0].content
        assert "vector_search" in system
        assert "keyword_search" in system

    def test_grading_prompt_includes_docs(self) -> None:
        docs = _make_docs(2)
        msgs = build_grading_prompt("question?", docs)
        assert "Doc 1" in msgs[-1].content
        assert "Doc 2" in msgs[-1].content

    def test_synthesis_prompt_has_numbered_refs(self) -> None:
        docs = _make_docs(2)
        msgs = build_synthesis_prompt("question?", docs, reasoning_trace="Step 1: searched")
        human = msgs[-1].content
        assert "[1]" in human
        assert "[2]" in human
        assert "Step 1: searched" in human

    def test_synthesis_prompt_without_trace(self) -> None:
        docs = _make_docs(1)
        msgs = build_synthesis_prompt("question?", docs)
        human = msgs[-1].content
        assert "Research process" not in human


# ═══════════════════════════════════════════════════════════════════════
# Node: analyze_query
# ═══════════════════════════════════════════════════════════════════════


class TestAnalyzeQuery:
    """Test the analyze_query node with a mocked LLM."""

    def _patch_and_run(self, llm_json: dict[str, Any], query: str = "What is Kubeflow?") -> dict:
        state = _make_state(query)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _fake_llm_response(json.dumps(llm_json))
        with patch("agentic_rag.agent.nodes.get_llm", return_value=mock_llm):
            return analyze_query(state)

    def test_produces_retrieval_plan(self) -> None:
        result = self._patch_and_run(
            {
                "intent": "conceptual",
                "key_concepts": ["Kubeflow", "ML platform"],
                "suggested_tools": ["vector_search"],
                "tool_queries": ["Kubeflow overview"],
                "reasoning": "Open-ended question.",
            }
        )
        assert len(result["retrieval_plan"]) == 1
        assert result["retrieval_plan"][0]["tool"] == "vector_search"

    def test_produces_reasoning_step(self) -> None:
        result = self._patch_and_run(
            {
                "intent": "factual",
                "key_concepts": ["KServe"],
                "suggested_tools": ["vector_search", "keyword_search"],
                "tool_queries": ["KServe autoscaling", "KServe docs"],
                "reasoning": "Needs specific docs.",
            }
        )
        assert len(result["reasoning_trace"]) == 1
        step = result["reasoning_trace"][0]
        assert step.node == "analyze_query"
        assert "factual" in step.thought

    def test_multi_tool_plan(self) -> None:
        result = self._patch_and_run(
            {
                "intent": "comparative",
                "key_concepts": ["Kubeflow", "Airflow"],
                "suggested_tools": ["vector_search", "web_search"],
                "tool_queries": ["Kubeflow vs Airflow", "Airflow overview"],
                "reasoning": "Comparison needs multiple sources.",
            }
        )
        assert len(result["retrieval_plan"]) == 2
        tools = [p["tool"] for p in result["retrieval_plan"]]
        assert "vector_search" in tools
        assert "web_search" in tools

    def test_handles_malformed_json(self) -> None:
        """Falls back to vector_search when LLM returns garbage."""
        state = _make_state("bad query")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _fake_llm_response("not json at all")
        with patch("agentic_rag.agent.nodes.get_llm", return_value=mock_llm):
            result = analyze_query(state)
        assert len(result["retrieval_plan"]) >= 1
        assert result["retrieval_plan"][0]["tool"] == "vector_search"


# ═══════════════════════════════════════════════════════════════════════
# Node: execute_tools
# ═══════════════════════════════════════════════════════════════════════


class TestExecuteTools:
    """Test execute_tools with mocked tool registry."""

    def test_calls_vector_search(self) -> None:
        docs = _make_docs(2)
        fake_tool = MagicMock()
        fake_tool.invoke.return_value = docs

        state = _make_state(
            retrieval_plan=[{"tool": "vector_search", "query": "Kubeflow"}],
        )
        with patch("agentic_rag.agent.nodes.TOOL_REGISTRY", {"vector_search": fake_tool}):
            result = execute_tools(state)

        assert len(result["documents"]) == 2
        assert len(result["tool_calls_made"]) == 1
        assert result["tool_calls_made"][0].tool_name == "vector_search"

    def test_deduplicates_documents(self) -> None:
        """Same document from two tools should appear only once."""
        doc = Document(page_content="unique content", metadata={"source": "a.md"})
        fake_tool = MagicMock()
        fake_tool.invoke.return_value = [doc]

        state = _make_state(
            retrieval_plan=[
                {"tool": "vector_search", "query": "q"},
                {"tool": "keyword_search", "query": "q"},
            ],
        )
        with patch(
            "agentic_rag.agent.nodes.TOOL_REGISTRY",
            {"vector_search": fake_tool, "keyword_search": fake_tool},
        ):
            result = execute_tools(state)

        # Only one copy despite two identical retrievals
        assert len(result["documents"]) == 1

    def test_handles_unknown_tool(self) -> None:
        state = _make_state(
            retrieval_plan=[{"tool": "nonexistent_tool", "query": "q"}],
        )
        with patch("agentic_rag.agent.nodes.TOOL_REGISTRY", {}):
            result = execute_tools(state)
        assert len(result["documents"]) == 0
        assert "not found" in result["reasoning_trace"][0].observation

    def test_handles_tool_exception(self) -> None:
        broken_tool = MagicMock()
        broken_tool.invoke.side_effect = RuntimeError("connection refused")

        state = _make_state(
            retrieval_plan=[{"tool": "vector_search", "query": "q"}],
        )
        with patch("agentic_rag.agent.nodes.TOOL_REGISTRY", {"vector_search": broken_tool}):
            result = execute_tools(state)
        assert len(result["tool_calls_made"]) == 1
        assert "ERROR" in result["tool_calls_made"][0].result_summary

    def test_increments_iteration(self) -> None:
        state = _make_state(
            retrieval_plan=[],
            iteration=1,
        )
        with patch("agentic_rag.agent.nodes.TOOL_REGISTRY", {}):
            result = execute_tools(state)
        assert result["iteration"] == 2


# ═══════════════════════════════════════════════════════════════════════
# Node: grade_results
# ═══════════════════════════════════════════════════════════════════════


class TestGradeResults:
    """Test the LLM-based grading node."""

    def test_sufficient_verdict(self) -> None:
        state = _make_state(documents=_make_docs(3), iteration=1)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _fake_llm_response(
            json.dumps({"verdict": "sufficient", "missing": "", "refined_query": ""})
        )
        with patch("agentic_rag.agent.nodes.get_llm", return_value=mock_llm):
            result = grade_results(state)
        assert result["needs_more_info"] is False

    def test_insufficient_triggers_retry(self) -> None:
        state = _make_state(documents=_make_docs(1), iteration=0, max_iterations=3)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _fake_llm_response(
            json.dumps({
                "verdict": "insufficient",
                "missing": "Need details about autoscaling",
                "refined_query": "KServe autoscaling configuration",
            })
        )
        with patch("agentic_rag.agent.nodes.get_llm", return_value=mock_llm):
            result = grade_results(state)
        assert result["needs_more_info"] is True
        assert len(result["retrieval_plan"]) == 1
        assert "autoscaling" in result["retrieval_plan"][0]["query"]

    def test_insufficient_at_max_iter_stops(self) -> None:
        state = _make_state(documents=_make_docs(1), iteration=3, max_iterations=3)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _fake_llm_response(
            json.dumps({"verdict": "insufficient", "missing": "more", "refined_query": "q2"})
        )
        with patch("agentic_rag.agent.nodes.get_llm", return_value=mock_llm):
            result = grade_results(state)
        # Hit the cap — don't loop again
        assert result["needs_more_info"] is False

    def test_no_documents_retries(self) -> None:
        state = _make_state(documents=[], iteration=0, max_iterations=3)
        result = grade_results(state)
        assert result["needs_more_info"] is True


# ═══════════════════════════════════════════════════════════════════════
# Node: synthesize
# ═══════════════════════════════════════════════════════════════════════


class TestSynthesize:
    """Test answer synthesis with citations."""

    def test_produces_answer_and_citations(self) -> None:
        docs = _make_docs(2)
        state = _make_state(documents=docs, reasoning_trace=[])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _fake_llm_response(
            "Kubeflow is an ML platform [1]. It runs on Kubernetes [2].\n\n"
            "**Sources**\n[1] docs/test-1.md §1\n[2] docs/test-2.md §2"
        )
        with patch("agentic_rag.agent.nodes.get_llm", return_value=mock_llm):
            result = synthesize(state)

        assert "Kubeflow" in result["answer"]
        assert len(result["citations"]) == 2
        assert result["citations"][0].citation_id == "[1]"
        assert result["citations"][0].source == "docs/test-1.md"

    def test_adds_ai_message(self) -> None:
        state = _make_state(documents=_make_docs(1))
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _fake_llm_response("Answer.")
        with patch("agentic_rag.agent.nodes.get_llm", return_value=mock_llm):
            result = synthesize(state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    def test_appends_reasoning_step(self) -> None:
        state = _make_state(documents=_make_docs(1))
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _fake_llm_response("Answer.")
        with patch("agentic_rag.agent.nodes.get_llm", return_value=mock_llm):
            result = synthesize(state)
        assert any(s.node == "synthesize" for s in result["reasoning_trace"])


# ═══════════════════════════════════════════════════════════════════════
# Routing: should_continue
# ═══════════════════════════════════════════════════════════════════════


class TestRouting:
    """Test the conditional edge function."""

    def test_routes_to_execute_tools_when_more_info(self) -> None:
        state = _make_state(needs_more_info=True)
        assert should_continue(state) == "execute_tools"

    def test_routes_to_synthesize_when_sufficient(self) -> None:
        state = _make_state(needs_more_info=False)
        assert should_continue(state) == "synthesize"


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════


class TestHelpers:
    """Test internal utility functions."""

    def test_safe_parse_json_valid(self) -> None:
        data = _safe_parse_json('{"intent": "factual"}', fallback_query="q")
        assert data["intent"] == "factual"

    def test_safe_parse_json_markdown_wrapped(self) -> None:
        data = _safe_parse_json('```json\n{"intent": "x"}\n```', fallback_query="q")
        assert data["intent"] == "x"

    def test_safe_parse_json_fallback(self) -> None:
        data = _safe_parse_json("garbage", fallback_query="q")
        assert data["suggested_tools"] == ["vector_search"]

    def test_format_reasoning_trace(self) -> None:
        steps = [
            ReasoningStep(node="a", thought="T1", action="A1", observation="O1"),
            ReasoningStep(node="b", thought="T2"),
        ]
        text = _format_reasoning_trace(steps)
        assert "Step 1 [a]: T1" in text
        assert "Step 2 [b]: T2" in text

    def test_format_empty_trace(self) -> None:
        assert _format_reasoning_trace([]) == ""


# ═══════════════════════════════════════════════════════════════════════
# Tool registry
# ═══════════════════════════════════════════════════════════════════════


class TestToolRegistry:
    """Verify tool definitions and registry."""

    def test_all_tools_registered(self) -> None:
        from agentic_rag.agent.tools import TOOL_REGISTRY

        assert "vector_search" in TOOL_REGISTRY
        assert "keyword_search" in TOOL_REGISTRY
        assert "document_lookup" in TOOL_REGISTRY
        assert "web_search" in TOOL_REGISTRY

    def test_web_search_stub_returns_document(self) -> None:
        from agentic_rag.agent.tools import web_search

        results = web_search.invoke({"query": "latest news"})
        assert len(results) == 1
        assert "stub" in results[0].page_content.lower()


# ═══════════════════════════════════════════════════════════════════════
# Graph compilation
# ═══════════════════════════════════════════════════════════════════════


class TestGraphCompilation:
    """Verify the graph compiles and has expected structure."""

    def test_graph_compiles(self) -> None:
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self) -> None:
        """The compiled graph should include all four custom nodes."""
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        for expected in ("analyze_query", "execute_tools", "grade_results", "synthesize"):
            assert expected in node_names, f"Missing node: {expected}"


# ═══════════════════════════════════════════════════════════════════════
# End-to-end (mocked LLM + tools)
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """Full graph invocation with all externals mocked."""

    def test_full_invocation(self) -> None:
        """The agent should analyse → retrieve → grade → synthesise."""
        docs = _make_docs(2)

        # LLM responses for each node that calls the LLM
        analysis_json = json.dumps(
            {
                "intent": "factual",
                "key_concepts": ["Kubeflow"],
                "suggested_tools": ["vector_search"],
                "tool_queries": ["Kubeflow overview"],
                "reasoning": "Simple factual question.",
            }
        )
        grading_json = json.dumps(
            {"verdict": "sufficient", "missing": "", "refined_query": ""}
        )
        synthesis_text = (
            "Kubeflow is an ML platform [1].\n\n"
            "**Sources**\n[1] docs/test-1.md §1"
        )

        # Track which call we're on
        call_counter = {"n": 0}
        responses = [analysis_json, grading_json, synthesis_text]

        def llm_side_effect(prompt: Any) -> MagicMock:
            idx = min(call_counter["n"], len(responses) - 1)
            call_counter["n"] += 1
            return _fake_llm_response(responses[idx])

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = llm_side_effect

        fake_tool = MagicMock()
        fake_tool.invoke.return_value = docs

        with (
            patch("agentic_rag.agent.nodes.get_llm", return_value=mock_llm),
            patch("agentic_rag.agent.nodes.TOOL_REGISTRY", {"vector_search": fake_tool}),
        ):
            graph = build_graph()
            state = create_initial_state("What is Kubeflow?")
            result = graph.invoke(state)

        # Answer produced
        assert "Kubeflow" in result["answer"]

        # Citations populated
        assert len(result["citations"]) >= 1

        # Reasoning trace has multiple steps
        assert len(result["reasoning_trace"]) >= 3  # analyze + execute + grade + synthesize

        # Tool calls logged
        assert len(result["tool_calls_made"]) >= 1

    def test_retry_loop(self) -> None:
        """When grading says insufficient, the agent should loop and retry."""
        docs_round1 = [Document(page_content="Partial info.", metadata={"source": "a.md", "chunk_index": 0})]
        docs_round2 = _make_docs(2)

        analysis_json = json.dumps(
            {
                "intent": "factual",
                "key_concepts": ["KServe"],
                "suggested_tools": ["vector_search"],
                "tool_queries": ["KServe autoscaling"],
                "reasoning": "Needs specifics.",
            }
        )
        grading_insufficient = json.dumps(
            {
                "verdict": "insufficient",
                "missing": "Need autoscaling details",
                "refined_query": "KServe autoscaling configuration",
            }
        )
        grading_sufficient = json.dumps(
            {"verdict": "sufficient", "missing": "", "refined_query": ""}
        )
        synthesis_text = "KServe supports autoscaling [1][2].\n\n**Sources**\n[1] a.md §0\n[2] docs/test-1.md §1"

        call_counter = {"n": 0}
        responses = [
            analysis_json,           # analyze_query
            grading_insufficient,     # grade_results (1st)
            grading_sufficient,       # grade_results (2nd)
            synthesis_text,           # synthesize
        ]

        def llm_side_effect(prompt: Any) -> MagicMock:
            idx = min(call_counter["n"], len(responses) - 1)
            call_counter["n"] += 1
            return _fake_llm_response(responses[idx])

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = llm_side_effect

        tool_call_counter = {"n": 0}

        def tool_side_effect(kwargs: Any) -> list[Document]:
            tool_call_counter["n"] += 1
            if tool_call_counter["n"] == 1:
                return docs_round1
            return docs_round2

        fake_tool = MagicMock()
        fake_tool.invoke.side_effect = tool_side_effect

        with (
            patch("agentic_rag.agent.nodes.get_llm", return_value=mock_llm),
            patch("agentic_rag.agent.nodes.TOOL_REGISTRY", {"vector_search": fake_tool}),
        ):
            graph = build_graph()
            state = create_initial_state("How does KServe autoscaling work?")
            result = graph.invoke(state)

        # Agent made at least 2 tool invocations (initial + retry)
        assert fake_tool.invoke.call_count >= 2
        # Answer produced
        assert result["answer"]
        # Iteration advanced
        assert result["iteration"] >= 2
