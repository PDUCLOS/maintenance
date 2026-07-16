"""Agent module — kept as a stub for backward compatibility.

The original closed-DSL tool-calling agent was removed in July 2026
when the project pivoted to a 100% catalogue (Schaeffler/SKF) focus.
There is no quantitative data to query, so no tool is needed.

The RAG chain alone (in `src.rag.chain`) is now sufficient to answer
all questions from the PDF corpus.

This module is preserved so that imports like
`from src.rag.agent import CMAPSSCopilotAgent` keep working in case
downstream code (or a stale test) references them. All public
functions raise a clear `NotImplementedError`.

If you ever want to bring back tool calling (e.g. to query a tabular
bearing spec table), see the git history for the original ReAct
implementation and the unit tests in tests/test_*.py that exercised it.
"""

from __future__ import annotations

from dataclasses import dataclass

_MSG = (
    "The closed-DSL tool-calling agent was removed when the project "
    "pivoted to 100% catalogue focus. Use the RAG chain (RAGChain.get()) "
    "directly. See src/rag/agent.py docstring for context."
)


@dataclass
class AgentResponse:
    """Stub for backward compatibility — never instantiated."""

    answer: str
    intermediate_steps: list


def _removed(*_args, **_kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError(_MSG)


# --- Public API stubs --------------------------------------------------------

SUPPORTED_OPS: tuple[str, ...] = ()  # was ("mean_sensor", ...) before
query_cmapss = _removed  # was a LangChain @tool
CMAPSSCopilotAgent = _removed  # was a class; now a function that raises
_parse_dsl_query = _removed
_query_cmapss_impl = _removed


__all__ = [
    "SUPPORTED_OPS",
    "AgentResponse",
    "CMAPSSCopilotAgent",
    "_parse_dsl_query",
    "_query_cmapss_impl",
    "query_cmapss",
]
