"""Agent with Python tool calling on the CMAPSS DataFrame.

This is the differentiator of the project (see PLAN.md Option A vs B/C).
The agent can:
- run the RAG chain for documentation / conceptual questions
- execute Python code on the loaded CMAPSS DataFrame for quantitative
  questions (e.g. "What is the mean RUL for FD001 at cycle 150?")

The tool is intentionally narrow: it accepts a small DSL (not arbitrary
code) to prevent runaway code generation. The DSL is a thin wrapper around
pandas that supports a handful of operations:

    - mean / min / max / std of a sensor at a given cycle range
    - mean RUL per subset
    - unit count per subset

If the LLM asks for an unsupported operation, the tool returns a clear
"unsupported" message and the agent falls back to the RAG-only response.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import tool

from src.rag.chain import RAGChain
from src.utils.logger import logger
from src.utils.timing import timed


# --- The tool ---------------------------------------------------------------
@tool
def query_cmapss(operation: str, subset: str = "FD001", **params: float) -> str:
    """Query the CMAPSS dataset for a specific aggregate.

    Args:
        operation: one of {"mean_sensor", "mean_rul", "unit_count"}.
        subset:    one of {"FD001", "FD002", "FD003", "FD004"}.
        **params:  operation-specific parameters (sensor_name, cycle, etc.).

    Returns:
        A short natural-language answer with the numerical value, or a
        clear "unsupported operation" message.
    """
    raise NotImplementedError(
        "CMAPSS tool: to be implemented in W2. Dispatch on `operation` and "
        "return a string like 'Mean sensor_11 at cycle 100 in FD001 = 642.1'."
    )


# --- The agent --------------------------------------------------------------
@dataclass
class AgentResponse:
    answer: str
    intermediate_steps: list[tuple[str, str]]  # (tool_input, tool_output)


class CMAPSSCopilotAgent:
    """LangChain agent combining RAG + CMAPSS Python tool.

    Currently uses a ReAct-style agent. The tool set is closed
    (query_cmapss only) — no arbitrary code execution.
    """

    def __init__(self) -> None:
        self.rag = RAGChain.get()
        self.tools = [query_cmapss]
        self._agent = self._build_agent()

    def _build_agent(self):
        raise NotImplementedError(
            "Agent: to be implemented in W2. Use langchain.agents.create_react_agent "
            "or AgentExecutor.from_agent_and_tools with our RAG chain + query_cmapss."
        )

    @timed
    def run(self, question: str) -> AgentResponse:
        if not question.strip():
            raise ValueError("question must not be empty")
        result = self._agent.invoke({"input": question})
        steps = [(s[0].tool_input, s[1]) for s in result.get("intermediate_steps", [])]
        logger.info("Agent run ok (steps={})", len(steps))
        return AgentResponse(answer=result["output"], intermediate_steps=steps)


__all__ = ["CMAPSSCopilotAgent", "AgentResponse", "query_cmapss"]
