"""Agent with Python tool calling on the CMAPSS DataFrame.

This is the differentiator of the project (see PLAN.md Option A vs B/C).
The agent can:
- run the RAG chain for documentation / conceptual questions
- execute Python pandas queries on the loaded CMAPSS DataFrame for
  quantitative questions (e.g. "What is the mean RUL for FD001 at
  cycle 150?")

The tool is intentionally narrow: it accepts a small DSL (not arbitrary
code) to prevent runaway code generation. The DSL is a thin wrapper
around pandas that supports a handful of operations:

    - mean / min / max of a sensor at a given cycle range
    - mean RUL per subset
    - unit count per subset
    - mean of one sensor at exact cycle

If the LLM asks for an unsupported operation, the tool returns a clear
"unsupported operation" message and the agent falls back to the RAG-only
response.

Caching: the DataFrames for the 4 subsets are loaded once on first
agent call (lazy) and kept in memory. Reloading requires re-instantiating
the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from src.config import settings
from src.ingestion.cmapss_loader import SUBSETS, load_rul, load_train
from src.utils.logger import logger
from src.utils.timing import timed


# --- The tool ----------------------------------------------------------------

SUPPORTED_OPS = (
    "mean_sensor",
    "min_sensor",
    "max_sensor",
    "sensor_at_cycle",
    "mean_rul",
    "unit_count",
    "cycles_stats",
)


def _get_dataframe(subset: str) -> pd.DataFrame:
    """Load (and cache) the CMAPSS train DataFrame for `subset`."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset: {subset}. Expected one of {SUBSETS}.")
    return load_train(subset)


def _format_number(value: float) -> str:
    """Format a float for natural-language output (avoids 4.0e-7 style noise)."""
    if pd.isna(value):
        return "no data"
    if abs(value) < 0.01 and value != 0:
        return f"{value:.6f}"
    return f"{value:.2f}"


def _query_cmapss_impl(operation: str, subset: str = "FD001", **params: Any) -> str:
    """Core implementation of the query_cmapss tool.

    The LLM never sees this function directly — it sees the @tool-wrapped
    `query_cmapss` below. Keeping the impl separate makes it testable
    without the LangChain tool decorator.
    """
    df = _get_dataframe(subset)
    sensor = params.get("sensor")
    cycle = params.get("cycle")

    if operation == "mean_sensor":
        if not sensor:
            return "Error: 'sensor' parameter is required for mean_sensor."
        col = f"sensor_{int(sensor):02d}"
        if col not in df.columns:
            return f"Error: {col} is not a valid column."
        return f"Mean of {col} in {subset}: {_format_number(df[col].mean())}."

    if operation == "min_sensor":
        if not sensor:
            return "Error: 'sensor' parameter is required for min_sensor."
        col = f"sensor_{int(sensor):02d}"
        return f"Min of {col} in {subset}: {_format_number(df[col].min())}."

    if operation == "max_sensor":
        if not sensor:
            return "Error: 'sensor' parameter is required for max_sensor."
        col = f"sensor_{int(sensor):02d}"
        return f"Max of {col} in {subset}: {_format_number(df[col].max())}."

    if operation == "sensor_at_cycle":
        if not sensor or cycle is None:
            return "Error: 'sensor' and 'cycle' parameters are required for sensor_at_cycle."
        col = f"sensor_{int(sensor):02d}"
        sub = df[df["time_cycles"] == int(cycle)]
        if sub.empty:
            return f"No data in {subset} at cycle {int(cycle)} (max cycle is {int(df['time_cycles'].max())})."
        return f"Mean of {col} at cycle {int(cycle)} in {subset}: {_format_number(sub[col].mean())}."

    if operation == "mean_rul":
        try:
            rul = load_rul(subset)
        except FileNotFoundError as e:
            return f"Error: {e}"
        return f"Mean RUL in {subset}: {_format_number(rul.mean())} cycles."

    if operation == "unit_count":
        n = df["unit_nr"].nunique()
        return f"{subset} has {int(n)} engines in the training set."

    if operation == "cycles_stats":
        cycles_per_unit = df.groupby("unit_nr")["time_cycles"].max()
        return (
            f"Cycles per engine in {subset}: "
            f"min={int(cycles_per_unit.min())}, "
            f"max={int(cycles_per_unit.max())}, "
            f"mean={_format_number(cycles_per_unit.mean())}."
        )

    return (
        f"Unsupported operation: {operation!r}. "
        f"Supported: {', '.join(SUPPORTED_OPS)}."
    )


@tool
def query_cmapss(operation: str, subset: str = "FD001", **params: Any) -> str:
    """Query the CMAPSS dataset for a specific aggregate.

    Args:
        operation: one of {"mean_sensor", "min_sensor", "max_sensor",
            "sensor_at_cycle", "mean_rul", "unit_count", "cycles_stats"}.
        subset: one of {"FD001", "FD002", "FD003", "FD004"}.
        **params: operation-specific parameters:
            - "sensor" (int, 1-21) for sensor_* and sensor_at_cycle
            - "cycle" (int) for sensor_at_cycle

    Returns:
        A short natural-language answer with the numerical value, or a
        clear "unsupported operation" / "Error: ..." message.
    """
    return _query_cmapss_impl(operation, subset, **params)


# --- The agent ---------------------------------------------------------------

@dataclass
class AgentResponse:
    answer: str
    intermediate_steps: list[tuple[dict, str]]  # (tool_input, tool_output)


class CMAPSSCopilotAgent:
    """LangChain agent combining RAG + CMAPSS Python tool.

    Uses a ReAct-style agent. The tool set is closed (query_cmapss only)
    — no arbitrary code execution.
    """

    def __init__(self) -> None:
        # Local import to avoid loading the heavy RAG chain at import time.
        from src.rag.chain import RAGChain

        self.rag = RAGChain.get()
        self.tools = [query_cmapss]
        self._agent = self._build_agent()

    def _build_agent(self):
        """Build a ReAct agent using the standard LangChain helper."""
        from langchain import hub
        from langchain.agents import AgentExecutor, create_react_agent

        settings.assert_apple_silicon()
        # Pull the standard ReAct prompt from LangChain Hub. Local fallback
        # if hub is unreachable (e.g. offline dev).
        try:
            prompt = hub.pull("hwchase17/react")
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not pull ReAct prompt from hub: {}. Using fallback.", e)
            prompt = _local_react_prompt()

        agent = create_react_agent(
            llm=self.rag.llm,
            tools=self.tools,
            prompt=prompt,
        )
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            max_iterations=4,        # bound the agent's reasoning loop
            early_stopping_method="generate",
            handle_parsing_errors=True,
        )

    @timed
    def run(self, question: str) -> AgentResponse:
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        result = self._agent.invoke({"input": question})
        steps: list[tuple[dict, str]] = []
        for s in result.get("intermediate_steps", []):
            # ReAct AgentAction has .tool_input (str or dict) and the
            # observation is the string returned by the tool.
            try:
                tool_input = s[0].tool_input  # type: ignore[attr-defined]
            except AttributeError:
                tool_input = str(s[0])
            tool_output = s[1] if len(s) > 1 else ""
            steps.append(({"tool_input": tool_input}, str(tool_output)))
        logger.info("Agent run ok (steps={})", len(steps))
        return AgentResponse(answer=result["output"], intermediate_steps=steps)


def _local_react_prompt():
    """A minimal ReAct prompt that works offline (no hub access)."""
    from langchain_core.prompts import PromptTemplate

    template = """Tu es un copilote technique pour la maintenance industrielle.
Tu as accès à des outils pour répondre aux questions.

When responding, use this exact format:

Question: the input question
Thought: think about what to do
Action: the action to take, one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have the answer
Final Answer: the final answer to the original question

Tu réponds en français sauf si la question est en anglais.

Tools available:
{tools}

Tool names: {tool_names}

Question: {input}
Thought:{agent_scratchpad}"""
    return PromptTemplate.from_template(template)


__all__ = ["CMAPSSCopilotAgent", "AgentResponse", "query_cmapss", "SUPPORTED_OPS"]
