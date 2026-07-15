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

ReAct compatibility (FIX Bug #5)
--------------------------------
The ReAct agent format expects tools to take a single string as
"Action Input". We expose `query_cmapss(query: str)` and parse the
DSL string internally — that way the LLM produces a single
``Action Input:`` line that we can validate, instead of trying to
emit a multi-key dict that the standard ReAct prompt doesn't expect.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from threading import Lock
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from src.config import settings
from src.ingestion.cmapss_loader import SUBSETS, load_rul, load_train
from src.utils.logger import logger
from src.utils.timing import timed


SUPPORTED_OPS: tuple[str, ...] = (
    "mean_sensor",
    "min_sensor",
    "max_sensor",
    "sensor_at_cycle",
    "mean_rul",
    "unit_count",
    "cycles_stats",
)


# --- DSL parsing ------------------------------------------------------------

# Sensor must be 1..21
_SENSOR_RE = re.compile(r"^([1-9]|1\d|2[01])$")


def _parse_dsl_query(query: str) -> tuple[str | None, str, dict[str, str]]:
    """Parse a single-string DSL query into (operation, subset, params).

    Accepted formats (case-insensitive operation, subset is one of FD001..FD004):
        "unit_count FD001"
        "mean_sensor subset=FD002 sensor=11"
        "sensor_at_cycle subset=FD003 sensor=7 cycle=150"

    Returns (None, "FD001", {}) on parse failure so the caller can
    produce a helpful error message.
    """
    if not query or not query.strip():
        return None, "FD001", {}
    try:
        tokens = shlex.split(query)
    except ValueError:
        # Unbalanced quotes etc. — fall through
        return None, "FD001", {}
    if not tokens:
        return None, "FD001", {}

    operation = tokens[0].lower()
    subset = "FD001"
    params: dict[str, str] = {}
    for tok in tokens[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            k, v = k.strip(), v.strip()
            if k == "subset" and v in SUBSETS:
                # explicit subset=... overrides the default; we don't
                # also store it in params (the tool reads it via the
                # `subset` arg, not via params["subset"]).
                subset = v
            else:
                params[k] = v
        elif tok in SUBSETS:
            # positional subset wins over the default
            subset = tok
        # else: unknown positional token — silently ignore
    return operation, subset, params


# --- Sensor validation ------------------------------------------------------

def _resolve_sensor(sensor: Any, df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Validate `sensor` and return (col_name, None) or (None, error_msg).

    FIX Bug #4: previously only `mean_sensor` validated the column
    existence; min/max/sensor_at_cycle could raise KeyError. Now all
    sensor operations share this helper.
    """
    if sensor is None or sensor == "":
        return None, "Error: 'sensor' parameter is required."
    sensor_str = str(sensor).strip()
    if not _SENSOR_RE.match(sensor_str):
        return None, f"Error: 'sensor' must be an integer in 1..21, got {sensor_str!r}."
    col = f"sensor_{int(sensor_str):02d}"
    if col not in df.columns:
        return None, f"Error: {col!r} is not a valid column in this subset."
    return col, None


def _resolve_cycle(cycle: Any) -> tuple[int | None, str | None]:
    if cycle is None or cycle == "":
        return None, "Error: 'cycle' parameter is required."
    try:
        c = int(cycle)
    except (ValueError, TypeError):
        return None, f"Error: 'cycle' must be an integer, got {cycle!r}."
    if c < 0:
        return None, f"Error: 'cycle' must be >= 0, got {c}."
    return c, None


# --- DataFrame loader -------------------------------------------------------

_dataframe_cache: dict[str, pd.DataFrame] = {}
_cache_lock = Lock()


def _get_dataframe(subset: str) -> pd.DataFrame:
    """Load (and cache) the CMAPSS train DataFrame for `subset`."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset: {subset}. Expected one of {SUBSETS}.")
    with _cache_lock:
        if subset not in _dataframe_cache:
            _dataframe_cache[subset] = load_train(subset)
        return _dataframe_cache[subset]


# --- Core implementation ----------------------------------------------------

def _format_number(value: float) -> str:
    if pd.isna(value):
        return "no data"
    if abs(value) < 0.01 and value != 0:
        return f"{value:.6f}"
    return f"{value:.2f}"


def _query_cmapss_impl(operation: str, subset: str, **params: Any) -> str:
    """Core implementation. Public API is the @tool-decorated `query_cmapss`."""
    try:
        df = _get_dataframe(subset)
    except (ValueError, FileNotFoundError) as e:
        return f"Error: {e}"

    if operation in ("mean_sensor", "min_sensor", "max_sensor", "sensor_at_cycle"):
        col, err = _resolve_sensor(params.get("sensor"), df)
        if err:
            return err

    if operation == "mean_sensor":
        return f"Mean of {col} in {subset}: {_format_number(df[col].mean())}."

    if operation == "min_sensor":
        return f"Min of {col} in {subset}: {_format_number(df[col].min())}."

    if operation == "max_sensor":
        return f"Max of {col} in {subset}: {_format_number(df[col].max())}."

    if operation == "sensor_at_cycle":
        cycle, err = _resolve_cycle(params.get("cycle"))
        if err:
            return err
        sub = df[df["time_cycles"] == cycle]
        if sub.empty:
            return f"No data in {subset} at cycle {cycle} (max cycle is {int(df['time_cycles'].max())})."
        return f"Mean of {col} at cycle {cycle} in {subset}: {_format_number(sub[col].mean())}."

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


# --- Tool wrapper -----------------------------------------------------------

@tool
def query_cmapss(query: str) -> str:
    """Query the CMAPSS dataset. The `query` is a small DSL string.

    Format (case-insensitive on the operation name):
        "OPERATION [SUBSET] [key=value ...]"

    Examples:
        query_cmapss("unit_count FD001")
        query_cmapss("mean_sensor subset=FD002 sensor=11")
        query_cmapss("sensor_at_cycle subset=FD003 sensor=7 cycle=150")
        query_cmapss("mean_rul FD002")
        query_cmapss("cycles_stats")

    SUBSET defaults to FD001 if omitted. `sensor` must be 1..21.
    `cycle` must be a non-negative integer.
    """
    operation, subset, params = _parse_dsl_query(query)
    if operation is None:
        return (
            "Error: empty query. Use one of: "
            + ", ".join(SUPPORTED_OPS)
        )
    if operation not in SUPPORTED_OPS:
        return (
            f"Unsupported operation: {operation!r}. "
            f"Supported: {', '.join(SUPPORTED_OPS)}."
        )
    return _query_cmapss_impl(operation, subset, **params)


# --- Agent ------------------------------------------------------------------

@dataclass
class AgentResponse:
    answer: str
    intermediate_steps: list[tuple[dict, str]]  # (tool_input_dict, tool_output)


class CMAPSSCopilotAgent:
    """LangChain agent combining RAG + CMAPSS Python tool (closed DSL)."""

    def __init__(self) -> None:
        from src.rag.chain import RAGChain

        self.rag = RAGChain.get()
        self.tools = [query_cmapss]
        self._agent = self._build_agent()

    def _build_agent(self):
        """Build a ReAct agent using the standard LangChain helper."""
        from langchain import hub
        from langchain.agents import AgentExecutor, create_react_agent

        settings.assert_apple_silicon()
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
            max_iterations=4,
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
Action Input: the input to the action (a SINGLE string)
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


__all__ = [
    "CMAPSSCopilotAgent",
    "AgentResponse",
    "query_cmapss",
    "SUPPORTED_OPS",
    "_parse_dsl_query",
    "_query_cmapss_impl",
]
