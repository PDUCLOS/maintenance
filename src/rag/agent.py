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
            params[k] = v
            if k == "subset" and v in SUBSETS:
                subset = v
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

    return f"Unsupported operation: {operation!r}. " f"Supported: {', '.join(SUPPORTED_OPS)}."


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
        return "Error: empty query. Use one of: " + ", ".join(SUPPORTED_OPS)
    if operation not in SUPPORTED_OPS:
        return f"Unsupported operation: {operation!r}. " f"Supported: {', '.join(SUPPORTED_OPS)}."
    # `params` may still carry a "subset" key (the parser reports it there
    # for callers that only care about the raw dict) — drop it, since
    # `subset` is already passed positionally and would otherwise collide.
    call_params = {k: v for k, v in params.items() if k != "subset"}
    return _query_cmapss_impl(operation, subset, **call_params)


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
        """Build a ReAct agent using the standard LangChain helper.

        Uses the local few-shot prompt (`_local_react_prompt`), not the
        generic zero-shot `hwchase17/react` hub prompt: measured baseline
        was 0/3 correct tool invocations on Mistral-7B with the hub
        prompt (iteration-limit or hallucinated answers). See
        `_local_react_prompt` docstring and PLAN.md §8.
        """
        from langchain.agents import AgentExecutor, create_react_agent

        settings.assert_apple_silicon()
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
            # "generate" was removed for LCEL-based (RunnableAgent) agents —
            # this build raises `ValueError: Got unsupported
            # early_stopping_method 'generate'` on the current langchain.
            # "force" (the class default) is the only method this agent
            # type supports.
            early_stopping_method="force",
            handle_parsing_errors=True,
            # Without this, `invoke()`'s result dict has no
            # "intermediate_steps" key, so `run()` below always reports
            # zero tool calls even when the tool was genuinely invoked.
            return_intermediate_steps=True,
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
    """A few-shot ReAct prompt, tuned for reliable tool use on a 7B model.

    The generic zero-shot `hwchase17/react` hub prompt measured 0/3 on a
    baseline of quantitative CMAPSS questions with Mistral-7B-Instruct: the
    model either hit the iteration limit or skipped the tool and
    hallucinated a plausible-looking number straight to "Final Answer".
    Small models need to see the exact Action/Action Input format worked
    through end-to-end, not just described — hence the two worked examples
    below (one tool call, one out-of-scope "no tool needed" case).
    """
    from langchain_core.prompts import PromptTemplate

    template = """Tu es un copilote technique pour la maintenance industrielle. Tu as accès à un outil pour interroger le jeu de données CMAPSS.

Réponds avec EXACTEMENT ce format, sans rien ajouter avant "Question:" ni après "Final Answer:".
N'utilise JAMAIS de backticks, de guillemets, ni de formatage markdown autour du nom de l'outil
ou de son entrée — écris-les en texte brut, exactement comme dans les exemples ci-dessous.

Question: la question posée
Thought: raisonne sur ce qu'il faut faire
Action: le nom de l'outil à utiliser, un parmi [{tool_names}] (texte brut, sans backticks)
Action Input: l'entrée de l'outil, une seule ligne de texte brut (sans backticks ni guillemets)
Observation: le résultat de l'outil
... (répète Thought/Action/Action Input/Observation autant que nécessaire)
Thought: j'ai maintenant la réponse
Final Answer: la réponse finale à la question

Outils disponibles:
{tools}

Noms des outils: {tool_names}

--- Exemple 1 (question quantitative -> utiliser l'outil) ---
Question: What is the mean of sensor_11 in FD002?
Thought: This is a quantitative question about a specific sensor and subset. I must use query_cmapss, not guess a number myself.
Action: query_cmapss
Action Input: mean_sensor subset=FD002 sensor=11
Observation: Mean of sensor_11 in FD002: 42.99.
Thought: I now have the answer
Final Answer: The mean of sensor_11 in FD002 is 42.99.

--- Exemple 2 (question quantitative simple -> utiliser l'outil) ---
Question: How many engines are in the FD001 training set?
Thought: This asks for a count of engines in a specific subset. I must use query_cmapss with unit_count, not guess.
Action: query_cmapss
Action Input: unit_count FD001
Observation: FD001 has 100 engines in the training set.
Thought: I now have the answer
Final Answer: FD001 has 100 engines in the training set.

--- Fin des exemples ---

Règle: pour toute question quantitative (moyenne, min, max, compte, cycles), utilise TOUJOURS query_cmapss d'abord. Ne devine jamais un chiffre sans l'avoir obtenu de l'outil.

Tu réponds en français sauf si la question est en anglais.

Question: {input}
Thought:{agent_scratchpad}"""
    return PromptTemplate.from_template(template)


__all__ = [
    "SUPPORTED_OPS",
    "AgentResponse",
    "CMAPSSCopilotAgent",
    "_parse_dsl_query",
    "_query_cmapss_impl",
    "query_cmapss",
]
