"""Question intents — guided question templates for the UI.

The copilot can answer many kinds of questions, but a user who doesn't
know the corpus may not know what to ask. The intent catalogue gives the
UI a small set of pre-defined question templates ("intents") with
minimal parameter fields. The user picks an intent, fills 1-3 fields,
and the UI generates a clean, well-formed question — much more
reliable than free-form typing for the data the corpus actually
contains.

This module is pure data + helpers. It has NO dependency on LangChain,
Streamlit, or any UI library. Both `src/ui/streamlit_app.py` and
`web/demo.html` can import it to render the same form.

Adding a new intent
--------------------
Just add an entry to `INTENTS` with:
  - key (slug used in URLs / config)
  - category (CMAPSS_FACTUAL, CMAPSS_AGENT, INDUSTRIAL, OUT_OF_SCOPE)
  - label (UI display name)
  - description (UI helper text)
  - fields (ordered list of FieldDef with name/label/options/required)
  - question_template (str.format(**fields) → question string)

The UI renders the form fields, validates the required ones, and calls
`build_question(intent_key, **values)` to get the final question.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Category(str, Enum):
    """Coarse category for the intent. Used for grouping in the UI."""

    CMAPSS_FACTUAL = "CMAPSS — factual"
    CMAPSS_REASONING = "CMAPSS — reasoning"
    CMAPSS_AGENT = "CMAPSS — agent tool calling"
    INDUSTRIAL = "Industrial catalogue (Schaeffler / SKF)"
    OUT_OF_SCOPE = "Out of scope (test)"

    @classmethod
    def display(cls, value: Category | str) -> str:
        """Return the human label of the category (the enum value itself)."""
        if isinstance(value, Category):
            return value.value
        return str(value)


# Catalogue of CMAPSS subsets and sensor IDs (used in the form widgets).
CMAPSS_SUBSETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")
CMAPSS_SENSORS: tuple[str, ...] = tuple(f"sensor_{i:02d}" for i in range(1, 22))
SENSOR_LABEL = "Capteur (1-21)"


@dataclass(frozen=True)
class FieldDef:
    """Definition of one form field in an intent."""

    name: str
    label: str
    kind: str = "select"  # "select" | "number" | "text"
    options: tuple[tuple[str, str], ...] = ()  # (value, label) for select
    required: bool = True
    placeholder: str = ""
    default: str = ""
    min_value: int | None = None
    max_value: int | None = None


@dataclass(frozen=True)
class Intent:
    """One pre-defined question template the user can pick from."""

    key: str
    category: Category
    label: str
    description: str
    fields: tuple[FieldDef, ...]
    question_template: str
    # Optional: a small icon/emoji for the UI
    icon: str = "💬"

    def build_question(self, **values: Any) -> str:
        """Render the question from the user's inputs.

        Missing required fields raise ValueError with a clear message.
        """
        missing = [
            f.name for f in self.fields
            if f.required and not values.get(f.name)
        ]
        if missing:
            raise ValueError(
                f"Missing required field(s): {', '.join(missing)}"
            )
        try:
            return self.question_template.format(**values)
        except KeyError as e:
            raise ValueError(
                f"Missing value for placeholder {e!s} in question template"
            ) from e


# ---------------------------------------------------------------------------
# The intent catalogue
# ---------------------------------------------------------------------------

# Field definitions reused across intents
_SUBSET_FIELD = FieldDef(
    name="subset",
    label="Subset CMAPSS",
    options=((s, s) for s in CMAPSS_SUBSETS),
    default="FD001",
)
_SENSOR_FIELD = FieldDef(
    name="sensor",
    label="Capteur (sensor_01 → sensor_21)",
    options=((s, s) for s in CMAPSS_SENSORS),
    default="sensor_11",
)
_CYCLE_FIELD = FieldDef(
    name="cycle",
    label="Cycle exact",
    kind="number",
    default="150",
    min_value=1,
    max_value=400,
    placeholder="Ex: 150",
)


INTENTS: tuple[Intent, ...] = (
    # --- CMAPSS — factual (RAG sur le texte sérialisé) ---
    Intent(
        key="cmapss_unit_count",
        category=Category.CMAPSS_FACTUAL,
        label="Combien de moteurs dans un subset ?",
        description="Statistique de base : nombre d'unités dans le training set.",
        icon="🔢",
        fields=(_SUBSET_FIELD,),
        question_template="How many turbofan engines are in the {subset} training set?",
    ),
    Intent(
        key="cmapss_max_cycles",
        category=Category.CMAPSS_FACTUAL,
        label="Nombre max de cycles (par unité) dans un subset",
        description="Durée de vie max enregistrée pour une unité du subset.",
        icon="🔢",
        fields=(_SUBSET_FIELD,),
        question_template="What is the maximum number of cycles observed for any unit in {subset}?",
    ),
    Intent(
        key="cmapss_sensor_mean",
        category=Category.CMAPSS_FACTUAL,
        label="Moyenne d'un capteur (tous cycles)",
        description="Moyenne arithmétique d'un capteur sur l'ensemble du subset.",
        icon="📊",
        fields=(_SUBSET_FIELD, _SENSOR_FIELD),
        question_template="What is the mean of {sensor} across all cycles in {subset}?",
    ),

    # --- CMAPSS — reasoning (trend) ---
    Intent(
        key="cmapss_sensor_trend",
        category=Category.CMAPSS_REASONING,
        label="Tendance d'un capteur (augmente / diminue / stable)",
        description="Le capteur monte, descend, ou reste stable avec l'usure ?",
        icon="📈",
        fields=(_SUBSET_FIELD, _SENSOR_FIELD),
        question_template=(
            "Does {sensor} tend to increase, decrease, or stay stable "
            "as the engine degrades in {subset}?"
        ),
    ),

    # --- CMAPSS — multi-hop (capteur à un cycle précis) ---
    Intent(
        key="cmapss_sensor_at_cycle",
        category=Category.CMAPSS_REASONING,
        label="Valeur moyenne d'un capteur à un cycle précis",
        description="Moyenne du capteur parmi toutes les unités à ce cycle exact.",
        icon="🎯",
        fields=(_SUBSET_FIELD, _SENSOR_FIELD, _CYCLE_FIELD),
        question_template=(
            "For {subset}, at cycle {cycle}, what is the mean of {sensor}?"
        ),
    ),

    # --- CMAPSS — agent tool calling (DSL fermé) ---
    Intent(
        key="cmapss_mean_rul",
        category=Category.CMAPSS_AGENT,
        label="Mean RUL (Remaining Useful Life)",
        description="RUL moyen du subset (nécessite l'agent avec tool calling).",
        icon="🤖",
        fields=(_SUBSET_FIELD,),
        question_template="What is the mean RUL in {subset}?",
    ),
    Intent(
        key="cmapss_min_max_sensor",
        category=Category.CMAPSS_AGENT,
        label="Min et max d'un capteur",
        description="Plage de variation d'un capteur dans le subset.",
        icon="🤖",
        fields=(_SUBSET_FIELD, _SENSOR_FIELD),
        question_template=(
            "What is the minimum and maximum of {sensor} in {subset}?"
        ),
    ),

    # --- Industrial catalogue (Schaeffler / SKF) ---
    Intent(
        key="industrial_ask",
        category=Category.INDUSTRIAL,
        label="Question sur les catalogues Schaeffler / SKF",
        description="Pose une question libre sur les roulements, montage, lubrification, etc.",
        icon="📚",
        fields=(
            FieldDef(
                name="topic",
                label="Sujet (libre, en anglais de préférence)",
                kind="text",
                placeholder="Ex: 'ball bearing load rating', 'FAG mounting procedure', 'SKF sealing'",
            ),
        ),
        question_template="What does the documentation say about {topic}?",
    ),

    # --- Out of scope (test) ---
    Intent(
        key="out_of_scope",
        category=Category.OUT_OF_SCOPE,
        label="Test hors-scope (doit dire 'I don't know')",
        description="Vérifie que le copilot refuse poliment les questions non-pertinentes.",
        icon="⚠️",
        fields=(
            FieldDef(
                name="question",
                label="Question complètement hors-scope",
                kind="text",
                placeholder="Ex: 'What is the best restaurant in Lyon?'",
            ),
        ),
        question_template="{question}",
    ),
)


def get_intent(key: str) -> Intent:
    """Return the intent with the given key. Raises KeyError if not found."""
    for i in INTENTS:
        if i.key == key:
            return i
    raise KeyError(f"Unknown intent: {key!r}. Known: {[i.key for i in INTENTS]}")


def build_question(intent_key: str, **values: Any) -> str:
    """Convenience wrapper: build the question for an intent + values."""
    return get_intent(intent_key).build_question(**values)


# Convenience: intent groups by category, in display order
def intents_by_category() -> dict[str, list[Intent]]:
    out: dict[str, list[Intent]] = {}
    for i in INTENTS:
        out.setdefault(Category.display(i.category), []).append(i)
    return out


__all__ = [
    "CMAPSS_SENSORS",
    "CMAPSS_SUBSETS",
    "INTENTS",
    "Category",
    "FieldDef",
    "Intent",
    "build_question",
    "get_intent",
    "intents_by_category",
]
