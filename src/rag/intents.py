"""Question intents — guided question templates for the UI.

The copilot can answer many kinds of questions, but a user who doesn't
know the corpus may not know what to ask. The intent catalogue gives the
UI a small set of pre-defined question templates ("intents") with
minimal parameter fields. The user picks an intent, fills 1-2 fields,
and the UI generates a clean, well-formed question — much more reliable
than free-form typing for the data the corpus actually contains.

This module is pure data + helpers. It has NO dependency on LangChain,
Streamlit, or any UI library. Both `src/ui/streamlit_app.py` and
`web/demo.html` can import it to render the same form.

Adding a new intent
--------------------
Just add an entry to `INTENTS` with:
  - key (slug used in URLs / config)
  - category (one of the `Category` enum values)
  - label (UI display name)
  - description (UI helper text)
  - fields (ordered list of FieldDef with name/label/options/required)
  - question_template (str.format(**fields) → question string)

The UI renders the form fields, validates the required ones, and calls
`build_question(intent_key, **values)` to get the final question.

Categories
---------
  - LOAD_RATING   — load / capacity questions (C, C0, fatigue life)
  - LUBRICATION   — grease, oil, lubrication intervals
  - MOUNTING      — installation, alignment, fits
  - DIAGNOSIS     — vibration, noise, temperature monitoring
  - FAILURE       — wear modes, contamination, expected life
  - FREE          — free-form question on any topic
  - OUT_OF_SCOPE  — adversarial / test questions that should be refused
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Category(str, Enum):
    """Coarse category for the intent. Used for grouping in the UI."""

    LOAD_RATING = "Capacité de charge & durée de vie"
    LUBRICATION = "Lubrification & graissage"
    MOUNTING = "Montage & ajustements"
    DIAGNOSIS = "Diagnostic & surveillance"
    FAILURE = "Modes de défaillance"
    FREE = "Question libre sur les catalogues"
    OUT_OF_SCOPE = "Hors-scope (test)"

    @classmethod
    def display(cls, value: Category | str) -> str:
        """Return the human label of the category (the enum value itself)."""
        if isinstance(value, Category):
            return value.value
        return str(value)


# Catalogue of common bearing topics — used as dropdown options in the
# UI for the "free question" intent. Helps the user craft a query
# without typing from scratch.
TOPICS: tuple[tuple[str, str], ...] = (
    ("ball bearing load rating", "Capacité de charge d'un roulement à billes"),
    ("roller bearing load rating", "Capacité de charge d'un roulement à rouleaux"),
    ("fatigue life", "Durée de vie en fatigue (L10)"),
    ("static load", "Charge statique (C0)"),
    ("dynamic load", "Charge dynamique (C)"),
    ("lubrication grease", "Graissage (graisse)"),
    ("lubrication oil", "Lubrification (huile)"),
    ("re-lubrication interval", "Intervalle de regraissage"),
    ("mounting procedure", "Procédure de montage"),
    ("dismounting", "Démontage"),
    ("alignment", "Alignement"),
    ("fits and tolerances", "Ajustements & tolérances"),
    ("sealing", "Étanchéité"),
    ("vibration monitoring", "Surveillance vibratoire"),
    ("acoustic monitoring", "Surveillance acoustique"),
    ("temperature limits", "Limites de température"),
    ("contamination", "Contamination"),
    ("wear modes", "Modes d'usure"),
    ("spalling", "Écaillage"),
    ("fatigue pitting", "Pit de fatigue"),
    ("false brinelling", "Faux-brinnelling"),
    ("creep", "Fluage"),
    ("misalignment diagnosis", "Diagnostic de désalignement"),
    ("lubricant selection", "Choix du lubrifiant"),
    ("bearing storage", "Stockage des roulements"),
)

DOCUMENT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("SKF", "SKF (tous catalogues)"),
    ("Schaeffler INA FAG", "Schaeffler (INA / FAG)"),
    ("NTN-SNR", "NTN-SNR (roulements & diagnostic)"),
    ("any", "N'importe lequel (recherche large)"),
)


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
        missing = [f.name for f in self.fields if f.required and not values.get(f.name)]
        if missing:
            raise ValueError(f"Missing required field(s): {', '.join(missing)}")
        try:
            return self.question_template.format(**values)
        except KeyError as e:
            raise ValueError(f"Missing value for placeholder {e!s} in question template") from e


# ---------------------------------------------------------------------------
# Reusable field definitions
# ---------------------------------------------------------------------------

_TOPIC_FIELD = FieldDef(
    name="topic",
    label="Sujet",
    options=TOPICS,
    default="ball bearing load rating",
)
_TOPIC_FREE_FIELD = FieldDef(
    name="topic",
    label="Sujet (tu peux aussi taper un mot-clé)",
    options=TOPICS,
    required=False,
    placeholder="Ex: 'spalling', 'grease interval', 'SKF Explorer'",
)
_DOC_FIELD = FieldDef(
    name="document",
    label="Source préférée",
    options=DOCUMENT_OPTIONS,
    default="any",
)


# ---------------------------------------------------------------------------
# The intent catalogue
# ---------------------------------------------------------------------------

INTENTS: tuple[Intent, ...] = (
    # --- LOAD_RATING (capacity & life) ---
    Intent(
        key="load_basic_dynamic",
        category=Category.LOAD_RATING,
        label="Capacité de charge dynamique de base (C)",
        description="Définition, mode de calcul, facteurs d'ajustement.",
        icon="⚙️",
        fields=(_TOPIC_FIELD,),
        question_template="What is the basic dynamic load rating (C) and how is it used to size a bearing?",
    ),
    Intent(
        key="load_fatigue_life",
        category=Category.LOAD_RATING,
        label="Durée de vie en fatigue (L10)",
        description="Formule L10 = (C/P)^p, facteurs correctifs (température, fiabilité, lubrification).",
        icon="📐",
        fields=(_TOPIC_FIELD,),
        question_template="How is the rating life (L10) of a rolling bearing calculated?",
    ),
    # --- LUBRICATION ---
    Intent(
        key="lubricant_selection",
        category=Category.LUBRICATION,
        label="Choix du lubrifiant (graisse ou huile)",
        description="Critères : température, vitesse, charge, environnement.",
        icon="🛢️",
        fields=(_TOPIC_FIELD, _DOC_FIELD),
        question_template="How do I select a lubricant (grease or oil) for a {topic}?",
    ),
    Intent(
        key="re_lubrication_interval",
        category=Category.LUBRICATION,
        label="Intervalle de regraissage",
        description="Recommandations SKF / Schaeffler selon conditions d'opération.",
        icon="⏱️",
        fields=(_TOPIC_FIELD,),
        question_template="What is the recommended re-lubrication interval for a {topic}?",
    ),
    # --- MOUNTING ---
    Intent(
        key="mounting_procedure",
        category=Category.MOUNTING,
        label="Procédure de montage / démontage",
        description="Outillage, chauffage, ajustements serrés / glissants.",
        icon="🔧",
        fields=(_TOPIC_FIELD, _DOC_FIELD),
        question_template="What is the recommended procedure to mount a {topic}?",
    ),
    # --- DIAGNOSIS ---
    Intent(
        key="vibration_diagnosis",
        category=Category.DIAGNOSIS,
        label="Diagnostic vibratoire d'un roulement",
        description="Fréquences caractéristiques, FFT, seuils de sévérité.",
        icon="📈",
        fields=(_TOPIC_FIELD,),
        question_template="How do I diagnose a bearing defect through vibration analysis?",
    ),
    Intent(
        key="temperature_limits",
        category=Category.DIAGNOSIS,
        label="Limites de température de fonctionnement",
        description="Plage normale, alerte, alarme selon lubrifiant et type.",
        icon="🌡️",
        fields=(_TOPIC_FIELD,),
        question_template="What are the operating temperature limits for a {topic}?",
    ),
    # --- FAILURE ---
    Intent(
        key="failure_modes",
        category=Category.FAILURE,
        label="Modes de défaillance courants",
        description="Fatigue, contamination, lubrification, faux-brinnelling, échauffement.",
        icon="💥",
        fields=(_TOPIC_FIELD,),
        question_template="What are the most common failure modes for a {topic}?",
    ),
    # --- FREE ---
    Intent(
        key="free_question",
        category=Category.FREE,
        label="Question libre sur les catalogues",
        description="Pose n'importe quelle question sur les roulements, lubrification, montage, etc.",
        icon="📚",
        fields=(
            FieldDef(
                name="topic",
                label="Sujet (libre, en anglais de préférence)",
                kind="text",
                placeholder="Ex: 'FAG self-aligning bearing', 'SKF grease LGEP 2'",
            ),
        ),
        question_template="What does the catalogue documentation say about {topic}?",
    ),
    # --- OUT_OF_SCOPE (test) ---
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
    "DOCUMENT_OPTIONS",
    "INTENTS",
    "TOPICS",
    "Category",
    "FieldDef",
    "Intent",
    "build_question",
    "get_intent",
    "intents_by_category",
]
