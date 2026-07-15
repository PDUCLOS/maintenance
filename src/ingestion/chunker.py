"""Chunking strategies for the RAG pipeline.

Default: recursive character chunking with overlap, sized to roughly match
the LLM context. Token counts use tiktoken (cl100k_base, the same tokenizer
family used by many open models).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import tiktoken

from src.config import settings

# cl100k_base is a reasonable proxy for Mistral's tokenizer; perfect alignment
# would require the model's tokenizer, but for chunk sizing the error is <10%.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Default separators, ordered from most-preferred to least.
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",  # paragraph
    "\n",    # line
    ". ",    # sentence
    " ",     # word
    "",      # character
)


@dataclass
class Chunk:
    """A chunk of text ready for embedding."""

    chunk_id: str
    text: str
    source: str          # e.g. "cmapss:train_FD001.txt" or "pdf:skf_6205.pdf"
    metadata: dict[str, str]


def count_tokens(text: str) -> int:
    """Return the number of tokens in `text`."""
    return len(_ENCODING.encode(text))


def recursive_split(
    text: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[str]:
    """Recursive character splitter with overlap.

    Tries the largest separator first; if a chunk is still too big, recurses
    on the next separator. Adds `chunk_overlap` characters of context to
    each chunk to preserve cross-boundary meaning.
    """
    raise NotImplementedError(
        "Recursive chunker: to be implemented in W1. Recursive character "
        "split with overlap (langchain.text_splitter.RecursiveCharacterTextSplitter "
        "is the reference implementation)."
    )


def build_chunks(pages: Iterable[tuple[str, str, dict[str, str]]]) -> list[Chunk]:
    """Build Chunk objects from an iterable of (text, source, metadata).

    Each item becomes one or more Chunks after recursive splitting.
    """
    raise NotImplementedError(
        "Chunk builder: to be implemented in W1. Iterate pages, call "
        "recursive_split, assign stable chunk_ids, attach metadata."
    )


__all__ = ["Chunk", "count_tokens", "recursive_split", "build_chunks"]
