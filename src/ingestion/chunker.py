"""Chunking strategies for the RAG pipeline.

Default: recursive character chunking with overlap, sized to roughly match
the LLM context. Token counts use tiktoken (cl100k_base, the same tokenizer
family used by many open models).

We implement the recursive splitter ourselves (no langchain-text-splitters
dependency) to keep the dep tree lean and the behaviour auditable. The
algorithm matches langchain's `RecursiveCharacterTextSplitter`:
  1. Try the largest separator. If a piece is still too big, recurse
     on the next smaller separator.
  2. Merge adjacent pieces greedily until adding the next would exceed
     `chunk_size`.
  3. Apply `chunk_overlap` by carrying forward the tail of the previous
     chunk into the next.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import tiktoken

from src.config import settings

# cl100k_base is a reasonable proxy for Mistral's tokenizer; perfect alignment
# would require the model's tokenizer, but for chunk sizing the error is <10%.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Default separators, ordered from most-preferred to least.
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",  # paragraph
    "\n",  # line
    ". ",  # sentence
    " ",  # word
    "",  # character
)


@dataclass
class Chunk:
    """A chunk of text ready for embedding."""

    chunk_id: str
    text: str
    source: str  # e.g. "pdf:skf_6205.pdf"
    metadata: dict[str, str]


def count_tokens(text: str) -> int:
    """Return the number of tokens in `text`."""
    return len(_ENCODING.encode(text))


def _split_once(text: str, separator: str) -> list[str]:
    """Split text by `separator`. Empty separator = list of characters.

    The separator is re-attached to each non-leading piece so the
    original formatting survives the split → re-merge.
    """
    if separator == "":
        return list(text)
    parts = text.split(separator)
    out: list[str] = []
    for i, p in enumerate(parts):
        if i == 0:
            out.append(p)
        else:
            out.append(separator + p)
    return out


def _take_tail_tokens(text: str, n_tokens: int) -> str:
    """Return the last `n_tokens` tokens of `text` (decoded back to string)."""
    if n_tokens <= 0 or not text:
        return ""
    token_ids = _ENCODING.encode(text)
    if len(token_ids) <= n_tokens:
        return text
    return _ENCODING.decode(token_ids[-n_tokens:])


def _merge_with_overlap(
    pieces: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Greedy merge of `pieces` into chunks of at most `chunk_size` tokens.

    When the current chunk would exceed `chunk_size`, it's flushed and
    the last `chunk_overlap` tokens of its text are carried forward as
    the head of the next chunk (if chunk_overlap > 0).
    """
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = current + piece
        if count_tokens(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                if chunk_overlap > 0:
                    overlap_text = _take_tail_tokens(current, chunk_overlap)
                    new_current = overlap_text + piece
                    if count_tokens(new_current) > chunk_size:
                        # The carried tail + this piece overflows: flush the
                        # tail as its own chunk (only if non-empty) and
                        # start the next chunk fresh with `piece`.
                        if overlap_text:
                            chunks.append(overlap_text)
                        current = piece
                    else:
                        current = new_current
                else:
                    current = piece
            else:
                # A single piece is too big — caller will recurse on a
                # smaller separator. For now, keep it.
                current = piece
    if current:
        chunks.append(current)
    return chunks


def recursive_split(
    text: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[str]:
    """Recursive character splitter with overlap.

    Tries the largest separator first; if a piece is still too big, recurses
    on the next separator. Adds `chunk_overlap` tokens of context to each
    chunk to preserve cross-boundary meaning.
    """
    if not text or not text.strip():
        return []
    if count_tokens(text) <= chunk_size:
        return [text]

    # Pick the first separator that actually occurs in the text
    sep = ""
    for s in separators:
        if s == "" or s in text:
            sep = s
            break

    pieces = _split_once(text, sep)

    # If a piece is still too big, recurse on the next-smaller separator.
    # The terminal "" separator always shrinks to 1-character pieces.
    refined: list[str] = []
    for p in pieces:
        if count_tokens(p) <= chunk_size:
            refined.append(p)
        else:
            if sep == "":
                # Hard cap reached: keep as-is, the merge step will flush it.
                refined.append(p)
            else:
                remaining = tuple(s for s in separators if s != sep) or ("",)
                refined.extend(recursive_split(p, chunk_size, chunk_overlap, remaining))

    return _merge_with_overlap(refined, chunk_size, chunk_overlap)


def build_chunks(pages: Iterable[tuple[str, str, dict[str, str]]]) -> list[Chunk]:
    """Build Chunk objects from an iterable of (text, source, metadata).

    Each item becomes one or more Chunks after recursive splitting.
    The chunk_id format is "<source>:<index>" so it's stable and unique
    within a source. Re-ingesting the same source produces the same ids,
    which lets ChromaDB upsert replace in place (idempotent).
    """
    chunks: list[Chunk] = []
    for text, source, metadata in pages:
        if not text or not text.strip():
            continue
        sub_texts = recursive_split(text)
        for j, sub in enumerate(sub_texts):
            chunk_id = f"{source}:{j}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=sub,
                    source=source,
                    # chunk_id is also kept in metadata so the BM25 retriever
                    # can map its hits back to Chroma entries by the same key.
                    metadata={**metadata, "chunk_id": chunk_id, "chunk_index": str(j)},
                )
            )
    return chunks


__all__ = ["Chunk", "build_chunks", "count_tokens", "recursive_split"]
