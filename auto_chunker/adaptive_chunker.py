"""
Adaptive Chunk Sizer
====================
Post-processes a list of chunks to ensure every chunk falls within
the target token range (200–2000 tokens).

  - Small chunks (< MIN_TOKENS) are merged with their nearest neighbor
  - Large chunks (> MAX_TOKENS) are split at sentence boundaries
  - Token count is approximated as: len(text) / 4

This is applied AFTER semantic segmentation or clustering to guarantee
consistent chunk sizes for the vector store.
"""
import re
from langchain_core.documents import Document

MIN_TOKENS = 200
MAX_TOKENS = 2000

# Approximate token count: ~4 characters per token for English text
_chars_per_token = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // _chars_per_token


def _split_at_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def adaptive_resize(chunks: list[Document]) -> list[Document]:
    """
    Resizes chunks so each one contains between MIN_TOKENS and MAX_TOKENS.

    - Merges consecutive small chunks until they reach MIN_TOKENS.
    - Splits chunks exceeding MAX_TOKENS at sentence boundaries.

    Args:
        chunks: List of Documents from semantic or clustering chunker.

    Returns:
        Resized list of Documents.
    """
    if not chunks:
        return chunks

    # ── Step 1: Merge small chunks ──
    merged = []
    buffer = chunks[0]

    for chunk in chunks[1:]:
        buf_tokens = _estimate_tokens(buffer.page_content)
        next_tokens = _estimate_tokens(chunk.page_content)

        if buf_tokens < MIN_TOKENS:
            # Merge into buffer
            buffer = Document(
                page_content=buffer.page_content + " " + chunk.page_content,
                metadata={**buffer.metadata, "resized": True},
            )
        else:
            merged.append(buffer)
            buffer = chunk

    merged.append(buffer)

    # ── Step 2: Split large chunks ──
    final = []
    for chunk in merged:
        tokens = _estimate_tokens(chunk.page_content)
        if tokens <= MAX_TOKENS:
            final.append(chunk)
            continue

        # Split at sentence boundaries to stay under MAX_TOKENS
        sentences = _split_at_sentences(chunk.page_content)
        current_parts = []
        current_len = 0

        for sent in sentences:
            sent_tokens = _estimate_tokens(sent)
            if current_len + sent_tokens > MAX_TOKENS and current_parts:
                final.append(Document(
                    page_content=" ".join(current_parts),
                    metadata={**chunk.metadata, "resized": True},
                ))
                current_parts = []
                current_len = 0
            current_parts.append(sent)
            current_len += sent_tokens

        if current_parts:
            final.append(Document(
                page_content=" ".join(current_parts),
                metadata={**chunk.metadata, "resized": True},
            ))

    return final
