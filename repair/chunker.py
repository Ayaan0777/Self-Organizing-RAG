"""
Repair Chunker — SH2: Adaptive Strategy
=========================================
Single adaptive rechunking function whose parameters are chosen by
the DECIDE stage in the orchestrator, based on the failure pattern.

Chunk sizes are always DIFFERENT from ingestion (1250/200) to actually
fix the problem. The DECIDE logic selects:
  - context_insufficient → chunk_size=1500 (more context per chunk)
  - hallucination_detected → chunk_size=400 (less noise, more precise)
  - complex query failure → chunk_size=1500
  - simple query failure → chunk_size=400
  - default → chunk_size=800

All sizes are within mxbai-embed-large's 512-token context window:
  - 1800 chars = ~450 tokens (safe max)
  - 400 chars = ~100 tokens
"""
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Repair-specific separators (same hierarchy as ingestion) ────────
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def _enforce_min_size(chunks: list[Document], min_chars: int = 150) -> list[Document]:
    """
    Merges chunks smaller than min_chars with their neighbor.
    Uses a slightly lower floor than ingestion (150 vs 200) since repair
    chunks may intentionally be smaller for precision.
    """
    if len(chunks) <= 1:
        return chunks

    result = []
    buffer = chunks[0]

    for chunk in chunks[1:]:
        if len(buffer.page_content) < min_chars:
            buffer = Document(
                page_content=buffer.page_content + " " + chunk.page_content,
                metadata={**buffer.metadata, **chunk.metadata},
            )
        else:
            result.append(buffer)
            buffer = chunk

    if result and len(buffer.page_content) < min_chars:
        last = result.pop()
        buffer = Document(
            page_content=last.page_content + " " + buffer.page_content,
            metadata={**last.metadata, **buffer.metadata},
        )
    result.append(buffer)
    return result


def rechunk_adaptive(
    text: str,
    source: str,
    chunk_size: int,
    overlap: int,
    repair_reason: str = "default_repair",
) -> list[Document]:
    """
    Adaptive repair chunking — parameters chosen by the DECIDE stage.
    NEVER uses the same params as ingestion (1250/200) to ensure the
    repair actually changes the chunk boundaries.

    Args:
        text:          Full text of the source document / concatenated chunks.
        source:        Source identifier for metadata.
        chunk_size:    Chunk size selected by DECIDE stage.
        overlap:       Overlap selected by DECIDE stage.
        repair_reason: Why this chunk size was chosen (for logging/audit).

    Returns:
        List of LangChain Documents with repair metadata.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        length_function=len,
    )

    chunks = splitter.create_documents(
        [text],
        metadatas=[{
            "source": source,
            "strategy": "adaptive_repair",
            "repair_chunk_size": chunk_size,
            "repair_reason": repair_reason,
        }],
    )

    # Enforce minimum size
    min_floor = 100 if chunk_size <= 400 else 150
    chunks = _enforce_min_size(chunks, min_chars=min_floor)

    return chunks


# ── DECIDE stage: strategy selection based on failure pattern ───────

def _analyse_query_type(query: str) -> str:
    """
    Classifies query complexity using regex heuristics.
    Reuses the same logic as dynamic_k._analyse_query_complexity()
    but returns a simpler label.

    Returns: "complex", "simple", or "medium"
    """
    q = query.lower().strip()

    # Comparison / multi-part → complex
    complex_patterns = [
        r"\bcompare\b", r"\bdifference\b", r"\bvs\.?\b", r"\bversus\b",
        r"\band\b.*\band\b", r"\bboth\b", r"\beach\b", r"\ball\b",
        r"\badvantages\b.*\bdisadvantages\b", r"\bpros\b.*\bcons\b",
        r"\bexplain\b", r"\bdescribe\b", r"\boverview\b", r"\bsummar",
        r"\bwhat are\b", r"\blist\b", r"\bdetail\b", r"\bdiscuss\b",
    ]
    if any(re.search(p, q) for p in complex_patterns):
        return "complex"

    if q.count("?") >= 2:
        return "complex"

    # Specific factual → simple
    simple_patterns = [
        r"\bwho\b", r"\bwhen\b", r"\bwhere\b",
        r"\bhow (?:much|many|old|long|far)\b",
        r"\bwhat (?:year|date|time|number|name|city|country)\b",
        r"\bwhich\b", r"\bdefine\b",
    ]
    if any(re.search(p, q) for p in simple_patterns):
        return "simple"

    # Long queries → complex
    if len(q.split()) >= 15:
        return "complex"

    return "medium"


def select_repair_params(query: str, detectors_triggered: list[str]) -> dict:
    """
    Mentor's DECIDE stage — selects chunk parameters for repair based on
    the failure pattern (which detectors triggered) and query complexity.

    The repair ALWAYS uses different params than ingestion (1250/200).

    Returns:
        dict with: chunk_size, overlap, reason
    """
    # Priority 1: Detector-based decisions (most specific signal)
    if "context_insufficient" in detectors_triggered:
        return {"chunk_size": 1500, "overlap": 300, "reason": "increase_context"}

    if "hallucination_detected" in detectors_triggered:
        return {"chunk_size": 400, "overlap": 80, "reason": "reduce_noise"}

    # Priority 2: Query-complexity-based decisions
    query_type = _analyse_query_type(query)

    if query_type == "complex":
        return {"chunk_size": 1500, "overlap": 300, "reason": "complex_query"}

    if query_type == "simple":
        return {"chunk_size": 400, "overlap": 80, "reason": "simple_query"}

    # Default: medium adjustment (still different from ingestion's 1250/200)
    return {"chunk_size": 800, "overlap": 150, "reason": "default_repair"}
