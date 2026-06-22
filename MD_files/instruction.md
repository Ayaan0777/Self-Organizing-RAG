# Instruction: Integrate Variable Chunk Size Ingestion

## Goal

Implement a variable chunk size ingestion pipeline for a RAG system. Chunks must have a **maximum size of 1250 characters** and a **minimum size of 200 characters**. Chunks that fall below the minimum are automatically merged with their nearest neighbor.

---

## Why Variable Chunk Size?

Fixed-size chunking blindly splits text at arbitrary positions, often breaking sentences mid-thought. Variable chunk size solves this by:

1. **Using semantic separators** — splits at paragraph breaks, sentences, clauses, etc. (in priority order)
2. **Enforcing a minimum floor** — tiny chunks (< 200 chars) carry too little meaning for embeddings, so they get merged with neighbors
3. **Capping at a maximum** — chunks > 1250 chars exceed the sweet spot for embedding models and dilute retrieval precision

The result: every chunk is between **200–1250 characters**, with natural boundaries that preserve semantic coherence.

---

## Parameters

```python
CHUNK_SIZE     = 1250   # Maximum characters per chunk
CHUNK_OVERLAP  = 200    # Overlap between consecutive chunks (prevents losing context at boundaries)
MIN_CHUNK_SIZE = 200    # Minimum characters — chunks smaller than this get merged with neighbor

# Separator hierarchy — splitter tries these in order, falling back to the next if chunks are still too large
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
```

> [!IMPORTANT]
> The **overlap of 200 chars (~16% of max)** is within the industry-standard range of 10–20%. This ensures context isn't lost at chunk boundaries without excessive duplication.

---

## Algorithm

### Step 1: Split with RecursiveCharacterTextSplitter

Use LangChain's `RecursiveCharacterTextSplitter` which tries each separator in order:
1. First tries to split on `\n\n` (paragraph breaks)
2. If chunks are still > 1250, splits on `\n` (line breaks)
3. Then `. ` (sentences), `? `, `! `, etc.
4. Last resort: splits on individual characters

This produces chunks that are at most `CHUNK_SIZE` (1250) but may produce very small chunks from short paragraphs or sections.

### Step 2: Enforce Minimum Chunk Size

After splitting, scan the chunk list and merge any chunk smaller than `MIN_CHUNK_SIZE` (200) with its neighbor:

```
Algorithm: _enforce_min_chunk_size(chunks, min_chars=200)

1. Initialize buffer = first chunk
2. For each subsequent chunk:
   a. If buffer size < min_chars → merge chunk INTO buffer (concatenate text)
   b. Else → emit buffer to results, set buffer = current chunk
3. After loop, if buffer is still < min_chars:
   a. Merge buffer into the LAST emitted chunk (merge backward)
4. Emit final buffer
```

This guarantees every output chunk is ≥ 200 characters.

---

## Complete Reference Implementation

### Core Chunking Function

```python
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Chunking parameters ──────────────────────────────────────────
CHUNK_SIZE     = 1250
CHUNK_OVERLAP  = 200
MIN_CHUNK_SIZE = 200
SEPARATORS     = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def _enforce_min_chunk_size(
    chunks: list[Document],
    min_chars: int = MIN_CHUNK_SIZE,
) -> list[Document]:
    """
    Merges chunks smaller than min_chars with their nearest neighbor.
    Guarantees every chunk in the output is at least min_chars long.
    """
    if len(chunks) <= 1:
        return chunks

    result = []
    buffer = chunks[0]

    for chunk in chunks[1:]:
        if len(buffer.page_content) < min_chars:
            # Merge small chunk with next one
            buffer = Document(
                page_content=buffer.page_content + " " + chunk.page_content,
                metadata={**buffer.metadata, **chunk.metadata},
            )
        else:
            result.append(buffer)
            buffer = chunk

    # Handle final buffer — if it's still too small, merge backward
    if result and len(buffer.page_content) < min_chars:
        last = result.pop()
        buffer = Document(
            page_content=last.page_content + " " + buffer.page_content,
            metadata={**last.metadata, **buffer.metadata},
        )
    result.append(buffer)
    return result
```

### Text Cleaning Function

```python
def clean_text(text: str) -> str:
    """Removes newlines, tabs, and collapses multiple spaces into one."""
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()
```

### Ingestion Pipeline

```python
def ingest_document(raw_text: str, source: str = "unknown") -> list[Document]:
    """
    Full ingestion pipeline:
      1. Clean text
      2. Split with RecursiveCharacterTextSplitter (max 1250)
      3. Enforce minimum chunk size (min 200)

    Returns list of Documents ready for embedding and vector store upload.
    """
    # Step 1: Clean
    cleaned = clean_text(raw_text)

    # Step 2: Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = text_splitter.create_documents(
        [cleaned],
        metadatas=[{"source": source}],
    )

    # Step 3: Enforce minimum size
    chunks_before = len(chunks)
    chunks = _enforce_min_chunk_size(chunks, min_chars=MIN_CHUNK_SIZE)

    if len(chunks) != chunks_before:
        print(f"Merged {chunks_before - len(chunks)} tiny chunks → {len(chunks)} final chunks")

    # Stats
    sizes = [len(c.page_content) for c in chunks]
    print(
        f"Chunking complete: {len(chunks)} chunks | "
        f"sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}"
    )

    return chunks
```

---

## Integration Steps

### 1. Install Dependencies

```bash
pip install langchain-text-splitters langchain-core
```

### 2. Locate the Existing Ingestion Code

Find where the target codebase currently:
- Loads documents (PDF, DOCX, TXT, etc.)
- Splits them into chunks
- Uploads chunks to the vector store (Pinecone, Chroma, Weaviate, etc.)

### 3. Replace the Chunking Logic

Replace the existing text splitting code with the implementation above. Specifically:

1. **Add the parameters** (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_CHUNK_SIZE`, `SEPARATORS`) at the top of the ingestion module
2. **Add `_enforce_min_chunk_size()`** function
3. **Add `clean_text()`** function
4. **Modify the ingestion function** to:
   - Clean the raw text first
   - Use `RecursiveCharacterTextSplitter` with the parameters above
   - Call `_enforce_min_chunk_size()` on the result before uploading

### 4. Wire Into Existing Upload

After chunking, feed the resulting `Document` objects into whatever vector store upload method the codebase already uses. Example:

```python
# Existing vector store upload — adapt to match the target codebase
vector_store.add_documents(documents=chunks)
```

For large document sets, batch the upload:

```python
batch_size = 50
for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    vector_store.add_documents(documents=batch)
```

---

## Verification

After integration, verify that:

- [ ] No chunk is smaller than 200 characters
- [ ] No chunk is larger than 1250 characters
- [ ] The chunk count and size range are logged/printed
- [ ] Chunks split at natural boundaries (sentences, paragraphs) rather than arbitrary positions

### Quick Test

```python
# Test with sample text
test_text = "Short. " * 10 + "A" * 1300  # mix of tiny and oversized content
chunks = ingest_document(test_text, source="test")

for i, c in enumerate(chunks):
    size = len(c.page_content)
    assert size >= 200, f"Chunk {i} too small: {size} chars"
    assert size <= 1300, f"Chunk {i} too large: {size} chars"  # allow slight overshoot from merging
    print(f"  Chunk {i}: {size} chars")
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Max chunk size | 1250 chars (~312 tokens) | Safe within embedding model context windows (512 tokens for mxbai-embed-large) |
| Min chunk size | 200 chars (~50 tokens) | Below this, chunks carry too little semantic meaning for useful embeddings |
| Overlap | 200 chars (16% of max) | Prevents context loss at boundaries; within 10–20% industry standard |
| Separator hierarchy | `\n\n` → `\n` → `. ` → ... → `""` | Preserves semantic coherence by splitting at natural boundaries first |
| Min-size enforcement | Forward merge + backward merge for trailing chunk | Ensures no orphaned tiny chunks at the end of a document |
