import os
import re
import tempfile
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from services.llm_factory import get_vector_store


# ── Chunking parameters (shared with repair for consistency) ────────────
CHUNK_SIZE    = 1250   # max chars per chunk
CHUNK_OVERLAP = 200    # overlap between consecutive chunks
MIN_CHUNK_SIZE = 200   # minimum chars — smaller chunks are merged with neighbor
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def clean_text(text: str) -> str:
    """Removes newlines, tabs, and collapses multiple spaces into one."""
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()


def _enforce_min_chunk_size(chunks: list[Document], min_chars: int = MIN_CHUNK_SIZE) -> list[Document]:
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

    # Handle final buffer
    if result and len(buffer.page_content) < min_chars:
        # Merge trailing small chunk with the last accepted chunk
        last = result.pop()
        buffer = Document(
            page_content=last.page_content + " " + buffer.page_content,
            metadata={**last.metadata, **buffer.metadata},
        )
    result.append(buffer)
    return result


def process_and_store_file(file: UploadFile, namespace: str = None):
    """
    Ingests an uploaded file into Pinecone under the given namespace.
    Supports PDF, DOCX, and TXT files.

    Chunking: RecursiveCharacterTextSplitter with variable sizing
      - Max chunk size: 1250 chars
      - Min chunk size: 200 chars (enforced by merging small chunks)
      - Overlap: 200 chars (~16% of max, industry standard 10-20%)
    """
    print(f"\n--- 📥 STARTING INGESTION: {file.filename} ---")
    file_extension = os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name

    try:
        print(f"⏳ Extracting text from {file_extension}...")
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)
        else:
            return {"error": "Unsupported file type"}

        raw_documents = loader.load()

        # --- CLEANING STEP ---
        print("🧹 Cleaning text (removing newlines and tabs)...")
        for doc in raw_documents:
            doc.page_content = clean_text(doc.page_content)
        # -------------------------

        print("✅ Text cleaning complete.")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # ── Variable-size chunking ──────────────────────────────────────
    print(f"✂️ Splitting into chunks (max={CHUNK_SIZE}, min={MIN_CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = text_splitter.split_documents(raw_documents)

    # Enforce minimum chunk size
    chunks_before_merge = len(chunks)
    chunks = _enforce_min_chunk_size(chunks, min_chars=MIN_CHUNK_SIZE)
    if len(chunks) != chunks_before_merge:
        print(f"   🔗 Merged {chunks_before_merge - len(chunks)} tiny chunks → {len(chunks)} final chunks")

    sizes = [len(c.page_content) for c in chunks]
    print(f"   📏 Chunk sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")

    # ── Upload to Pinecone ──────────────────────────────────────────
    vector_store = get_vector_store(namespace)
    ns_label = namespace or "default"
    print(f"🚀 Uploading {len(chunks)} chunks to Pinecone (namespace={ns_label})...")

    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        current_batch = (i // batch_size) + 1
        print(f"   ⚙️ Embedding & Uploading batch {current_batch}/{total_batches}...")
        vector_store.add_documents(documents=batch)

    print(f"🎉 SUCCESS: {file.filename} is now indexed! ({len(chunks)} chunks, {min(sizes)}-{max(sizes)} chars each)\n")
    return {
        "message": f"Successfully ingested {len(chunks)} chunks (sizes: {min(sizes)}-{max(sizes)} chars).",
        "namespace": ns_label,
        "chunk_count": len(chunks),
        "chunk_size_range": f"{min(sizes)}-{max(sizes)}",
    }