import os
import re
import tempfile
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from services.llm_factory import get_vector_store

# ── Variable chunk size parameters ──────────────────────────────
CHUNK_SIZE     = 1250   # Maximum characters per chunk
CHUNK_OVERLAP  = 200    # Overlap between consecutive chunks (16% of max)
MIN_CHUNK_SIZE = 500    # Minimum characters — chunks smaller than this get merged

# Separator hierarchy — splitter tries these in order, falling back to the next
# if chunks are still too large. Preserves semantic coherence.
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def clean_text(text: str) -> str:
    """Removes newlines, tabs, and collapses multiple spaces into one."""
    # Replace \n, \t, \r and multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    return cleaned.strip()


def _enforce_min_chunk_size(
    chunks: list[Document],
    min_chars: int = MIN_CHUNK_SIZE,
) -> list[Document]:
    """
    Merges chunks smaller than min_chars with their nearest neighbor.
    Guarantees every chunk in the output is at least min_chars long.

    Algorithm:
      1. Buffer starts as the first chunk
      2. For each subsequent chunk: if buffer < min_chars → merge into buffer
      3. After loop, if trailing buffer is still < min_chars → merge backward
    """
    if len(chunks) <= 1:
        return chunks

    result = []
    buffer = chunks[0]

    for chunk in chunks[1:]:
        if len(buffer.page_content) < min_chars:
            # Merge small chunk into buffer
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


def process_and_store_file(file: UploadFile, namespace: str = None):
    """
    Ingests an uploaded file into Pinecone under the given namespace.
    Supports PDF, DOCX, and TXT files.

    Uses variable chunk sizing (500–1250 chars) with semantic separators
    and min-size enforcement to guarantee embedding quality.
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
            doc.metadata["source"] = file.filename  # Store real filename, not temp path
        # -------------------------

        print("✅ Text cleaning complete.")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    print("✂️ Splitting into chunks (variable size 500–1250 chars)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = text_splitter.split_documents(raw_documents)

    # Enforce minimum chunk size — merge tiny chunks with neighbors
    chunks_before = len(chunks)
    chunks = _enforce_min_chunk_size(chunks, min_chars=MIN_CHUNK_SIZE)
    if len(chunks) != chunks_before:
        print(f"📐 Merged {chunks_before - len(chunks)} tiny chunks → {len(chunks)} final chunks")

    # Log chunk size stats
    sizes = [len(c.page_content) for c in chunks]
    if not sizes:
        return {"status": "warning", "message": "No chunks produced — document may be empty"}
    print(f"📊 Chunk stats: {len(chunks)} chunks | "
          f"min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)} chars")

    vector_store = get_vector_store(namespace)
    ns_label = namespace or "default"
    print(f"🚀 Uploading {len(chunks)} chunks to Pinecone (namespace={ns_label})...")
    
    # --- ADD THIS BATCHING LOGIC BACK ---
    batch_size = 50  # Safe batch size for local Ollama models
    total_batches = (len(chunks) + batch_size - 1) // batch_size 
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        current_batch = (i // batch_size) + 1
        print(f"   ⚙️ Embedding & Uploading batch {current_batch}/{total_batches}...")
        vector_store.add_documents(documents=batch)
    # ------------------------------------

    print(f"🎉 SUCCESS: {file.filename} is now clean and indexed!\n")
    return {
        "message": f"Successfully ingested {len(chunks)} clean chunks.",
        "namespace": ns_label,
    }