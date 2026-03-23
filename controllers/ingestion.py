import os
import re
import tempfile
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.llm_factory import get_vector_store, get_embeddings
from pinecone import Pinecone
from config import settings

def clean_text(text: str) -> str:
    """Removes newlines, tabs, and collapses multiple spaces into one."""
    # Replace \n, \t, \r and multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    return cleaned.strip()

def process_and_store_file(file: UploadFile, strategy: str = "semantic"):
    print(f"\n--- STARTING INGESTION: {file.filename} [{strategy.upper()} STRATEGY] ---")
    file_extension = os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name

    try:
        print(f"Extracting text from {file_extension}...")
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
        print("Cleaning text (removing newlines and tabs)...")
        for doc in raw_documents:
            doc.page_content = clean_text(doc.page_content)
        # -------------------------

        print("Text cleaning complete.")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # --- CHUNKING LOGIC ---
    print(f"Splitting into chunks using '{strategy}' strategy...")
    MAX_CHUNK_SIZE = 1000
    overlap = 100
    chunks = []

    if strategy == "recursive":
        print("Using basic RecursiveCharacterTextSplitter...")
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=overlap
        )
        chunks = recursive_splitter.split_documents(raw_documents)

    elif strategy == "hybrid":
        print("Using HYBRID strategy (structural recursive + semantic)...")
        # 1. First split by large structural blocks (paragraphs/pages)
        structural_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200
        )
        structural_chunks = structural_splitter.split_documents(raw_documents)
        
        # 2. Then apply SemanticChunking inside those blocks
        embeddings = get_embeddings()
        semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"
        )
        print(f"Applying semantic chunking to {len(structural_chunks)} structural blocks...")
        semantic_chunks = semantic_splitter.split_documents(structural_chunks)

        # 3. Final safety fall-back exactly like purely semantic
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=overlap
        )
        for chunk in semantic_chunks:
            if len(chunk.page_content) <= MAX_CHUNK_SIZE:
                chunks.append(chunk)
            else:
                chunks.extend(fallback_splitter.split_documents([chunk]))

    else:
        # Default: Pure semantic with size limit fallback
        print("Using SEMANTIC strategy...")
        embeddings = get_embeddings()
        semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"
        )
        semantic_chunks = semantic_splitter.split_documents(raw_documents)
        
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=overlap
        )
        for chunk in semantic_chunks:
            if len(chunk.page_content) <= MAX_CHUNK_SIZE:
                chunks.append(chunk)
            else:
                chunks.extend(fallback_splitter.split_documents([chunk]))

    # Calculate metrics
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    min_size = min(chunk_sizes) if chunk_sizes else 0
    max_size = max(chunk_sizes) if chunk_sizes else 0

    print(f"Created {len(chunks)} {strategy} chunks")
    print(f"  Average size: {avg_size:.0f} chars")
    print(f"  Min size: {min_size} chars")
    print(f"  Max size: {max_size} chars")
    # -------------------------

    print(f"Uploading {len(chunks)} {strategy} chunks to Pinecone...")
    vector_store = get_vector_store()
    vector_store.add_documents(documents=chunks)

    print(f"SUCCESS: {file.filename} is now indexed with {strategy} chunking!\n")
    return {
        "message": f"Successfully ingested {len(chunks)} chunks using {strategy} strategy.",
        "strategy": strategy,
        "chunks_created": len(chunks),
        "avg_chunk_size": round(avg_size, 2),
        "min_chunk_size": min_size,
        "max_chunk_size": max_size
    }


def clear_vector_store():
    """Deletes all vectors from the Pinecone index."""
    print("\n--- CLEARING PINECONE INDEX ---")
    try:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)

        # Delete all vectors by deleting all IDs (Pinecone specific)
        index.delete(delete_all=True)

        print("SUCCESS: All vectors cleared from Pinecone!\n")
        return {"message": "All vectors have been cleared from the database."}
    except Exception as e:
        print(f"ERROR clearing Pinecone: {str(e)}")
        return {"error": f"Failed to clear database: {str(e)}"}