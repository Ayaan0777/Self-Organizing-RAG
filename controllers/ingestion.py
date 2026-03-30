import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    text = re.sub(r'[\t\r]+', ' ', text)
    text = re.sub(r' {3,}', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    cleaned = '\n'.join(lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

def process_and_store_file(file: UploadFile, strategy: str = "semantic", namespace: str = "default"):
    print(f"\n--- 📥 STARTING INGESTION: {file.filename} into namespace: {namespace} [{strategy.upper()} STRATEGY] ---")
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
        if strategy not in ["markdown", "beautifulsoup"]:
            print("🧹 Cleaning text (removing newlines and tabs)...")
            for doc in raw_documents:
                doc.page_content = clean_text(doc.page_content)
            print("✅ Text cleaning complete.")
        else:
            print(f"⏩ Skipping strict text cleaning to preserve {strategy} structure tags.")
        # -------------------------

        # --- CHUNKING LOGIC ---
        print(f"Splitting into chunks using '{strategy}' strategy...")
        MAX_CHUNK_SIZE = 500
        overlap = 50
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

        elif strategy == "markdown":
            print("Using MARKDOWNHeaderTextSplitter (pymupdf4llm)...")
            from langchain_text_splitters import MarkdownHeaderTextSplitter
            
            md_text = ""
            if file_extension == ".pdf":
                import pymupdf4llm
                md_text = pymupdf4llm.to_markdown(temp_file_path)
            else:
                md_text = "\n\n".join([doc.page_content for doc in raw_documents])

            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_docs = markdown_splitter.split_text(md_text)

            fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=overlap)
            chunks = fallback_splitter.split_documents(md_docs)
            
        elif strategy == "beautifulsoup":
            print("Using BeautifulSoup (HTMLHeaderTextSplitter)...")
            from langchain_text_splitters import HTMLHeaderTextSplitter
            
            html_text = ""
            if file_extension == ".pdf":
                import fitz
                doc = fitz.open(temp_file_path)
                for page in doc:
                    html_text += page.get_text("html")
                doc.close()
            else:
                raw_t = "\n\n".join([doc.page_content for doc in raw_documents])
                html_text = f"<html><body><p>{raw_t}</p></body></html>"

            headers_to_split_on = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
            ]
            html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            html_docs = html_splitter.split_text(html_text)

            fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=overlap)
            chunks = fallback_splitter.split_documents(html_docs)
            
        else:
            # Default: Pure semantic with size limit fallback
            print("Using SEMANTIC strategy...")
            
            # 1. Pre-split into structural blocks to safely avoid Ollama 400 context length errors
            # SemanticChunker can easily breach 256 tokens. Set pre-split to 400, and buffer to 0.
            pre_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
            safe_docs = pre_splitter.split_documents(raw_documents)

            embeddings = get_embeddings()
            semantic_splitter = SemanticChunker(
                embeddings=embeddings,
                buffer_size=0,
                breakpoint_threshold_type="percentile"
            )
            semantic_chunks = semantic_splitter.split_documents(safe_docs)
            
            fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=MAX_CHUNK_SIZE,
                chunk_overlap=overlap
            )
            for chunk in semantic_chunks:
                if len(chunk.page_content) <= MAX_CHUNK_SIZE:
                    chunks.append(chunk)
                else:
                    chunks.extend(fallback_splitter.split_documents([chunk]))

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

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

    print(f"🚀 Uploading {len(chunks)} {strategy} chunks to Pinecone namespace '{namespace}'...")
    vector_store = get_vector_store(namespace=namespace)
    vector_store.add_documents(documents=chunks)

    print(f"🎉 SUCCESS: {file.filename} is now indexed with {strategy} chunking!\n")
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

if __name__ == "__main__":
    import sys
    
    class MockUploadFile:
        def __init__(self, filename, filepath):
            self.filename = filename
            self.file = open(filepath, 'rb')

    def run_all_ingestions():
        doc_path = r"C:\Users\hegde\Downloads\contexts.docx"
        if not os.path.exists(doc_path):
            print(f"Error: Document {doc_path} not found.")
            sys.exit(1)
            
        strategies = ["semantic", "recursive", "markdown", "beautifulsoup"]
        
        for strat in strategies:
            print(f"\n--- Ingesting {doc_path} with '{strat}' strategy into namespace '{strat}' ---")
            file_mock = MockUploadFile(os.path.basename(doc_path), doc_path)
            result = process_and_store_file(file_mock, strategy=strat, namespace=strat)
            print("Ingestion result:", result)
            
    run_all_ingestions()