import os
import re
import tempfile
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.llm_factory import get_vector_store

def clean_text(text: str) -> str:
    text = re.sub(r'[\t\r]+', ' ', text)
    text = re.sub(r' {3,}', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    cleaned = '\n'.join(lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()
def process_and_store_file(file: UploadFile, namespace: str = "default"):
    print(f"\n--- 📥 STARTING INGESTION: {file.filename} into namespace: {namespace} ---")
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
        
        print("✅ Text cleaning complete.")
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    print("✂️ Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(raw_documents)
    
    print(f"🚀 Uploading {len(chunks)} cleaned chunks to Pinecone namespace '{namespace}' in batches...")
    vector_store = get_vector_store(namespace=namespace)
    
    # --- NEW BATCHING LOGIC ---
    batch_size = 100
    # Calculates the exact number of batches needed
    total_batches = (len(chunks) + batch_size - 1) // batch_size 
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        current_batch_num = (i // batch_size) + 1
        
        print(f"   ⚙️ Embedding and Uploading batch {current_batch_num}/{total_batches} (Chunks {i} to {i + len(batch)})...")
        # Embeds and uploads ONLY 100 at a time
        vector_store.add_documents(documents=batch)
    # --------------------------
    
    print(f"🎉 SUCCESS: {file.filename} is now clean and indexed!\n")
    return {"message": f"Successfully ingested {len(chunks)} clean chunks into namespace '{namespace}'."}