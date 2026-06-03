import os
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from services.llm_factory import get_vector_store


def ingest_dataset(input_path: str, namespace: str = None):
    """
    Reads a file, chunks using RecursiveCharacterTextSplitter (500 chars, 50 overlap),
    and stores directly in Pinecone under the given namespace.
    """
    print(f"📥 Loading dataset from: {input_path}")

    if not os.path.exists(input_path):
        print(f"❌ Error: File '{input_path}' not found!")
        return

    raw_documents = []
    file_ext = os.path.splitext(input_path)[1].lower()

    # --- Load Data Based on File Type ---
    if file_ext == '.json':
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for idx, item in enumerate(data):
                ans_list = item.get("ans", [])
                for ans in ans_list:
                    if isinstance(ans, str) and len(ans) > 5:
                        raw_documents.append(Document(page_content=ans, metadata={"source": input_path, "index": idx}))
    elif file_ext == '.pdf':
        loader = PyPDFLoader(input_path)
        raw_documents = loader.load()
    elif file_ext == '.txt':
        loader = TextLoader(input_path, encoding='utf-8')
        raw_documents = loader.load()
    elif file_ext == '.docx':
        loader = Docx2txtLoader(input_path)
        raw_documents = loader.load()
    else:
        print(f"❌ Unsupported file type: {file_ext}")
        return

    if not raw_documents:
        print("⚠️ No text content could be extracted from the dataset.")
        return

    print(f"🧹 Extracted {len(raw_documents)} raw document elements. Cleaning whitespace...")

    # Clean whitespace
    for doc in raw_documents:
        doc.page_content = " ".join(doc.page_content.split())

    # --- Recursive Character Text Splitting ---
    print("✂️ Splitting with RecursiveCharacterTextSplitter (chunk_size=1000, overlap=100)...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_documents)

    # Show stats
    sizes = [len(c.page_content) for c in chunks]
    print(f"   ✅ Created {len(chunks)} chunks")
    print(f"   📊 Chunk sizes — min: {min(sizes)}, max: {max(sizes)}, avg: {sum(sizes)//len(sizes)}")

    # Store in Pinecone
    ns_label = namespace or "default"
    print(f"🚀 Uploading {len(chunks)} chunks to Pinecone (namespace={ns_label})...")
    vector_store = get_vector_store(namespace)
    vector_store.add_documents(documents=chunks)

    print(f"🎉 SUCCESS: {os.path.basename(input_path)} is now indexed and ready for querying!")


if __name__ == "__main__":
    dataset_path = r"C:\Users\Kishore\OneDrive\Desktop\HPE-RAG\contexts.docx"

    if not dataset_path:
        dataset_path = input("Please enter the absolute path to your dataset file: ").strip()

    if dataset_path:
        # Ingest into the mxbai-embed-large namespace in rag-index-1024
        ingest_dataset(dataset_path, namespace="mxbai-embed-large")
