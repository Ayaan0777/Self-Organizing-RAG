import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# -----------------------------
# Load document
# -----------------------------
print("Loading document...")

loader = Docx2txtLoader("documents/sample.docx")
documents = loader.load()


# -----------------------------
# Chunk documents
# -----------------------------
print("Splitting document into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)

chunks = text_splitter.split_documents(documents)

print("Total chunks created:", len(chunks))


# -----------------------------
# Load embedding model
# -----------------------------
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# Connect to Pinecone
# -----------------------------
print("Connecting to Pinecone...")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "rag-index"


# -----------------------------
# Upload embeddings
# -----------------------------
print("Uploading embeddings to Pinecone...")

vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

print("✅ Embeddings uploaded successfully!")
print("Total vectors stored:", len(chunks))