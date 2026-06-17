import os
from dotenv import load_dotenv

# Pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# LangChain utilities
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# -----------------------------
# Connect to Pinecone
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "rag-index"


# -----------------------------
# Load embedding model
# (same one used during ingestion)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# Connect to existing Pinecone index
# (NO uploading here)
# -----------------------------
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

print("✅ Connected to Pinecone index")


# -----------------------------
# Initialize Local LLM (Ollama)
# -----------------------------
llm = ChatOllama(
    model="llama3.1"
)

print("✅ Ollama LLM loaded")


# -----------------------------
# Create Retriever
# -----------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


# -----------------------------
# Interactive Q&A Loop
# -----------------------------
print("\n===== RAG Question Answering System =====")

while True:

    query = input("\nAsk a question (type 'quit' to exit): ")

    if query.lower() == "quit":
        print("\nExiting RAG system. Goodbye!")
        break

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    if not docs:
        print("\nAnswer:")
        print("I cannot answer this question as the content is unavailable in the provided documents.")
        continue

    print("\nRetrieved context:\n")

    for doc in docs:
        print(doc.page_content[:300])
        print("\n---\n")

    # Build context
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You must answer ONLY using the provided context.

If the answer is not contained in the context,
reply with:
"I cannot answer this question as the content is unavailable in the provided documents."

Context:
{context}

Question:
{query}
"""

    # Generate answer
    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)