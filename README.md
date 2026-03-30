Step 1 - Install dependencies in a virtual environment
pip install fastapi uvicorn pydantic-settings langchain langchain-pinecone pinecone-client langchain-ollama langchain-google-genai ragas datasets

Step 2 - Create an account in pinecone and an index with the dimensions of the embedding model used 
-> Enter name of index
-> Select Custom settings
-> Enter dimensions

Step 3 - Create .env with API keys

Step 4 - Upload the files on /ingest endpoint and query on /query endpoint. To test search quality locally, you can use the run_evaluation.py script.
