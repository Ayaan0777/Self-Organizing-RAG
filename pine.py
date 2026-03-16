from pinecone import Pinecone
pc = Pinecone(api_key="pcsk_7NB7zH_SpE3gUarYzzrtVkH4GQ8kVKDp5fwkbUwgpNpmpZfGGPpKMUBPkGnPobfjsJZJiR")

assistant = pc.assistant.create_assistant(
    assistant_name="example-assistant", 
    instructions="Answer in polite, short sentences. Use American English spelling and vocabulary.", 
    timeout=30 # Wait 30 seconds for assistant operation to complete.
)