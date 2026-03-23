import os
import sys

from controllers.ingestion import clear_vector_store, process_and_store_file

class MockUploadFile:
    def __init__(self, filename, filepath):
        self.filename = filename
        self.file = open(filepath, 'rb')

def main():
    print("Clearing Pinecone Database...")
    clear_result = clear_vector_store()
    print("Clear result:", clear_result)
    
    doc_path = r"C:\Users\hegde\Downloads\contexts.docx"
    if not os.path.exists(doc_path):
        print(f"Error: Document {doc_path} not found.")
        sys.exit(1)
        
    print(f"Ingesting {doc_path} with 'hybrid' strategy...")
    file_mock = MockUploadFile("contexts.docx", doc_path)
    result = process_and_store_file(file_mock, strategy="hybrid")
    print("Ingestion result:", result)

if __name__ == "__main__":
    main()
