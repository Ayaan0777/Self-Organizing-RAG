import os
import sys

from controllers.ingestion import clear_vector_store, process_and_store_file

class MockUploadFile:
    def __init__(self, filename, filepath):
        self.filename = filename
        self.file = open(filepath, 'rb')

def main():
    # Update this to point to your actual PDF or DOCX file
    doc_path = r"C:\Users\hegde\Downloads\contexts.docx"
    
    if not os.path.exists(doc_path):
        print(f"Error: Document {doc_path} not found. Please update 'doc_path' in this script!")
        sys.exit(1)
        
    print(f"--- Ingesting {doc_path} with 'markdown' strategy ---")
    file_mock_1 = MockUploadFile("contexts.pdf", doc_path)
    # Using namespace="markdown" and strategy="markdown"
    result_1 = process_and_store_file(file_mock_1, strategy="markdown", namespace="markdown")
    print("Markdown Ingestion result:", result_1)

    print(f"\n--- Ingesting {doc_path} with 'beautifulsoup' strategy ---")
    file_mock_2 = MockUploadFile("contexts.pdf", doc_path)
    # Using namespace="beautifulsoup" and strategy="beautifulsoup"
    result_2 = process_and_store_file(file_mock_2, strategy="beautifulsoup", namespace="beautifulsoup")
    print("BeautifulSoup Ingestion result:", result_2)
    
    print("\n✅ Both strategies have been ingested. You can now run `python run_evaluation.py` to evaluate them!")

if __name__ == "__main__":
    main()
