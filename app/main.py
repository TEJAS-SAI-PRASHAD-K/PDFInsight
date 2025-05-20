# main.py - Main entry point for the RAG application

from loaders.document_loader import load_pdf_document
from processors.document_processor import split_documents
from vectorstores.vector_store import add_documents
from engines.rag_engine import process_query


def assign_document_ids(document_chunks):
    # Assign unique IDs to document chunks
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in document_chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
            last_page_id = current_page_id
            
        chunk.metadata["id"] = f"{source}:{page}:{current_chunk_index}"
    
    return document_chunks

def main():
    # PDF filepath
    pdf_path = "/Users/tejassaiprashad/Desktop/blackjack.pdf"
    
    # Load and process documents
    print("Loading PDF documents...")
    documents = load_pdf_document(pdf_path)
    
    print("Splitting documents into chunks...")
    document_chunks = split_documents(documents)
    
    print("Assigning IDs to document chunks...")
    processed_chunks = assign_document_ids(document_chunks)
    
    print("Adding documents to vector store...")
    add_documents(processed_chunks)
    
    # Process a sample query
    sample_query = "What are the payouts and odds of the game?"
    print(f"\nProcessing query: '{sample_query}'")
    response = process_query(sample_query)
    
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()