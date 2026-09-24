# main.py - Command-line entry point for the RAG application
# Usage (from the app/ directory): python main.py <path/to/file.pdf> ["your question"]

import sys
from loaders.document_loader import load_pdf_document
from processors.document_processor import split_documents, assign_document_ids
from vectorstores.vector_store import add_documents
from engines.rag_engine import process_query


def ingest_pdf(pdf_path):
    # Load -> chunk -> ID -> embed + store. Returns the number of new chunks added.
    print("Loading PDF documents...")
    documents = load_pdf_document(pdf_path)

    print("Splitting documents into chunks...")
    document_chunks = split_documents(documents)

    print("Assigning IDs to document chunks...")
    processed_chunks = assign_document_ids(document_chunks)

    print("Adding documents to vector store...")
    return add_documents(processed_chunks)

def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py <path/to/file.pdf> ["your question"]')
        sys.exit(1)

    pdf_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "Summarize the main points of this document."

    ingest_pdf(pdf_path)

    print(f"\nProcessing query: '{query}'")
    result = process_query(query)

    print("\nResponse:")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"  {source['id']} (distance: {source['score']:.3f})")

if __name__ == "__main__":
    main()
