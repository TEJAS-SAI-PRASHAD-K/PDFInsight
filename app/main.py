# main.py - Command-line entry point for the RAG application
# Usage (from the app/ directory): python main.py <path/to/file.pdf> ["your question"]

import os
import sys
import httpx
from loaders.document_loader import load_pdf_document
from processors.document_processor import split_documents, assign_document_ids
from vectorstores.vector_store import CHROMA_PERSIST_DIR, add_documents
from engines.rag_engine import OLLAMA_BASE_URL, process_query


def ingest_pdf(pdf_path, persist_directory=CHROMA_PERSIST_DIR):
    # Load -> chunk -> ID -> embed + store. Returns the number of new chunks added.
    print("Loading PDF documents...")
    documents = load_pdf_document(pdf_path)

    print("Splitting documents into chunks...")
    document_chunks = split_documents(documents)

    print("Assigning IDs to document chunks...")
    processed_chunks = assign_document_ids(document_chunks)

    print("Adding documents to vector store...")
    return add_documents(processed_chunks, persist_directory=persist_directory)

def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py <path/to/file.pdf> ["your question"]')
        sys.exit(1)

    # Absolute path so the stored "source" metadata is the same regardless of where you run from
    pdf_path = os.path.abspath(sys.argv[1])
    query = sys.argv[2] if len(sys.argv) > 2 else "Summarize the main points of this document."

    if not os.path.isfile(pdf_path):
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)

    ingest_pdf(pdf_path)

    print(f"\nProcessing query: '{query}'")
    try:
        result = process_query(query, source=pdf_path)
    except httpx.ConnectError:
        print(f"Error: can't reach Ollama at {OLLAMA_BASE_URL}. Open the Ollama app or run `ollama serve`, then retry.")
        sys.exit(1)

    print("\nResponse:")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"  {source['id']} (distance: {source['score']:.3f})")

if __name__ == "__main__":
    main()