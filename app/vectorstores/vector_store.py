# vector_store.py - Manages operations on the vector database, including adding and retrieving documents

from embeddings.embedding_service import create_embedding_model
from langchain_chroma import Chroma
from langchain.schema.document import Document

CHROMA_PERSIST_DIR = "/Users/tejassaiprashad/Documents/my_workspace/PDFInsight/chroma_db"

def get_db_connection(persist_directory=CHROMA_PERSIST_DIR):
    # Get connection to the Chroma vector store
    embedding_function = create_embedding_model()
    return Chroma(
        persist_directory=persist_directory, 
        embedding_function=embedding_function
    )
        
def add_documents(documents: list[Document], persist_directory=CHROMA_PERSIST_DIR):
    # Add documents to the vector store, avoiding duplicates
    db = get_db_connection(persist_directory)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Filter out already existing documents
    new_documents = [doc for doc in documents if doc.metadata["id"] not in existing_ids]

    if not new_documents:
        print("No new documents to add.")
        return

    new_document_ids = [doc.metadata["id"] for doc in new_documents]
    db.add_documents(new_documents, ids=new_document_ids)
    print(f"Added {len(new_documents)} new documents.")
        
def similarity_search(query_text: str, top_k=5, persist_directory=CHROMA_PERSIST_DIR):
    # Perform similarity search on the vector store
    db = get_db_connection(persist_directory)
    results = db.similarity_search_with_score(query_text, k=top_k)
    return results