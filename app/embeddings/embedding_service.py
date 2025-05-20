# embedding_service.py - Creates and configures the embedding model for vector embeddings

from langchain_huggingface import HuggingFaceEmbeddings

def create_embedding_model():
    # Create and return a HuggingFace embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings