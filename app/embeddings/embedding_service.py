# embedding_service.py - Creates and configures the embedding model for vector embeddings

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

@lru_cache(maxsize=1)
def create_embedding_model():
    # Create the HuggingFace embeddings model once and reuse it (loading it is slow)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
