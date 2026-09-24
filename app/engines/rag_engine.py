# rag_engine.py - Orchestrates the RAG pipeline components to process queries

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import vectorstores.vector_store as vector_store

# Configurable via environment variables; defaults assume a local Ollama server
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Default prompt template for RAG
DEFAULT_PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}
"""

def create_llm_model(model_name=OLLAMA_MODEL, temperature=0.5):
    # Create an Ollama LLM model instance
    return OllamaLLM(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature
    )

def process_query(query_text: str, top_k=5, model_name=OLLAMA_MODEL, temperature=0.5):
    # Process a query through the RAG pipeline; returns the answer and the chunks it was based on
    # Retrieve relevant documents
    results = vector_store.similarity_search(query_text, top_k=top_k)

    # Format context from retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Create prompt with context
    prompt_template = ChatPromptTemplate.from_template(DEFAULT_PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate response
    model = create_llm_model(model_name, temperature)
    response = model.invoke(prompt)
    sources = [{"id": doc.metadata.get("id"), "score": score} for doc, score in results]
    return {"answer": response, "sources": sources}
