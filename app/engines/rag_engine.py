# rag_engine.py - Orchestrates the RAG pipeline components to process queries

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import vectorstores.vector_store as vector_store

# Default prompt template for RAG
DEFAULT_PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}
"""

def create_llm_model(model_name="mistral", temperature=0.5):
    # Create an Ollama LLM model instance
    return OllamaLLM(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=temperature
    )

def process_query(query_text: str, top_k=5, model_name="mistral", temperature=0.5):
    # Process a query through the RAG pipeline
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
    return response