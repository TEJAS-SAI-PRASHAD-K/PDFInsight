# document_processor.py - Processes documents by splitting them into manageable chunks

from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document

def split_documents(documents: list[Document]):
    # Split documents into smaller chunks for better processing
    splitter = MarkdownTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    document_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            document_chunks.append(Document(page_content=chunk, metadata=dict(doc.metadata)))

    return document_chunks

def assign_document_ids(document_chunks):
    # Assign deterministic IDs ("source:page:chunk_index") so re-ingesting a file doesn't create duplicates
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
