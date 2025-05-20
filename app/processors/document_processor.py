# document_processor.py - Processes documents by splitting them into manageable chunks

from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema.document import Document 

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
            document_chunks.append(Document(page_content=chunk, metadata=doc.metadata)) 

    return document_chunks