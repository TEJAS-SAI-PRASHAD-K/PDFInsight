# document_loader.py - Loads PDF documents using PDFPlumber

from langchain_community.document_loaders import PDFPlumberLoader

def load_pdf_document(pdf_path):
    # Load a PDF file and return the document objects
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    return documents