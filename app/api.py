# api.py - FastAPI REST layer over the RAG pipeline
# Run (from the app/ directory): uvicorn api:app --reload

import shutil
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from main import ingest_pdf
from engines.rag_engine import process_query

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="PDFInsight", description="Ask questions about your PDFs using local RAG")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    filename: str | None = None  # restrict the search to one uploaded PDF


class Source(BaseModel):
    id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


# Handlers are plain `def` (not `async def`) because loading, embedding and the LLM call are
# blocking; FastAPI runs sync handlers in a threadpool so they don't block the event loop.
@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Keep only the base name to prevent path traversal (e.g. "../../etc/passwd.pdf")
    saved_path = UPLOAD_DIR / Path(file.filename).name
    with saved_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    chunks_added = ingest_pdf(str(saved_path))
    return {"filename": saved_path.name, "chunks_added": chunks_added}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        source = str(UPLOAD_DIR / Path(request.filename).name) if request.filename else None
        return process_query(request.question, top_k=request.top_k, source=source)
    except Exception as e:
        # Most commonly: Ollama isn't running or the model hasn't been pulled
        raise HTTPException(status_code=503, detail=f"LLM backend error: {e}")
