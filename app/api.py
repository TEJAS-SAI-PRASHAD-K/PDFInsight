# api.py - FastAPI REST layer over the RAG pipeline
# Run (from the app/ directory): uvicorn api:app --reload
#
# Session flow:
#   POST /start  (upload PDFs)   -> creates sessions/<session_id>/ with its own Chroma DB, returns session_id
#   POST /query  (session_id)    -> ask questions against that session's PDFs only
#   POST /stop   (session_id)    -> deletes the session's PDFs and Chroma DB

import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from main import ingest_pdf
from engines.rag_engine import process_query
from vectorstores.vector_store import delete_db

SESSIONS_DIR = Path(__file__).resolve().parents[1] / "sessions"

app = FastAPI(title="PDFInsight", description="Ask questions about your PDFs using local RAG")


class QueryRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    filename: str | None = None  # restrict the search to one PDF in the session


class StopRequest(BaseModel):
    session_id: str


class Source(BaseModel):
    id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


def get_session_dir(session_id: str) -> Path:
    # Only accept real UUIDs, so a session_id like "../../" can never point outside SESSIONS_DIR
    try:
        session_dir = SESSIONS_DIR / str(uuid.UUID(session_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")
    return session_dir


def delete_session(session_dir: Path):
    delete_db(str(session_dir / "chroma_db"))
    shutil.rmtree(session_dir, ignore_errors=True)


@app.get("/health")
def health():
    return {"status": "ok"}


# Handlers are plain `def` (not `async def`) because loading, embedding and the LLM call are
# blocking; FastAPI runs sync handlers in a threadpool so they don't block the event loop.
@app.post("/start")
def start_session(files: list[UploadFile] = File(...)):
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {file.filename}")

    session_id = str(uuid.uuid4())
    session_dir = SESSIONS_DIR / session_id
    upload_dir = session_dir / "uploads"
    upload_dir.mkdir(parents=True)

    try:
        chunks_added = 0
        for file in files:
            # Keep only the base name to prevent path traversal (e.g. "../../etc/passwd.pdf")
            saved_path = upload_dir / Path(file.filename).name
            with saved_path.open("wb") as out:
                shutil.copyfileobj(file.file, out)
            chunks_added += ingest_pdf(str(saved_path), persist_directory=str(session_dir / "chroma_db"))
    except Exception:
        # Don't leave a half-built session behind
        delete_session(session_dir)
        raise

    return {"session_id": session_id, "files": [Path(f.filename).name for f in files], "chunks_added": chunks_added}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    session_dir = get_session_dir(request.session_id)
    source = str(session_dir / "uploads" / Path(request.filename).name) if request.filename else None
    try:
        return process_query(request.question, top_k=request.top_k, source=source,
                             persist_directory=str(session_dir / "chroma_db"))
    except Exception as e:
        # Most commonly: Ollama isn't running or the model hasn't been pulled
        raise HTTPException(status_code=503, detail=f"LLM backend error: {e}")


@app.post("/stop")
def stop_session(request: StopRequest):
    session_dir = get_session_dir(request.session_id)
    delete_session(session_dir)
    return {"session_id": request.session_id, "status": "stopped"}
