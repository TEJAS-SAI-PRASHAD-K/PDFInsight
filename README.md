# 📄 PDFInsight

An AI-powered local-first system for extracting insights from PDF documents using Retrieval-Augmented Generation (RAG). This app loads documents, processes them into chunks, generates embeddings using sentence-transformers, and performs semantic search via a Chroma vector store, and answers questions with a local LLM served by Ollama. A FastAPI backend exposes it as a REST API.

---

## 🧠 Features

* 📂 Load and parse PDF or text-based documents
* 🧼 Preprocess and chunk documents for optimal embedding
* 🔎 Semantic search using vector similarity (Chroma)
* 🧠 Sentence-transformer-based embedding generation
* 🔄 Retrieval-Augmented Generation engine (RAG)
* 🚀 FastAPI backend for RESTful document insight queries
* 🛠️ Modular, clean, and extensible codebase

---

## 🗂 Project Structure

```
pdfinsight/
├── app/
│   ├── main.py                   # CLI entry point + ingest pipeline
│   ├── api.py                    # FastAPI REST API
│   │
│   ├── loaders/
│   │   └── document_loader.py    # Load PDF/text files into memory
│   │
│   ├── processors/
│   │   └── document_processor.py # Clean and split text into chunks
│   │
│   ├── embeddings/
│   │   └── embedding_service.py  # Generate vector embeddings for text chunks
│   │
│   ├── vectorstores/
│   │   └── vector_store.py       # Store and query embedding vectors in Chroma
│   │
│   └── engines/
│       └── rag_engine.py         # Perform RAG (retrieve + generate)
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation (this file)
└── .gitignore                    # Git ignored files
```

---

## 🔧 Tech Stack

* Python 3.10+ (tested on 3.12)
* PdfPlumber (for PDF parsing)
* langchain
* sentence-transformers
* Chroma (for vector search)
* Ollama (local LLM, default `llama3.1:8b`)
* FastAPI + Uvicorn (for API layer)

---

## ▶️ Running

```bash
# 1. Install Ollama (https://ollama.com), then pull a model and start it
ollama pull llama3.1:8b
ollama serve                    # or just open the Ollama app

# 2. Set up Python
python3.12 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt

# 3a. CLI: ingest a PDF and ask a question
cd app
python main.py /path/to/file.pdf "What is this document about?"

# 3b. REST API (interactive docs at http://localhost:8000/docs)
cd app
uvicorn api:app --reload

# Start a session: upload one or more PDFs -> returns a session_id (each session gets its own Chroma DB)
curl -F "files=@/path/to/a.pdf" -F "files=@/path/to/b.pdf" localhost:8000/start

# Ask questions within the session (optionally add "filename": "a.pdf" to search one PDF only)
curl -H "content-type: application/json" -d '{"session_id": "<id>", "question": "What is this about?"}' localhost:8000/query

# Stop the session: deletes its PDFs and its Chroma DB
curl -H "content-type: application/json" -d '{"session_id": "<id>"}' localhost:8000/stop
```
/Users/tejassaiprashad/Desktop/IMP

Use a different model with `OLLAMA_MODEL=mistral` (after `ollama pull mistral`), or a remote Ollama with `OLLAMA_BASE_URL`.

---

## 🪪 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

See `CONTRIBUTING.md` for guidelines.

---

## 📬 Contact

For questions, suggestions, or feedback, open an issue or contact @TEJAS-SAI-PRASHAD-K on GitHub.

---
