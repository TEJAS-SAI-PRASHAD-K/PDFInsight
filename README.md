# 📄 PDFInsight

An AI-powered local-first system for extracting insights from PDF documents using Retrieval-Augmented Generation (RAG). This app loads documents, processes them into chunks, generates embeddings using sentence-transformers, and performs semantic search via a vector store (e.g., FAISS). A FastAPI backend can be used for interactive queries and integration.

---

## 🧠 Features

* 📂 Load and parse PDF or text-based documents
* 🧼 Preprocess and chunk documents for optimal embedding
* 🔎 Semantic search using vector similarity (e.g., FAISS)
* 🧠 Sentence-transformer-based embedding generation
* 🔄 Retrieval-Augmented Generation engine (RAG)
* 🚀 FastAPI backend for RESTful document insight queries
* 🛠️ Modular, clean, and extensible codebase

---

## 🗂 Project Structure

```
pdfinsight/
├── app/
│   ├── main.py                   # Application entry point (e.g., FastAPI setup)
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
│   │   └── vector_store.py       # Store and query embedding vectors using FAISS or similar
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

* Python 3.9+
* PdfPlumber (for PDF parsing)
* langchain
* sentence-transformers
* Chroma (for vector search)
* FastAPI + Uvicorn (for API layer)

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
