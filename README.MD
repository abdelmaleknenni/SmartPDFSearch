# 🔎 Smart Document Search Engine

A project that uses AI to search inside PDF documents using **semantic understanding**, not just keyword matching.

## ✅ Features:
- Reads and processes multiple PDF files.
- Breaks content into paragraphs for precision.
- Allows you to ask **natural language questions**.
- Returns the most semantically relevant results.

## 🧠 Technologies Used:
- Python
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) for reading PDFs
- [SentenceTransformers](https://www.sbert.net/) (MiniLM)
- [FAISS](https://github.com/facebookresearch/faiss) for fast similarity search

## 📦 Requirements:
```bash
pip install sentence-transformers faiss-cpu pymupdf
