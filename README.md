# PDF OCR + Vector Search with SurrealDB

A CLI tool that extracts text from PDF files using OCR and indexes them in SurrealDB for semantic vector search.

## Features

- **PDF text extraction** using PyMuPDF with Tesseract OCR fallback for scanned documents
- **Semantic chunking** with configurable overlap for better search relevance
- **Vector embeddings** via sentence-transformers (all-MiniLM-L6-v2)
- **SurrealDB storage** with HNSW vector indexing and cosine similarity search
- **Rich CLI** with progress bars, formatted tables, and colored output

## Prerequisites

- Python 3.12+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (`sudo apt install tesseract-ocr`)
- [Poppler](https://poppler.freedesktop.org/) (`sudo apt install poppler-utils`)
- [SurrealDB](https://surrealdb.com/install) (`curl -sSf https://install.surrealdb.com | sh`)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Start SurrealDB

```bash
surreal start --user root --pass root --bind 0.0.0.0:8000 memory
```

### 2. Index PDFs

```bash
python3 -m app.main index pdfs/
```

Options:
- `--chunk-size` — Characters per chunk (default: 500)
- `--overlap` — Overlap between chunks (default: 50)

### 3. Search

```bash
python3 -m app.main search "your natural language query"
```

Options:
- `--top-k` — Number of results (default: 5)

### 4. Stats

```bash
python3 -m app.main stats
```
