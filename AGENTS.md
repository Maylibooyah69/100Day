# AGENTS

## Cursor Cloud specific instructions

This is a Python CLI application that performs OCR on PDF files, indexes them in SurrealDB for semantic vector search, and provides topic modeling with interactive visualizations.

### Architecture
- **`app/ocr.py`** — PDF text extraction (PyMuPDF) with Tesseract OCR fallback for scanned pages
- **`app/chunker.py`** — Overlapping text chunking with sentence boundary detection
- **`app/embeddings.py`** — Embedding generation using `sentence-transformers` (all-MiniLM-L6-v2, 384 dimensions)
- **`app/db.py`** — SurrealDB storage with HNSW vector index and cosine similarity search
- **`app/topics.py`** — Topic modeling using BERTopic with UMAP + HDBSCAN + MMR
- **`app/visualize.py`** — 7 interactive visualization types (Plotly HTML + matplotlib word clouds)
- **`app/main.py`** — CLI entry point (Click + Rich) with `index`, `search`, `stats`, and `topics` commands

### System dependencies
- **Python 3.12+** (system Python)
- **Tesseract OCR** (`tesseract-ocr` package)
- **Poppler** (`poppler-utils` for `pdf2image`)
- **SurrealDB** v3.x (installed at `~/.surrealdb/surreal`)
- **python3-dev + build-essential** (for building hdbscan C extension)

### Running the app
1. Start SurrealDB: `surreal start --user root --pass root --bind 0.0.0.0:8000 memory`
2. Index PDFs: `python3 -m app.main index pdfs/`
3. Search: `python3 -m app.main search "your query"`
4. Stats: `python3 -m app.main stats`
5. Topic discovery + visualizations: `python3 -m app.main topics discover pdfs/ --min-topic-size 2`

### Gotchas
- **SurrealDB v3 KNN operator**: The `<|k|>` KNN operator has issues in SurrealDB v3.0.x. The app uses brute-force `vector::similarity::cosine()` with `ORDER BY` + `LIMIT` instead. The HNSW index is still defined for future compatibility.
- **SurrealDB v3 index syntax**: Use `HNSW` (not `MTREE`) for vector indexes in SurrealDB v3.
- **SurrealDB Python SDK**: No `.connect()` method — the `Surreal('ws://...')` constructor connects immediately. Query results are returned directly (not wrapped in `{"result": [...]}`).
- **PATH**: SurrealDB binary is at `~/.surrealdb/surreal`; pip scripts at `~/.local/bin/`. Both need to be on PATH.
- **Embedding model**: First run downloads the model from HuggingFace (~80MB). Subsequent runs use the cached version.
- **No deduplication**: Re-indexing the same PDFs creates duplicate entries. Clear the DB (restart with `memory` backend) before re-indexing if needed.
- **BERTopic min_topic_size**: With small document sets, use `--min-topic-size 2` to allow smaller clusters. Default HDBSCAN min_cluster_size=3 may merge too aggressively on small corpora.
- **Visualizations**: All Plotly HTML files are interactive. Open in Chrome for best experience. Word clouds are saved as PNG images.
