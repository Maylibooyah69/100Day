# PDF OCR + Vector Search + Topic Modeling

A CLI tool that extracts text from PDF files using OCR, indexes them in SurrealDB for semantic vector search, and performs topic modeling with interactive visualizations.

## Features

- **PDF text extraction** using PyMuPDF with Tesseract OCR fallback for scanned documents
- **Semantic chunking** with configurable overlap for better search relevance
- **Vector embeddings** via sentence-transformers (all-MiniLM-L6-v2)
- **SurrealDB storage** with HNSW vector indexing and cosine similarity search
- **BERTopic modeling** with UMAP dimensionality reduction and HDBSCAN clustering
- **7 visualization types**: topic overview, barchart, intertopic distance map, similarity heatmap, document clusters, topic hierarchy, and word clouds
- **Rich CLI** with progress bars, formatted tables, and colored output

## Prerequisites

- Python 3.12+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (`sudo apt install tesseract-ocr`)
- [Poppler](https://poppler.freedesktop.org/) (`sudo apt install poppler-utils`)
- [SurrealDB](https://surrealdb.com/install) (`curl -sSf https://install.surrealdb.com | sh`)
- Build tools (`sudo apt install python3-dev build-essential`)

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

### 5. Topic Discovery + Visualizations

```bash
python3 -m app.main topics discover pdfs/ --min-topic-size 2 --output-dir output
```

This will:
1. Extract text from all PDFs in the folder
2. Generate embeddings for each text chunk
3. Run BERTopic to discover topics
4. Generate interactive visualizations in the output directory

Options:
- `--min-topic-size` — Minimum documents per topic (default: 3)
- `--nr-topics` — Target number of topics (auto if unset)
- `--output-dir` — Directory for HTML/PNG visualizations (default: output)

### Generated Visualizations

| Visualization | File | Description |
|---|---|---|
| Topic Overview | `topic_overview.html` | Bar chart of topic distribution |
| Word Scores | `topic_barchart.html` | Top words per topic with scores |
| Intertopic Map | `intertopic_map.html` | 2D topic distance visualization |
| Similarity Heatmap | `topic_heatmap.html` | Topic-to-topic similarity matrix |
| Document Clusters | `document_clusters.html` | 2D document scatter by topic |
| Topic Hierarchy | `topic_hierarchy.html` | Hierarchical clustering dendrogram |
| Word Clouds | `wordclouds/*.png` | Per-topic word cloud images |
