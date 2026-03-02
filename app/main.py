"""CLI entry point for PDF OCR + SurrealDB vector indexing + topic modeling."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from app.ocr import extract_text_from_pdf, scan_pdf_folder
from app.chunker import chunk_text
from app.embeddings import generate_embeddings, generate_embedding, MODEL_NAME, DIMENSION
from app.db import get_connection, init_schema, store_document, search_similar, get_stats

console = Console()

DEFAULT_OUTPUT_DIR = Path("output")


@click.group()
def cli():
    """PDF OCR & Vector Search powered by SurrealDB + Topic Modeling."""
    pass


# ---------------------------------------------------------------------------
# Index command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--chunk-size", default=500, help="Characters per chunk.")
@click.option("--overlap", default=50, help="Overlap between chunks.")
def index(folder: str, chunk_size: int, overlap: int):
    """Scan a folder of PDFs, extract text via OCR, and index in SurrealDB."""
    pdf_files = scan_pdf_folder(folder)
    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {folder}[/yellow]")
        return

    console.print(f"[bold]Found {len(pdf_files)} PDF(s) in [cyan]{folder}[/cyan][/bold]")
    console.print(f"Embedding model: [cyan]{MODEL_NAME}[/cyan] ({DIMENSION}d)")

    db = get_connection()
    init_schema(db)
    console.print("[green]✓[/green] Connected to SurrealDB, schema initialized\n")

    total_chunks = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing PDFs...", total=len(pdf_files))

        for pdf_path in pdf_files:
            progress.update(task, description=f"Processing {pdf_path.name}...")

            doc_data = extract_text_from_pdf(pdf_path)

            all_chunks = []
            for page in doc_data["pages"]:
                page_chunks = chunk_text(page["text"], chunk_size, overlap)
                for i, chunk_text_str in enumerate(page_chunks):
                    all_chunks.append({
                        "page_number": page["page_number"],
                        "chunk_index": i,
                        "text": chunk_text_str,
                    })

            if not all_chunks:
                console.print(f"  [yellow]⚠ {pdf_path.name}: no text extracted[/yellow]")
                progress.advance(task)
                continue

            texts = [c["text"] for c in all_chunks]
            embeddings = generate_embeddings(texts)
            for c, emb in zip(all_chunks, embeddings):
                c["embedding"] = emb

            doc_id = store_document(db, doc_data, all_chunks)
            total_chunks += len(all_chunks)

            ocr_flag = " [dim](OCR)[/dim]" if doc_data["ocr_used"] else ""
            console.print(
                f"  [green]✓[/green] {pdf_path.name}: "
                f"{doc_data['num_pages']} pages, "
                f"{len(all_chunks)} chunks{ocr_flag}"
            )
            progress.advance(task)

    stats = get_stats(db)
    db.close()

    console.print(f"\n[bold green]Indexing complete![/bold green]")
    console.print(f"  Documents: {stats['documents']}  |  Chunks: {stats['chunks']}")


# ---------------------------------------------------------------------------
# Search command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of results to return.")
def search(query: str, top_k: int):
    """Search indexed PDFs using natural language."""
    console.print(f"[bold]Query:[/bold] {query}")
    console.print(f"Generating embedding...", end=" ")

    query_emb = generate_embedding(query)
    console.print("[green]done[/green]\n")

    db = get_connection()
    results = search_similar(db, query_emb, top_k)
    db.close()

    if not results:
        console.print("[yellow]No results found. Have you indexed any PDFs?[/yellow]")
        return

    table = Table(title=f"Top {top_k} Results", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("File", style="cyan", max_width=30)
    table.add_column("Page", justify="center", width=5)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Text", max_width=80)

    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        text_preview = r.get("text", "")[:200] + ("..." if len(r.get("text", "")) > 200 else "")
        table.add_row(
            str(i),
            str(r.get("filename", "?")),
            str(r.get("page_number", "?")),
            score_str,
            text_preview,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Stats command
# ---------------------------------------------------------------------------

@cli.command()
def stats():
    """Show indexing statistics."""
    db = get_connection()
    s = get_stats(db)
    db.close()

    console.print(f"[bold]Database Stats[/bold]")
    console.print(f"  Documents indexed: [cyan]{s['documents']}[/cyan]")
    console.print(f"  Total chunks:      [cyan]{s['chunks']}[/cyan]")


# ---------------------------------------------------------------------------
# Topics command group
# ---------------------------------------------------------------------------

@cli.group()
def topics():
    """Topic modeling and visualization commands."""
    pass


@topics.command("discover")
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--chunk-size", default=500, help="Characters per chunk.")
@click.option("--overlap", default=50, help="Overlap between chunks.")
@click.option("--min-topic-size", default=3, help="Minimum documents per topic.")
@click.option("--nr-topics", default=None, type=int, help="Target number of topics (auto if unset).")
@click.option("--output-dir", default="output", help="Directory for visualizations.")
def topics_discover(folder: str, chunk_size: int, overlap: int, min_topic_size: int, nr_topics: int | None, output_dir: str):
    """Discover topics in a folder of PDFs and generate visualizations."""
    from app.topics import discover_topics, get_topic_summary, reduce_embeddings_2d
    from app.visualize import (
        save_topic_barchart,
        save_topic_similarity_heatmap,
        save_intertopic_distance_map,
        save_document_clusters,
        save_topic_wordclouds,
        save_topic_overview,
        save_topic_hierarchy,
    )
    import numpy as np

    pdf_files = scan_pdf_folder(folder)
    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {folder}[/yellow]")
        return

    console.print(Panel.fit(
        f"[bold]Topic Discovery[/bold]\n"
        f"PDFs: [cyan]{len(pdf_files)}[/cyan] in {folder}\n"
        f"Model: [cyan]{MODEL_NAME}[/cyan] ({DIMENSION}d)\n"
        f"Min topic size: [cyan]{min_topic_size}[/cyan]",
        title="Configuration",
    ))

    # --- Step 1: Extract text ---
    console.print("\n[bold]Step 1/4:[/bold] Extracting text from PDFs...")
    all_texts = []
    all_sources = []

    for pdf_path in pdf_files:
        doc_data = extract_text_from_pdf(pdf_path)
        for page in doc_data["pages"]:
            chunks = chunk_text(page["text"], chunk_size, overlap)
            for c in chunks:
                all_texts.append(c)
                all_sources.append(f"{pdf_path.name} p{page['page_number']}")

    console.print(f"  [green]✓[/green] Extracted [cyan]{len(all_texts)}[/cyan] chunks from {len(pdf_files)} PDFs")

    if len(all_texts) < 4:
        console.print("[yellow]⚠ Too few text chunks for topic modeling (need at least 4). Add more PDFs.[/yellow]")
        return

    # --- Step 2: Generate embeddings ---
    console.print("\n[bold]Step 2/4:[/bold] Generating embeddings...")
    embeddings = generate_embeddings(all_texts)
    emb_array = np.array(embeddings)
    console.print(f"  [green]✓[/green] Generated {len(embeddings)} embeddings ({DIMENSION}d)")

    # --- Step 3: Discover topics ---
    console.print("\n[bold]Step 3/4:[/bold] Running BERTopic...")
    target_topics = nr_topics if nr_topics else "auto"
    model, topic_ids, probs = discover_topics(
        all_texts,
        embeddings=embeddings,
        min_topic_size=min_topic_size,
        nr_topics=target_topics,
    )

    summaries = get_topic_summary(model)
    real_topics = [s for s in summaries if s["topic_id"] != -1]

    console.print(f"  [green]✓[/green] Discovered [bold cyan]{len(real_topics)}[/bold cyan] topics\n")

    table = Table(title="Discovered Topics", show_lines=True)
    table.add_column("ID", style="bold", width=4)
    table.add_column("Label", style="cyan", max_width=40)
    table.add_column("Docs", justify="center", width=5)
    table.add_column("Top Words", max_width=60)

    for s in summaries:
        style = "dim" if s["topic_id"] == -1 else ""
        table.add_row(
            str(s["topic_id"]),
            s["label"],
            str(s["count"]),
            ", ".join(s["top_words"][:6]),
            style=style,
        )
    console.print(table)

    # --- Step 4: Generate visualizations ---
    console.print(f"\n[bold]Step 4/4:[/bold] Generating visualizations → [cyan]{output_dir}/[/cyan]")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated = []

    try:
        p = save_topic_overview(model, all_texts, topic_ids, out / "topic_overview.html")
        generated.append(("Topic Overview", p))
        console.print(f"  [green]✓[/green] Topic overview → {p}")
    except Exception as e:
        console.print(f"  [yellow]⚠[/yellow] Topic overview: {e}")

    try:
        p = save_topic_barchart(model, out / "topic_barchart.html")
        generated.append(("Topic Barchart", p))
        console.print(f"  [green]✓[/green] Topic barchart → {p}")
    except Exception as e:
        console.print(f"  [yellow]⚠[/yellow] Topic barchart: {e}")

    if len(real_topics) >= 2:
        try:
            p = save_intertopic_distance_map(model, out / "intertopic_map.html")
            generated.append(("Intertopic Map", p))
            console.print(f"  [green]✓[/green] Intertopic map → {p}")
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] Intertopic map: {e}")

        try:
            p = save_topic_similarity_heatmap(model, out / "topic_heatmap.html")
            generated.append(("Similarity Heatmap", p))
            console.print(f"  [green]✓[/green] Similarity heatmap → {p}")
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] Similarity heatmap: {e}")

        try:
            p = save_topic_hierarchy(model, out / "topic_hierarchy.html")
            generated.append(("Topic Hierarchy", p))
            console.print(f"  [green]✓[/green] Topic hierarchy → {p}")
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] Topic hierarchy: {e}")

    try:
        console.print("  Reducing embeddings to 2D...")
        reduced = reduce_embeddings_2d(emb_array)
        p = save_document_clusters(model, all_texts, reduced, out / "document_clusters.html")
        generated.append(("Document Clusters", p))
        console.print(f"  [green]✓[/green] Document clusters → {p}")
    except Exception as e:
        console.print(f"  [yellow]⚠[/yellow] Document clusters: {e}")

    try:
        paths = save_topic_wordclouds(model, out / "wordclouds")
        for p in paths:
            generated.append(("Word Cloud", p))
        console.print(f"  [green]✓[/green] Word clouds ({len(paths)} images) → {out / 'wordclouds/'}")
    except Exception as e:
        console.print(f"  [yellow]⚠[/yellow] Word clouds: {e}")

    console.print(f"\n[bold green]Done![/bold green] Generated {len(generated)} visualizations in [cyan]{output_dir}/[/cyan]")
    console.print("[dim]Open the HTML files in a browser to explore interactively.[/dim]")


if __name__ == "__main__":
    cli()
