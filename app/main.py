"""CLI entry point for PDF OCR + SurrealDB vector indexing."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from app.ocr import extract_text_from_pdf, scan_pdf_folder
from app.chunker import chunk_text
from app.embeddings import generate_embeddings, generate_embedding, MODEL_NAME, DIMENSION
from app.db import get_connection, init_schema, store_document, search_similar, get_stats

console = Console()


@click.group()
def cli():
    """PDF OCR & Vector Search powered by SurrealDB."""
    pass


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


@cli.command()
def stats():
    """Show indexing statistics."""
    db = get_connection()
    s = get_stats(db)
    db.close()

    console.print(f"[bold]Database Stats[/bold]")
    console.print(f"  Documents indexed: [cyan]{s['documents']}[/cyan]")
    console.print(f"  Total chunks:      [cyan]{s['chunks']}[/cyan]")


if __name__ == "__main__":
    cli()
