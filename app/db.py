"""SurrealDB storage and vector search operations."""

from surrealdb import Surreal

from app.embeddings import DIMENSION

SURREAL_URL = "ws://localhost:8000/rpc"
SURREAL_USER = "root"
SURREAL_PASS = "root"
NAMESPACE = "pdf_ocr"
DATABASE = "documents"


def get_connection() -> Surreal:
    """Create and return a connected SurrealDB client."""
    db = Surreal(SURREAL_URL)
    db.signin({"username": SURREAL_USER, "password": SURREAL_PASS})
    db.use(NAMESPACE, DATABASE)
    return db


def init_schema(db: Surreal) -> None:
    """Initialize the database schema with vector index."""
    db.query("""
        DEFINE TABLE IF NOT EXISTS document SCHEMAFULL;
        DEFINE FIELD IF NOT EXISTS filename ON document TYPE string;
        DEFINE FIELD IF NOT EXISTS filepath ON document TYPE string;
        DEFINE FIELD IF NOT EXISTS num_pages ON document TYPE int;
        DEFINE FIELD IF NOT EXISTS ocr_used ON document TYPE bool;
        DEFINE FIELD IF NOT EXISTS indexed_at ON document TYPE datetime;
    """)

    db.query(f"""
        DEFINE TABLE IF NOT EXISTS chunk SCHEMAFULL;
        DEFINE FIELD IF NOT EXISTS doc ON chunk TYPE record<document>;
        DEFINE FIELD IF NOT EXISTS page_number ON chunk TYPE int;
        DEFINE FIELD IF NOT EXISTS chunk_index ON chunk TYPE int;
        DEFINE FIELD IF NOT EXISTS text ON chunk TYPE string;
        DEFINE FIELD IF NOT EXISTS embedding ON chunk TYPE array<float>;
        DEFINE INDEX IF NOT EXISTS idx_chunk_embedding ON chunk
            FIELDS embedding HNSW DIMENSION {DIMENSION} DIST COSINE TYPE F32;
    """)


def store_document(db: Surreal, doc_meta: dict, chunks: list[dict]) -> str:
    """Store a document and its chunks in SurrealDB.

    Returns the document record ID as a string.
    """
    result = db.query("""
        CREATE document CONTENT {
            filename: $filename,
            filepath: $filepath,
            num_pages: $num_pages,
            ocr_used: $ocr_used,
            indexed_at: time::now()
        };
    """, {
        "filename": doc_meta["filename"],
        "filepath": doc_meta["filepath"],
        "num_pages": doc_meta["num_pages"],
        "ocr_used": doc_meta["ocr_used"],
    })

    doc_id = _extract_id(result)

    for c in chunks:
        db.query("""
            CREATE chunk CONTENT {
                doc: $doc_id,
                page_number: $page_number,
                chunk_index: $chunk_index,
                text: $text,
                embedding: $embedding
            };
        """, {
            "doc_id": doc_id,
            "page_number": c["page_number"],
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "embedding": c["embedding"],
        })

    return str(doc_id)


def search_similar(db: Surreal, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Find the top_k most similar chunks to the query embedding."""
    result = db.query(f"""
        SELECT
            id,
            text,
            page_number,
            chunk_index,
            doc.filename AS filename,
            vector::similarity::cosine(embedding, $query_emb) AS score
        FROM chunk
        ORDER BY score DESC
        LIMIT {top_k};
    """, {"query_emb": query_embedding})

    if isinstance(result, list):
        return result
    return []


def get_stats(db: Surreal) -> dict:
    """Return counts of documents and chunks."""
    doc_result = db.query("SELECT count() FROM document GROUP ALL;")
    chunk_result = db.query("SELECT count() FROM chunk GROUP ALL;")

    return {
        "documents": _extract_count(doc_result),
        "chunks": _extract_count(chunk_result),
    }


def _extract_id(result):
    """Extract record ID from a SurrealDB query result."""
    if isinstance(result, list) and result:
        rec = result[0]
        if isinstance(rec, dict) and "id" in rec:
            return rec["id"]
    if isinstance(result, dict) and "id" in result:
        return result["id"]
    return ""


def _extract_count(result) -> int:
    """Extract count from a GROUP ALL query result."""
    if isinstance(result, list) and result:
        rec = result[0]
        if isinstance(rec, dict) and "count" in rec:
            return rec["count"]
    if isinstance(result, dict) and "count" in result:
        return result["count"]
    return 0
