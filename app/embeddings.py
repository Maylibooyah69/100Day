"""Embedding generation using sentence-transformers."""

from sentence_transformers import SentenceTransformer

_model = None
MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSION = 384


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for a list of text strings."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [emb.tolist() for emb in embeddings]


def generate_embedding(text: str) -> list[float]:
    """Generate a single embedding vector."""
    return generate_embeddings([text])[0]
