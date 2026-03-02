"""Topic modeling using BERTopic with pre-computed embeddings."""

from __future__ import annotations

import numpy as np
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from umap import UMAP
from hdbscan import HDBSCAN

from app.embeddings import get_model


def build_topic_model(
    min_topic_size: int = 3,
    nr_topics: int | str | None = "auto",
    diversity: float = 0.3,
) -> BERTopic:
    """Create a configured BERTopic model.

    Uses the same sentence-transformer as the embedding pipeline,
    UMAP for dimensionality reduction, HDBSCAN for clustering,
    and MMR for diverse topic representations.
    """
    embedding_model = get_model()

    umap_model = UMAP(
        n_neighbors=min(15, max(2, min_topic_size)),
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=1,
        metric="euclidean",
        prediction_data=True,
    )

    representation_model = MaximalMarginalRelevance(diversity=diversity)

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        nr_topics=nr_topics,
        verbose=False,
    )


def discover_topics(
    texts: list[str],
    embeddings: list[list[float]] | None = None,
    min_topic_size: int = 3,
    nr_topics: int | str | None = "auto",
) -> tuple[BERTopic, list[int], np.ndarray]:
    """Run topic discovery on a list of text chunks.

    Returns (model, topic_ids, probabilities).
    """
    model = build_topic_model(min_topic_size=min_topic_size, nr_topics=nr_topics)

    emb_array = np.array(embeddings) if embeddings else None
    topics, probs = model.fit_transform(texts, embeddings=emb_array)

    return model, topics, probs


def get_topic_summary(model: BERTopic) -> list[dict]:
    """Return a summary of discovered topics."""
    topic_info = model.get_topic_info()
    summaries = []
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            label = "Outliers"
        else:
            label = row.get("Name", f"Topic {topic_id}")

        words = model.get_topic(topic_id)
        top_words = [w for w, _ in words[:8]] if words else []

        summaries.append({
            "topic_id": topic_id,
            "label": label,
            "count": row["Count"],
            "top_words": top_words,
        })

    return summaries


def reduce_embeddings_2d(embeddings: list[list[float]] | np.ndarray) -> np.ndarray:
    """Reduce embeddings to 2D for visualization."""
    emb = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
    reducer = UMAP(n_neighbors=min(15, max(2, len(emb) - 1)), n_components=2, metric="cosine", random_state=42)
    return reducer.fit_transform(emb)
