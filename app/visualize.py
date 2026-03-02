"""Interactive visualizations for topic modeling results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_topic_barchart(
    model: BERTopic,
    output_path: str | Path,
    top_n_topics: int = 10,
    n_words: int = 8,
) -> Path:
    """Generate and save a topic word-score barchart as HTML."""
    output_path = Path(output_path)
    fig = model.visualize_barchart(
        top_n_topics=top_n_topics,
        n_words=n_words,
        title="Topic Word Scores",
    )
    fig.write_html(str(output_path))
    return output_path


def save_topic_similarity_heatmap(
    model: BERTopic,
    output_path: str | Path,
) -> Path:
    """Generate and save a topic similarity heatmap as HTML."""
    output_path = Path(output_path)
    fig = model.visualize_heatmap(title="Topic Similarity Matrix")
    fig.write_html(str(output_path))
    return output_path


def save_intertopic_distance_map(
    model: BERTopic,
    output_path: str | Path,
) -> Path:
    """Generate and save the intertopic distance map as HTML."""
    output_path = Path(output_path)
    fig = model.visualize_topics(title="Intertopic Distance Map")
    fig.write_html(str(output_path))
    return output_path


def save_document_clusters(
    model: BERTopic,
    docs: list[str],
    reduced_embeddings: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Generate and save a 2D document cluster visualization as HTML."""
    output_path = Path(output_path)
    fig = model.visualize_documents(
        docs,
        reduced_embeddings=reduced_embeddings,
        title="Document Clusters by Topic",
        hide_document_hover=False,
    )
    fig.write_html(str(output_path))
    return output_path


def save_topic_hierarchy(
    model: BERTopic,
    output_path: str | Path,
) -> Path:
    """Generate and save a topic hierarchy dendrogram as HTML."""
    output_path = Path(output_path)
    fig = model.visualize_hierarchy(title="Topic Hierarchy")
    fig.write_html(str(output_path))
    return output_path


def save_topic_wordclouds(
    model: BERTopic,
    output_dir: str | Path,
) -> list[Path]:
    """Generate word cloud images for each topic."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    topics = model.get_topic_info()
    real_topics = topics[topics["Topic"] != -1]

    for _, row in real_topics.iterrows():
        topic_id = row["Topic"]
        words = model.get_topic(topic_id)
        if not words:
            continue

        word_freq = {word: float(score) for word, score in words}

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=30,
        ).generate_from_frequencies(word_freq)

        label = row.get("Name", f"Topic_{topic_id}")
        safe_label = label.replace(" ", "_").replace("/", "_")[:40]
        path = output_dir / f"wordcloud_topic_{topic_id}_{safe_label}.png"

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Topic {topic_id}: {label}", fontsize=14, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        saved.append(path)

    return saved


def save_topic_overview(
    model: BERTopic,
    docs: list[str],
    topics: list[int],
    output_path: str | Path,
) -> Path:
    """Generate a comprehensive single-page topic overview as HTML."""
    output_path = Path(output_path)
    topic_info = model.get_topic_info()

    labels = []
    sizes = []
    colors = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        name = row.get("Name", f"Topic {tid}")
        labels.append(name)
        sizes.append(row["Count"])
        colors.append(tid)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=sizes,
        marker=dict(
            color=colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Topic ID"),
        ),
        text=sizes,
        textposition="auto",
    ))

    fig.update_layout(
        title="Topic Distribution Overview",
        xaxis_title="Topics",
        yaxis_title="Number of Documents",
        template="plotly_white",
        height=500,
        xaxis_tickangle=-45,
    )

    fig.write_html(str(output_path))
    return output_path
