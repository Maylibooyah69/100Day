"""Text chunking utilities for vector indexing."""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by character count.

    Tries to break on sentence boundaries when possible.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            break_point = _find_break_point(text, start, end)
            if break_point > start:
                end = break_point

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


def _find_break_point(text: str, start: int, end: int) -> int:
    """Find the best sentence/paragraph break point near `end`."""
    search_region = text[max(start, end - 100):end]

    for sep in ["\n\n", ".\n", ". ", "? ", "! ", "\n", "; ", ", "]:
        idx = search_region.rfind(sep)
        if idx != -1:
            return max(start, end - 100) + idx + len(sep)

    return end
