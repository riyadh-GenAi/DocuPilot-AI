"""Small, deterministic character-based text chunker."""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring whitespace boundaries."""
    text = " ".join(text.split())
    if not text:
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [chunk for chunk in chunks if chunk]
