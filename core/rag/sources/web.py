"""
WEB SOURCE — Source 3
Fetches live web search results and indexes them temporarily.
No vector store — fetches fresh content per query.

Why no persistent store here?
Web content changes constantly. Caching it defeats the purpose
of live retrieval. Instead we fetch, chunk, embed, search,
and discard — all in memory, per query.
"""
import time
from ..chunker import Chunker
from ..embedder import Embedder


def web_retrieve(
    query: str,
    embedder: Embedder,
    top_k: int = 3,
    max_results: int = 5,
) -> list[dict]:
    """
    Search the web and return relevant chunks for a query.

    Uses DuckDuckGo search (no API key required).

    Pipeline:
        query → DuckDuckGo → snippets → chunk → embed → rank → return top_k

    Args:
        query      : user's question
        embedder   : shared Embedder instance (avoid reloading model)
        top_k      : how many chunks to return
        max_results: how many search results to fetch

    Returns:
        list of relevant chunks with source URLs
    """
    try:
        from ddgs import DDGS
    except ImportError:
        print("[WebSource] duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return []

    # Fetch search results
    chunks_raw = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        chunker = Chunker(chunk_size=300, overlap=60)

        for result in results:
            # Each result has: title, href (URL), body (snippet)
            text = f"{result.get('title', '')}. {result.get('body', '')}"
            if not text.strip():
                continue

            chunks = chunker.chunk_text(
                text=text,
                source=result.get('href', 'web'),
                metadata={"title": result.get('title', '')},
            )
            chunks_raw.extend(chunks)

    except Exception as e:
        print(f"[WebSource] Search failed: {e}")
        return []

    if not chunks_raw:
        return []

    # Embed all chunks and the query
    query_vec = embedder.embed_text(query)
    embedded  = embedder.embed_chunks(chunks_raw)

    # Rank by cosine similarity manually (no persistent store needed)
    import numpy as np
    query_arr = np.array(query_vec)

    scored = []
    for item in embedded:
        chunk_arr = np.array(item['embedding'])
        # Cosine similarity
        similarity = float(
            np.dot(query_arr, chunk_arr) /
            (np.linalg.norm(query_arr) * np.linalg.norm(chunk_arr) + 1e-8)
        )
        scored.append({
            "text":       item['text'],
            "source":     item['source'],
            "similarity": round(similarity, 4),
            "metadata":   item.get('metadata', {}),
        })

    # Sort by similarity, return top_k
    scored.sort(key=lambda x: x['similarity'], reverse=True)
    return scored[:top_k]
