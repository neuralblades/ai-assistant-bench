"""
RETRIEVER
=========
The main interface for the RAG system.
Orchestrates: document loading → chunking → embedding → storing → searching.

This is the only file the rest of your app imports from.
Everything else (chunker, embedder, store) is internal plumbing.

Usage:
    # Index documents once
    retriever = Retriever()
    retriever.index_text("Paris is the capital of France.", source="facts.txt")

    # Retrieve at query time
    chunks = retriever.retrieve("What is France's capital?")
    # → [{"text": "Paris is the capital of France.", "similarity": 0.94, ...}]

    # Inject into model prompt
    context = retriever.format_context(chunks)
    # → "Relevant context:\n[1] Paris is the capital of France.\n"
"""

from pathlib import Path
from .chunker import Chunker
from .embedder import Embedder
from .store import VectorStore


class Retriever:
    """
    Single entry point for all RAG operations.

    Args:
        collection  : name of the vector store collection
        chunk_size  : target chunk size in characters
        overlap     : overlap between adjacent chunks
        top_k       : how many chunks to retrieve per query
        store_path  : where to persist vectors (None = default)
    """

    def __init__(
        self,
        collection: str = "documents",
        chunk_size: int = 400,
        overlap: int = 80,
        top_k: int = 3,
        store_path: str | None = None,
        embedder: Embedder | None = None,  # ← add this
    ):
        self._chunker  = Chunker(chunk_size=chunk_size, overlap=overlap)
        self._embedder = embedder or Embedder()
        self._store    = VectorStore(collection=collection, store_path=store_path)
        self._top_k    = top_k

    # ── INDEXING ─────────────────────────────────────────────────────

    def index_text(self, text: str, source: str, metadata: dict = {}) -> int:
        """
        Index a raw text string into the vector store.

        Pipeline: text → chunks → embeddings → stored

        Args:
            text     : the document text to index
            source   : identifier (filename, URL, etc.)
            metadata : extra info to attach to all chunks

        Returns:
            number of chunks created and stored
        """
        chunks   = self._chunker.chunk_text(text, source, metadata)
        embedded = self._embedder.embed_chunks(chunks)
        self._store.add(embedded)
        return len(chunks)

    def index_file(self, filepath: str) -> int:
        """
        Read a text file and index its contents.

        Args:
            filepath : path to .txt file

        Returns:
            number of chunks created
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        if path.suffix not in ['.txt', '.md']:
            raise ValueError(f"Only .txt and .md files supported. Got: {path.suffix}")

        text = path.read_text(encoding='utf-8')
        return self.index_text(text, source=path.name)

    def index_documents(self, documents: list[dict]) -> int:
        """
        Index multiple documents at once.

        Args:
            documents: list of {"text": ..., "source": ..., "metadata": ...}

        Returns:
            total chunks created
        """
        total = 0
        for doc in documents:
            total += self.index_text(
                text=doc['text'],
                source=doc['source'],
                metadata=doc.get('metadata', {}),
            )
        return total

    # ── RETRIEVAL ────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
        min_similarity: float = 0.6,
    ) -> list[dict]:
        """
        Find the most relevant chunks for a query.

        Args:
            query          : the user's question
            top_k          : override default top_k for this query
            source_filter  : only search within a specific source
            min_similarity : filter out chunks below this threshold
                             prevents returning irrelevant results
                             when nothing in the store matches well

        Returns:
            list of relevant chunks, ordered by similarity
        """
        if self._store.count() == 0:
            return []

        query_vec = self._embedder.embed_text(query)
        results   = self._store.search(
            query_embedding=query_vec,
            top_k=top_k or self._top_k,
            source_filter=source_filter,
        )

        # Filter out low-similarity results
        # If nothing in the store is relevant, return empty
        # rather than feeding irrelevant context to the model
        filtered = [r for r in results if r['similarity'] >= min_similarity]
        return filtered

    def format_context(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a string for injection
        into the model's system prompt.

        Format:
            Relevant context:
            [1] (from facts.txt) Paris is the capital of France...
            [2] (from facts.txt) France is a country in Western Europe...

        Args:
            chunks : output from retrieve()

        Returns:
            formatted string ready to prepend to system prompt
        """
        if not chunks:
            return ""

        lines = ["Relevant context from knowledge base:"]
        for i, chunk in enumerate(chunks, 1):
            lines.append(
                f"[{i}] (from {chunk['source']}, "
                f"similarity: {chunk['similarity']}) "
                f"{chunk['text']}"
            )

        return "\n".join(lines)

    # ── UTILITIES ─────────────────────────────────────────────────────

    @property
    def document_count(self) -> int:
        """Total number of indexed chunks."""
        return self._store.count()

    def clear(self) -> None:
        """Wipe all indexed documents."""
        self._store.clear()
        print("[Retriever] All documents cleared.")
