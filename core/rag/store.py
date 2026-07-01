"""
VECTOR STORE
============
Persists embeddings to disk and enables similarity search.

We use ChromaDB — a lightweight vector database that:
- Stores vectors and their associated text on disk
- Computes similarity between vectors efficiently
- Requires zero server setup (it's just a folder on disk)

How similarity search works:
  Given a query vector Q and stored vectors [V1, V2, V3...],
  ChromaDB computes the "distance" between Q and every stored vector.
  The vectors with the smallest distance = most similar meaning.

  Distance metric we use: cosine similarity
  - 1.0 = identical meaning
  - 0.0 = completely unrelated
  - Works regardless of vector length (normalized)
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path


STORE_PATH = Path(__file__).parent.parent.parent / "data" / "vector_store"


class VectorStore:
    """
    Wraps ChromaDB for storing and searching document embeddings.

    Each "collection" is a named group of vectors — like a table
    in a regular database. We use one collection per document source
    so you can search within a specific source or across all of them.

    Args:
        store_path  : directory where ChromaDB persists data to disk
        collection  : name of the collection to use
    """

    def __init__(
        self,
        store_path: str | None = None,
        collection: str = "documents",
    ):
        path = Path(store_path or STORE_PATH)
        path.mkdir(parents=True, exist_ok=True)

        # PersistentClient stores data to disk
        # Data survives process restarts — unlike an in-memory store
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection_name = collection

        # get_or_create: if collection exists, load it; if not, create it
        # This means you can safely call VectorStore() multiple times
        # without duplicating data
        self._collection = self._client.get_or_create_collection(
            name=collection,
            # cosine distance: best for semantic similarity
            # alternative: "l2" (euclidean), "ip" (inner product)
            metadata={"hnsw:space": "cosine"},
        )

        print(f"[VectorStore] Collection '{collection}' "
              f"({self._collection.count()} vectors)")

    # ── WRITING ──────────────────────────────────────────────────────

    def add(self, embedded_chunks: list[dict]) -> None:
        """
        Add embedded chunks to the store.

        ChromaDB requires:
        - ids      : unique string ID for each vector
        - embeddings: the actual vectors
        - documents : the original text (stored alongside for retrieval)
        - metadatas : any extra info (source, chunk_index, etc.)

        Args:
            embedded_chunks : output from Embedder.embed_chunks()
        """
        if not embedded_chunks:
            return

        ids         = []
        embeddings  = []
        documents   = []
        metadatas   = []

        for i, chunk in enumerate(embedded_chunks):
            # ID must be unique across the entire collection
            # source + chunk_index gives us a natural unique key
            chunk_id = f"{chunk['source']}::chunk_{chunk['chunk_index']}"

            ids.append(chunk_id)
            embeddings.append(chunk['embedding'])
            documents.append(chunk['text'])
            metadatas.append({
                "source":      chunk['source'],
                "chunk_index": chunk['chunk_index'],
                **chunk.get('metadata', {}),
            })

        # upsert: insert if new, update if ID already exists
        # Safe to call multiple times with the same documents
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"[VectorStore] Added {len(embedded_chunks)} chunks. "
              f"Total: {self._collection.count()}")

    # ── SEARCHING ────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        Find the top_k most similar chunks to a query vector.

        Args:
            query_embedding : vector from Embedder.embed_text(query)
            top_k           : how many results to return
            source_filter   : if set, only search within this source

        Returns:
            list of dicts with text, source, similarity score
            ordered by relevance (most relevant first)
        """
        where = {"source": source_filter} if source_filter else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns results in a nested list format
        # (supports batch queries, we only do one at a time)
        chunks = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append({
                    "text":       doc,
                    "source":     meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    # Convert distance to similarity score (0-1)
                    # cosine distance: 0=identical, 2=opposite
                    # similarity: 1=identical, 0=unrelated
                    "similarity": round(1 - dist / 2, 4),
                })

        return chunks

    def count(self) -> int:
        """Total number of vectors stored."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete all vectors in this collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[VectorStore] Cleared collection '{self._collection_name}'")
