"""
EMBEDDER
========
Converts text chunks into numerical vectors (embeddings) using
a local sentence-transformer model.

Why local instead of an API?
- OpenAI's embedding API costs money per token
- sentence-transformers runs on CPU, completely free
- all-MiniLM-L6-v2 is only 22MB and loads in ~2 seconds
- Quality is sufficient for retrieval tasks

The model produces 384-dimensional vectors — meaning each piece
of text becomes a list of 384 floating point numbers that capture
its semantic meaning.
"""

from sentence_transformers import SentenceTransformer
from .chunker import Chunk


class Embedder:
    """
    Wraps a sentence-transformer model for embedding text.

    The model is loaded once at initialization and reused
    for all subsequent embed calls — same pattern as QwenLocalAdapter.

    Args:
        model_name : sentence-transformers model to use.
                     all-MiniLM-L6-v2 is the best speed/quality tradeoff
                     for retrieval tasks on CPU.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"[Embedder] Loading {model_name}...")
        # Downloads on first run (~22MB), cached locally after that
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        print(f"[Embedder] Ready. Embedding dimension: {self.dimension}")

    @property
    def dimension(self) -> int:
        """The size of vectors this model produces."""
        return self._model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """
        Convert a single string to a vector.
        Used for embedding the USER'S QUERY at retrieval time.

        Args:
            text : the string to embed

        Returns:
            list of floats representing the text's meaning
        """
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_chunks(self, chunks: list[Chunk]) -> list[dict]:
        """
        Convert a list of Chunks to embeddable dicts.
        Used at INDEX TIME — when you're loading documents.

        Batches all chunks together for efficiency — running
        the model once on 100 chunks is much faster than
        running it 100 times on 1 chunk each.

        Args:
            chunks : list of Chunk objects from Chunker

        Returns:
            list of dicts with text, embedding, source, metadata
            ready to insert into the vector store
        """
        if not chunks:
            return []

        # Extract just the text for batch encoding
        texts = [chunk.text for chunk in chunks]

        print(f"[Embedder] Embedding {len(texts)} chunks...")

        # encode() runs the model on all texts at once
        # show_progress_bar=True shows a progress bar for large batches
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
            batch_size=32,  # process 32 chunks at a time
        )

        # Combine chunk metadata with its embedding
        embedded = []
        for chunk, vector in zip(chunks, vectors):
            embedded.append({
                "text":       chunk.text,
                "embedding":  vector.tolist(),
                "source":     chunk.source,
                "chunk_index": chunk.chunk_index,
                "metadata":   chunk.metadata,
            })

        print(f"[Embedder] Done. Each vector has {len(vectors[0])} dimensions.")
        return embedded
