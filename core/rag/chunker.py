"""
CHUNKER
=======
Splits documents into overlapping chunks for embedding and retrieval.

Key concept: OVERLAP
When you split a document into chunks, important information sometimes
sits right at the boundary between two chunks. Overlap solves this by
letting adjacent chunks share some text:

  Chunk 1: "...the boiling point of water is 100 degrees Celsius at"
  Chunk 2: "at sea level, which is equivalent to 212 degrees Fahrenheit..."

Without overlap, the complete fact gets split across chunks and
neither chunk alone answers "what is the boiling point of water?"
With overlap, chunk 2 starts before the split point and captures
the full context.

Typical values:
  chunk_size:    200-500 tokens for factual content
  overlap:       10-20% of chunk_size
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """
    A single piece of a document ready for embedding.

    Fields:
        text       : the actual text content
        source     : where this came from (filename, URL, etc.)
        chunk_index: position in the original document
        metadata   : any extra info (page number, section title, etc.)
    """
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.text[:50].replace('\n', ' ')
        return f"Chunk({self.source}[{self.chunk_index}]: '{preview}...')"


class Chunker:
    """
    Splits text into overlapping chunks of roughly equal size.

    Args:
        chunk_size : target size of each chunk in characters
                     (~200 chars ≈ ~50 tokens, good for factual content)
        overlap    : how many characters adjacent chunks share
                     (~20% of chunk_size is a good default)
    """

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, source: str, metadata: dict = {}) -> list[Chunk]:
        """
        Split a text string into overlapping chunks.

        Strategy: split on sentence boundaries (periods, newlines)
        rather than hard character cuts — this prevents cutting mid-sentence
        which would produce incoherent chunks.

        Args:
            text     : the full document text to chunk
            source   : identifier for where this text came from
            metadata : optional extra info to attach to all chunks

        Returns:
            list of Chunk objects ready for embedding
        """
        if not text.strip():
            return []

        # Clean up whitespace
        text = ' '.join(text.split())

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                # Last chunk — take everything remaining
                chunk_text = text[start:]
            else:
                # Find the nearest sentence boundary before the cut point
                # Look for '. ', '? ', '! ', or '\n' near the end of our window
                boundary = self._find_boundary(text, end)
                chunk_text = text[start:boundary]
                end = boundary

            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_index,
                    metadata=metadata,
                ))
                chunk_index += 1

            # Move start forward by chunk_size minus overlap
            # This is what creates the overlapping window
            start = end - self.overlap
            if start >= len(text):
                break

        return chunks

    def chunk_documents(self, documents: list[dict]) -> list[Chunk]:
        """
        Chunk multiple documents at once.

        Args:
            documents: list of dicts with 'text', 'source', and optional 'metadata'

        Returns:
            flat list of all chunks across all documents
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(
                text=doc['text'],
                source=doc['source'],
                metadata=doc.get('metadata', {}),
            )
            all_chunks.extend(chunks)
            print(f"  [Chunker] {doc['source']}: {len(chunks)} chunks")
        return all_chunks

    def _find_boundary(self, text: str, position: int) -> int:
        """
        Find the nearest sentence boundary at or before position.
        Falls back to the exact position if no boundary found nearby.
        """
        # Look back up to 100 chars for a sentence boundary
        search_start = max(0, position - 100)
        window = text[search_start:position]

        # Find the last sentence-ending punctuation in the window
        for punct in ['. ', '? ', '! ', '\n']:
            idx = window.rfind(punct)
            if idx != -1:
                # Return position relative to full text
                return search_start + idx + len(punct)

        # No boundary found — cut at exact position
        return position
