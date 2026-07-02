"""
RAG — Retrieval Augmented Generation
=====================================
Public interface for the RAG module.

Three retrieval sources:
  1. Document store  — fixed files indexed at startup
  2. User uploads    — files uploaded through the UI
  3. Web search      — live DuckDuckGo results per query

Typical usage:
    from core.rag import RAGPipeline

    rag = RAGPipeline()
    rag.index_knowledge_base("data/docs/")        # Source 1
    context = rag.retrieve("What is recursion?")  # auto-selects best source
    # inject context into system prompt before model call
"""

from .retriever import Retriever
from .embedder import Embedder
from .chunker import Chunker, Chunk
from .store import VectorStore


class RAGPipeline:
    """
    Unified interface for all three retrieval sources.

    Manages one shared Embedder (model loaded once, reused everywhere)
    and separate Retrievers per source type.

    Args:
        use_web      : enable live web search fallback
        collection   : name for the fixed document store collection
        top_k        : chunks to retrieve per query
        min_similarity: minimum similarity score to include a chunk
    """

    def __init__(
        self,
        use_web: bool = True,
        collection: str = "knowledge_base",
        top_k: int = 3,
        min_similarity: float = 0.6,
    ):
        # One embedder shared across all sources
        # Loading it once here means it's ready for all retrieve calls
        self._embedder       = Embedder()
        self._top_k          = top_k
        self._min_similarity = min_similarity
        self._use_web        = use_web

        # Source 1: fixed document store
        self._doc_retriever = Retriever(
            collection=collection,
            top_k=top_k,
            embedder=self._embedder,
        )

        # Source 2: per-session upload retrievers
        # key = session_id, value = Retriever for that session
        self._upload_retrievers: dict[str, Retriever] = {}

        print(f"[RAG] Pipeline ready. "
              f"Document store: {self._doc_retriever.document_count} chunks. "
              f"Web search: {'enabled' if use_web else 'disabled'}")

    # ── INDEXING ─────────────────────────────────────────────────────

    def index_knowledge_base(self, docs_dir: str) -> int:
        """Load all .txt/.md files from a directory. Source 1."""
        from .sources.document import load_knowledge_base
        return load_knowledge_base(docs_dir, self._doc_retriever)

    def index_upload(self, file_path: str, session_id: str) -> int:
        """Index a user-uploaded file for a specific session. Source 2."""
        from .sources.upload import handle_upload
        retriever, count = handle_upload(file_path, session_id)
        self._upload_retrievers[session_id] = retriever
        return count

    # ── RETRIEVAL ────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        source: str = "auto",
    ) -> list[dict]:
        """
        Retrieve relevant chunks for a query.

        Source selection:
        "auto"     → uploads first, then doc store, then web fallback
        "document" → only search fixed document store
        "upload"   → only search this session's uploaded documents
        "web"      → only live web search

        Priority in "auto" mode:
        1. Session uploads (user explicitly provided these — highest intent)
        2. Fixed document store (pre-loaded knowledge base)
        3. Web search (live fallback when nothing local matches)

        Args:
            query      : the user's question
            session_id : required for upload source to work
            source     : which source(s) to search

        Returns:
            list of relevant chunks ordered by similarity
        """
        if source == "document":
            return self._doc_retriever.retrieve(
                query,
                min_similarity=self._min_similarity,
            )

        elif source == "upload":
            if not session_id or session_id not in self._upload_retrievers:
                print("[RAG] No uploaded documents for this session.")
                return []
            return self._upload_retrievers[session_id].retrieve(
                query,
                min_similarity=self._min_similarity,
            )

        elif source == "web":
            return self._web_retrieve(query)

        else:  # "auto" — uploads take priority over fixed store
            # Step 1: Check session uploads FIRST
            # User explicitly uploaded this — highest priority
            if session_id and session_id in self._upload_retrievers:
                upload_results = self._upload_retrievers[session_id].retrieve(
                    query,
                    min_similarity=self._min_similarity,
                )
                if upload_results:
                    print(f"[RAG] Found {len(upload_results)} results from upload")
                    return upload_results

            # Step 2: Check fixed document store
            doc_results = self._doc_retriever.retrieve(
                query,
                min_similarity=self._min_similarity,
            )
            if doc_results:
                print(f"[RAG] Found {len(doc_results)} results from doc store")
                return doc_results

            # Step 3: Fall back to web search
            if self._use_web:
                print("[RAG] No local results — falling back to web search")
                return self._web_retrieve(query)

            return []
            
    def format_context(self, chunks: list[dict]) -> str:
        """Format chunks for injection into the model's system prompt."""
        return self._doc_retriever.format_context(chunks)

    def build_augmented_prompt(
        self,
        base_system_prompt: str,
        query: str,
        session_id: str | None = None,
        source: str = "auto",
    ) -> tuple[str, str]:
        """
        Returns (system_prompt, context_string) as separate values.
        Context is returned separately so the caller can inject it
        into the user turn rather than the system prompt — models
        follow user-turn context more reliably than system-prompt context
        when it conflicts with pretraining knowledge.
        """
        chunks = self.retrieve(query, session_id=session_id, source=source)

        if not chunks:
            return base_system_prompt, ""

        context = self.format_context(chunks)
        return base_system_prompt, context

    # ── PRIVATE ──────────────────────────────────────────────────────

    def _web_retrieve(self, query: str) -> list[dict]:
        """Live web search retrieval."""
        from .sources.web import web_retrieve
        return web_retrieve(
            query=query,
            embedder=self._embedder,
            top_k=self._top_k,
        )

    # ── PROPERTIES ───────────────────────────────────────────────────

    @property
    def indexed_chunks(self) -> int:
        """Total chunks in the fixed document store."""
        return self._doc_retriever.document_count
