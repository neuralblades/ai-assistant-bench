"""
DOCUMENT SOURCE — Source 1
Fixed documents loaded from disk at startup.
These are pre-indexed once and reused across all queries.
"""
from pathlib import Path
from ..retriever import Retriever


def load_knowledge_base(
    docs_dir: str,
    retriever: Retriever,
) -> int:
    """
    Load all .txt and .md files from a directory into the retriever.

    Args:
        docs_dir  : path to folder containing your documents
        retriever : Retriever instance to index into

    Returns:
        total chunks indexed
    """
    path = Path(docs_dir)
    if not path.exists():
        print(f"[DocumentSource] Directory not found: {docs_dir}")
        return 0

    total = 0
    files = list(path.glob("*.txt")) + list(path.glob("*.md"))

    if not files:
        print(f"[DocumentSource] No .txt or .md files found in {docs_dir}")
        return 0

    print(f"[DocumentSource] Loading {len(files)} files from {docs_dir}...")
    for f in files:
        count = retriever.index_file(str(f))
        print(f"  {f.name}: {count} chunks")
        total += count

    print(f"[DocumentSource] Done. {total} total chunks indexed.")
    return total
