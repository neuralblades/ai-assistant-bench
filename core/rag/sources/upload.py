"""
UPLOAD SOURCE — Source 2
Handles documents uploaded by the user through the UI.
Each upload is indexed into a session-specific collection.
"""
import tempfile
from pathlib import Path
from ..retriever import Retriever


def handle_upload(
    file_path: str,
    session_id: str,
) -> tuple[Retriever, int]:
    """
    Index a user-uploaded file into a session-specific retriever.

    Each session gets its own collection so uploaded documents
    don't bleed between users.

    Args:
        file_path  : path to the uploaded file (Gradio provides this)
        session_id : unique ID for this user session

    Returns:
        (retriever for this session, number of chunks indexed)
    """
    collection = f"upload_{session_id}"
    retriever  = Retriever(collection=collection)

    path = Path(file_path)

    # Handle different file types
    if path.suffix == '.txt':
        count = retriever.index_file(file_path)

    elif path.suffix == '.pdf':
        # Extract text from PDF using PyMuPDF if available
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            text = "\n".join(page.get_text() for page in doc)
            count = retriever.index_text(text, source=path.name)
        except ImportError:
            print("[UploadSource] PyMuPDF not installed, cannot read PDF")
            count = 0

    elif path.suffix in ['.md']:
        count = retriever.index_file(file_path)

    else:
        print(f"[UploadSource] Unsupported file type: {path.suffix}")
        count = 0

    print(f"[UploadSource] Indexed {count} chunks from {path.name}")
    return retriever, count
