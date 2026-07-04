from app.loaders import load_documents
from app.splitter import split_documents
from app.vectordb import (
    create_vector_db,
    add_documents_to_db,
)

from pathlib import Path
from app.config import CHROMA_PATH

def ingest_documents(data_path):
    documents = load_documents(data_path)

    print(f"Documents: {len(documents)}")

    if not documents:
        return 0, 0

    chunks = split_documents(documents)

    print(f"Chunks: {len(chunks)}")

    if not chunks:
        return len(documents), 0

    if Path(CHROMA_PATH).exists():
        print("Existing Chroma found.")

        add_documents_to_db(chunks)

    else:
        print("Creating new Chroma.")

        create_vector_db(chunks)

    return len(documents), len(chunks)