from langchain_chroma import Chroma

from app.embeddings import get_embedding_model

from app.config import CHROMA_PATH

PERSIST_DIRECTORY = CHROMA_PATH


def create_vector_db(chunks):
    if not chunks:
        raise ValueError("No chunks were generated from the documents.")

    embedding_model = get_embedding_model()

    # Check first chunk
    print("First chunk preview:")
    print(repr(chunks[0].page_content[:200]))

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
    )

    return vector_db

def load_vector_db():

    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )

def add_documents_to_db(chunks):
    """
    Add new chunks to an existing Chroma database.
    """

    if not chunks:
        return

    vector_db = load_vector_db()

    vector_db.add_documents(chunks)

    return vector_db