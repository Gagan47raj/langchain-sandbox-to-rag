from langchain_chroma import Chroma

from app.embeddings import get_embedding_model

from app.config import CHROMA_PATH

PERSIST_DIRECTORY = CHROMA_PATH


def create_vector_db(chunks):

    embedding_model = get_embedding_model()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )

    return vector_db

def load_vector_db():

    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )