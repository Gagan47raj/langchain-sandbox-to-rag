from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MODEL_NAME = "mistral"

EMBEDDING_MODEL = "nomic-embed-text"

CHROMA_PATH = "chroma_db"

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

TOP_K = 4

UPLOAD_FOLDER = DATA_DIR / "uploads"

