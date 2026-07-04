from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader
)

from pathlib import Path

def load_documents(data_dir):
    
    documents = []

    data_dir = Path(data_dir)

    # PDFs
    for pdf_file in data_dir.glob("*.pdf"):

        loader = PyPDFLoader(str(pdf_file))

        documents.extend(loader.load())

    # TXT
    for txt_file in data_dir.glob("*.txt"):

        loader = TextLoader(
            str(txt_file),
            encoding="utf-8"
        )

        documents.extend(loader.load())

    # CSV
    for csv_file in data_dir.glob("*.csv"):

        loader = CSVLoader(
            file_path=str(csv_file)
        )

        documents.extend(loader.load())

    return documents