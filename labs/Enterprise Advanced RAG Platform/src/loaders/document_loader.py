from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)


class EnterpriseDocumentLoader:

    def __init__(self):

        self.loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
            ".csv": CSVLoader,
            ".md": UnstructuredMarkdownLoader,
        }

    def load_file(self, file_path):

        file_path = Path(file_path)

        suffix = file_path.suffix.lower()

        if suffix not in self.loaders:
            print(f"Unsupported File : {file_path.name}")
            return []

        loader = self.loaders[suffix](str(file_path))

        docs = loader.load()

        for doc in docs:

            doc.metadata["filename"] = file_path.name
            doc.metadata["extension"] = suffix
            doc.metadata["domain"] = file_path.parent.name

        return docs

    def load_directory(self, directory):

        directory = Path(directory)

        all_documents = []

        for file in directory.rglob("*"):

            if file.suffix.lower() in self.loaders:

                docs = self.load_file(file)

                all_documents.extend(docs)

        return all_documents