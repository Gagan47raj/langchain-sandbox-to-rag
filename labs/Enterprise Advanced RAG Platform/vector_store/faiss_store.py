from pathlib import Path

from langchain_community.vectorstores import FAISS


class EnterpriseFAISS:

    def __init__(

        self,

        embedding_model,

        save_path="../vector_store/faiss_index"

    ):

        self.embedding_model = embedding_model

        self.save_path = Path(save_path)

        self.db = None

    def create(self, documents):

        self.db = FAISS.from_documents(

            documents,

            self.embedding_model.model

        )

        return self.db

    def save(self):

        self.db.save_local(

            str(self.save_path)

        )

    def load(self):

        self.db = FAISS.load_local(

            str(self.save_path),

            self.embedding_model.model,

            allow_dangerous_deserialization=True

        )

        return self.db

    def search(self, query, k=5):

        return self.db.similarity_search(

            query,

            k=k

        )

    def retriever(self, k=5):

        return self.db.as_retriever(

            search_kwargs={"k":k}

        )