from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


class EnterpriseHybridRetriever:

    def __init__(

        self,

        documents,

        vector_retriever,

        bm25_weight=0.4,

        vector_weight=0.6,

        k=5

    ):

        self.documents = documents

        self.vector_retriever = vector_retriever

        self.k = k

        self.bm25 = BM25Retriever.from_documents(
            documents
        )

        self.bm25.k = k

        self.hybrid = EnsembleRetriever(

            retrievers=[
                self.bm25,
                self.vector_retriever
            ],

            weights=[
                bm25_weight,
                vector_weight
            ]
        )

    def search(self, query):

        return self.hybrid.invoke(query)