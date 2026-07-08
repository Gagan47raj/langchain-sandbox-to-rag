from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker
)

from langchain_classic.retrievers import (
    ContextualCompressionRetriever
)

from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class EnterpriseReranker:

    def __init__(

        self,

        retriever,

        model_name="BAAI/bge-reranker-base",

        top_n=5

    ):

        self.cross_encoder = HuggingFaceCrossEncoder(

            model_name=model_name

        )

        self.reranker = CrossEncoderReranker(

            model=self.cross_encoder,

            top_n=top_n

        )

        self.retriever = ContextualCompressionRetriever(

            base_retriever=retriever,

            base_compressor=self.reranker

        )

    def search(self, query):

        return self.retriever.invoke(query)