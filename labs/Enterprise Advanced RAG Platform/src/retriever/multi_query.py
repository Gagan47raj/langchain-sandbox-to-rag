import logging

from langchain_ollama import ChatOllama

from langchain_classic.retrievers import MultiQueryRetriever

class EnterpriseMultiQuery:

    def __init__(

        self,

        retriever,

        model="mistral",

        temperature=0

    ):

        logging.basicConfig()

        logging.getLogger(

            "langchain.retrievers.multi_query"

        ).setLevel(logging.INFO)

        self.llm = ChatOllama(

            model=model,

            temperature=temperature

        )

        self.retriever = MultiQueryRetriever.from_llm(

            retriever=retriever,

            llm=self.llm

        )

    def search(self, query):

        return self.retriever.invoke(query)