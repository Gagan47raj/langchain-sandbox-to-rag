from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingManager:

    def __init__(

        self,

        provider="ollama",

        model_name="nomic-embed-text"

    ):

        self.provider = provider

        self.model_name = model_name

        self.model = self.load_model()

    def load_model(self):

        if self.provider == "ollama":

            return OllamaEmbeddings(

                model=self.model_name

            )

        elif self.provider == "huggingface":

            return HuggingFaceEmbeddings(

                model_name=self.model_name

            )

        raise ValueError("Unsupported Provider")

    def embed_query(self, query):

        return self.model.embed_query(query)

    def embed_documents(self, docs):

        return self.model.embed_documents(docs)