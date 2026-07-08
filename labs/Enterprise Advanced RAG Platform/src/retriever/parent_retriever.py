import faiss

from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

class EnterpriseParentRetriever:

    def __init__(
        self,
        embedding_model,
        parent_size=2000,
        child_size=400,
        overlap=80
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=200
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=overlap
        )

        # Create empty FAISS index
        embedding = embedding_model.model.embed_query("test")
        index = faiss.IndexFlatL2(len(embedding))

        self.vectorstore = FAISS(
            embedding_function=embedding_model.model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        self.store = InMemoryStore()

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter
        )

    def add_documents(self, docs):

        self.retriever.add_documents(docs)

    def search(self, query):

        return self.retriever.invoke(query)