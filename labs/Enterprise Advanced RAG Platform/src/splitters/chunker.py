from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

class EnterpriseChunker:

    def __init__(
        self,
        chunk_size=500,
        overlap=100,
        strategy="recursive"
    ):

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy

    def get_splitter(self):

        if self.strategy == "character":

            return CharacterTextSplitter(

                separator="\n",

                chunk_size=self.chunk_size,

                chunk_overlap=self.overlap
            )

        return RecursiveCharacterTextSplitter(

            chunk_size=self.chunk_size,

            chunk_overlap=self.overlap,

            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def split_documents(self, docs):

        splitter = self.get_splitter()

        chunks = splitter.split_documents(docs)

        for idx, chunk in enumerate(chunks):

            chunk.metadata["chunk_id"] = idx
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks