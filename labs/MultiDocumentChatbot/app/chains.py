from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

from app.prompts import RAG_PROMPT, RAG_PROMPT_WITH_HISTORY
from app.retriever import get_retriever
from app.vectordb import load_vector_db
from app.config import MODEL_NAME


def format_docs(docs):

    return "\n\n".join(

        f"Source: {doc.metadata.get('source')}, "
        f"Page: {doc.metadata.get('page', 'N/A')}\n"
        f"{doc.page_content}"

        for doc in docs
    )

def build_chain():

    vector_db = load_vector_db()

    retriever = get_retriever(vector_db)

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    chain = (

        {
            "context": retriever | RunnableLambda(format_docs),

            "question": RunnablePassthrough(),

        }

        | RAG_PROMPT

        | llm

        | StrOutputParser()

    )

    return chain

def build_chatbot():

    vector_db = load_vector_db()

    retriever = get_retriever(vector_db)

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    chain = (

        {
            "context": retriever | RunnableLambda(format_docs),

            "question": RunnablePassthrough(),

        }

        | RAG_PROMPT_WITH_HISTORY

        | llm

        | StrOutputParser()

    )

    return chain