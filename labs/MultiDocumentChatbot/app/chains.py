from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from app import settings
from app.prompts import RAG_PROMPT_WITH_HISTORY

from app.prompts import RAG_PROMPT
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

        model=settings.model,

        temperature=settings.temperature,

        num_predict=settings.max_tokens

    )

    chain = (
        {
            "context": (
                RunnableLambda(lambda x: x["question"])
                | retriever
                | RunnableLambda(format_docs)
            ),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def build_chatbot(settings):
    vector_db = load_vector_db()

    retriever = vector_db.as_retriever(

        search_type=settings.search_type,

        search_kwargs={

            "k": settings.top_k

        }
    )

    llm = ChatOllama(

        model=settings.model,

        temperature=settings.temperature,

        num_predict=settings.max_tokens

    )

    chain = (
        {
            "context": (
                RunnableLambda(lambda x: x["question"])
                | retriever
                | RunnableLambda(format_docs)
            ),
            "question": RunnableLambda(lambda x: x["question"]),
            "history": RunnableLambda(lambda x: x.get("history", [])),
        }
        | RAG_PROMPT_WITH_HISTORY
        | llm
        | StrOutputParser()
    )

    return chain