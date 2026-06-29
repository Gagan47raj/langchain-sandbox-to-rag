import traceback

from langchain_ollama import ChatOllama

from app.loaders import load_documents
from app.splitter import split_documents
from app.embeddings import get_embedding_model
from app.vectordb import create_vector_db, load_vector_db
from app.retriever import get_retriever
from app.prompts import RAG_PROMPT

from app.config import MODEL_NAME


def format_docs(docs):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in docs
    )


def separator(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():

    try:

        separator("STEP 1 : Loading Documents")

        documents = load_documents("data")

        print(f"Loaded {len(documents)} documents")



        separator("STEP 2 : Splitting Documents")

        chunks = split_documents(documents)

        print(f"Created {len(chunks)} chunks")



        separator("STEP 3 : Loading Embedding Model")

        embedding_model = get_embedding_model()

        print(type(embedding_model))



        separator("STEP 4 : Creating / Loading ChromaDB")

        create_vector_db(chunks)

        vector_db = load_vector_db()

        print(vector_db)



        separator("STEP 5 : Creating Retriever")

        retriever = get_retriever(vector_db)

        print("Retriever Created Successfully")



        separator("STEP 6 : Retrieval Test")

        question = "What is MRI?"

        docs = retriever.invoke(question)

        print(f"Retrieved {len(docs)} chunks")



        separator("STEP 7 : Retrieved Context")

        context = format_docs(docs)

        print(context[:1000])



        separator("STEP 8 : Prompt Test")

        prompt = RAG_PROMPT.invoke(
            {
                "context": context,
                "question": question,
                "history": []      # remove if your prompt doesn't use history
            }
        )

        print(prompt)



        separator("STEP 9 : Loading Mistral")

        llm = ChatOllama(
            model=MODEL_NAME,
            temperature=0
        )



        separator("STEP 10 : LLM Response")

        response = llm.invoke(prompt)

        print(response.content)



        separator("ALL TESTS PASSED")

        print("Backend is working successfully!")

    except Exception as e:

        separator("TEST FAILED")

        print(e)

        traceback.print_exc()


if __name__ == "__main__":

    main()