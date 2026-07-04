from app import settings


def get_retriever(

    vector_db,

    k=4

):

    return vector_db.as_retriever(

        search_type=settings.search_type,

        search_kwargs={

            "k":settings.top_k

        }

    )

