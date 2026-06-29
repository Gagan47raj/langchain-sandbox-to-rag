def get_retriever(

    vector_db,

    k=4

):

    return vector_db.as_retriever(

        search_type="similarity",

        search_kwargs={

            "k":k

        }

    )

