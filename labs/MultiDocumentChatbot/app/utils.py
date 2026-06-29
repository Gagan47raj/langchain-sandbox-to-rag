def format_docs(docs):

    if not docs:

        return ""

    return "\n\n".join(

        doc.page_content

        for doc in docs

    )