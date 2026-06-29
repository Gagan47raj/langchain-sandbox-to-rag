from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

SYSTEM_PROMPT = """
You are an AI assistant.

Answer ONLY using the provided context.

If the answer is not available, say:

"I couldn't find that information in the uploaded documents."
"""

# -------------------
# No Memory Prompt
# -------------------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),

        ("human",
"""
Context:
{context}

Question:
{question}
""")
    ]
)

# -------------------
# Memory Prompt
# -------------------

RAG_PROMPT_WITH_HISTORY = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),

        MessagesPlaceholder("history"),

        ("human",
"""
Context:
{context}

Question:
{question}
""")
    ]
)