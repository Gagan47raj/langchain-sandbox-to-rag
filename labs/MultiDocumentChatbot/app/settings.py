from dataclasses import dataclass


@dataclass
class ChatSettings:

    model: str = "mistral"

    temperature: float = 0.0

    top_k: int = 4

    search_type: str = "similarity"

    max_tokens: int = 2048