"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv


load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "openai:gpt-5-nano-2025-08-07")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

if RAG_TOP_K < 1:
    raise ValueError("RAG_TOP_K must be at least 1")
if RAG_CHUNK_SIZE < 1:
    raise ValueError("RAG_CHUNK_SIZE must be at least 1")
if not 0 <= RAG_CHUNK_OVERLAP < RAG_CHUNK_SIZE:
    raise ValueError(
        "RAG_CHUNK_OVERLAP must be at least 0 and less than RAG_CHUNK_SIZE"
    )
