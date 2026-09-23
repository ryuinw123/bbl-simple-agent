from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE


def _load_knowledge_base_documents() -> list[Document]:
    """Load the packaged Hotel Californian knowledge base."""
    file_path = Path(__file__).resolve().with_name("knowledge_base.txt")
    return TextLoader(str(file_path), encoding="utf-8").load()


def _split_documents(docs_list: list[Document]) -> list[Document]:
    """Split source documents into overlapping chunks for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )
    return text_splitter.split_documents(docs_list)


def load_documents_from_knowledge_base() -> list[Document]:
    """Load and split the knowledge base into retrievable documents."""
    docs_list = _load_knowledge_base_documents()
    return _split_documents(docs_list)
