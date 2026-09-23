from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from functools import lru_cache
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from ..config import EMBEDDING_MODEL, RAG_TOP_K
from ..rag.process_txt import load_documents_from_knowledge_base


@lru_cache(maxsize=1)
def _get_retriever() -> VectorStoreRetriever:
    """Build and cache the in-memory semantic retriever."""
    doc_splits = load_documents_from_knowledge_base()
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        ),
    )
    return vectorstore.as_retriever(
        search_kwargs={"k": RAG_TOP_K},
    )


@tool
def retrieve_hotel_data(query: str) -> str:
    """Search the Hotel Californian knowledge base for relevant information."""
    retriever = _get_retriever()
    retrieved_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])


@tool
def transfer_to_report_generator(
    runtime: ToolRuntime,
) -> Command:
    """Transfer retrieved context to the Report Generator agent."""
    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content="Transferred to report generator agent from data retriever agent",
        tool_call_id=runtime.tool_call_id,
    )
    messages = runtime.state["messages"]
    retrieval_message = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, ToolMessage)
            and message.name == "retrieve_hotel_data"
            and message.status == "success"
            and message.content
        ),
        None,
    )

    return Command(
        goto="report_generator_agent",
        update={
            "messages": [last_ai_message, transfer_message],
            "retrieved_context": retrieval_message.content,
        },
        graph=Command.PARENT,
    )
