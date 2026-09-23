from ..config import CHAT_MODEL
from .state import ContextAgentState
from .tools import retrieve_hotel_data, transfer_to_report_generator
from langchain_core.messages import SystemMessage, HumanMessage
from .prompts import DATA_RETRIEVER_PROMPT, REPORT_GENERATOR_PROMPT
from langchain.agents import create_agent
from langgraph.types import Command
from langchain.agents.middleware import ToolCallLimitMiddleware


# ============================================================
# Agent Creation
# ============================================================
# Create agents with domain-specific tools using create_agent
# Each agent is a compiled subgraph that can invoke tools during reasoning

data_retriever_agent = create_agent(
    model=CHAT_MODEL,
    tools=[retrieve_hotel_data, transfer_to_report_generator],
    system_prompt=DATA_RETRIEVER_PROMPT,
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="retrieve_hotel_data",
            run_limit=1,
            exit_behavior="continue",
        ),
    ],
)

report_generator_agent = create_agent(
    model=CHAT_MODEL,
    tools=[],
)

# ============================================================
# Helper Functions
# ============================================================


def _invoke_agent(agent, prompt: str, messages: list, agent_name: str):
    """Helper to invoke an agent and return formatted response.

    This consolidates the common pattern of:
    1. Adding system prompt to messages
    2. Invoking the agent subgraph
    3. Extracting and naming the response message
    """
    agent_input = {"messages": [SystemMessage(content=prompt)] + messages}
    agent_result = agent.invoke(agent_input)
    response_message = agent_result["messages"][-1]
    response_message.name = agent_name
    return response_message

# ============================================================
# Node Functions
# ============================================================


def call_report_generator_agent(state: ContextAgentState) -> Command:
    """Generate the final grounded response and end the workflow."""
    retrieved_context = state.get(
        "retrieved_context", "No relevant context was retrieved.").strip()
    question = next(
        message.content
        for message in reversed(state["messages"])
        if isinstance(message, HumanMessage)
    )
    prompt = REPORT_GENERATOR_PROMPT.format(
        question=question, context=retrieved_context)
    response = _invoke_agent(report_generator_agent,
                             prompt, [], "report_generator_agent")
    return Command(
        goto="__end__",
        update={"messages": [response]}
    )


def call_data_retriever_agent(state: ContextAgentState) -> Command:
    """Retrieve relevant hotel context and hand it to the Report Generator."""
    _ = data_retriever_agent.invoke(state)

    # If no handoff occurs, continue without retrieved context so the Report
    # Generator can return its configured insufficient-information response.
    return Command(
        goto="report_generator_agent"
    )
