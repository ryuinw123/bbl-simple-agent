from langchain.agents import AgentState
from typing_extensions import NotRequired

# ============================================================
# State
# ============================================================


class ContextAgentState(AgentState):
    retrieved_context: NotRequired[str]
