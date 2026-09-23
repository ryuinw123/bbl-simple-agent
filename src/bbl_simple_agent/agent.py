import argparse
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from .utils.nodes import call_data_retriever_agent, call_report_generator_agent
from .utils.state import ContextAgentState
from .utils.visualization import save_mermaid_diagram


def build_graph() -> CompiledStateGraph:
    """Build the Data Retriever-to-Report Generator workflow."""
    builder = StateGraph(ContextAgentState)

    builder.add_node(
        "data_retriever_agent",
        call_data_retriever_agent,
        destinations=("report_generator_agent",),
    )
    builder.add_node(
        "report_generator_agent",
        call_report_generator_agent,
        destinations=(END,),
    )

    builder.add_edge(START, "data_retriever_agent")

    return builder.compile()


graph = build_graph()


def main() -> None:
    """Run one command-line query through the compiled workflow."""
    # save_mermaid_diagram(graph)

    # Accept an optional query while keeping a useful default for local testing.
    parser = argparse.ArgumentParser(description="Command-based routing")
    parser.add_argument(
        "query", nargs="?", default="Hi, I'm having trouble with my account login. Can you help?"
    )
    args = parser.parse_args()

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": args.query,
                }
            ]
        }
    )

    result["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
