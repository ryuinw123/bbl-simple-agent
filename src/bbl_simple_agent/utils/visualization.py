from pathlib import Path


def save_mermaid_diagram(
    graph,
    output_path: str = "artifacts/graph/graph_setup.png",
) -> None:
    """Save the workflow graph as Mermaid source and a PNG image.

    Args:
        graph: The compiled LangGraph to visualize.
        output_path: Path for the generated PNG image.
    """
    output_file = Path(output_path)
    mermaid_file = output_file.with_suffix(".mmd")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Render the outer workflow. Expanding nested agent graphs with xray=True
    # can produce escaped node identifiers that mermaid.ink rejects.
    graph_view = graph.get_graph()
    mermaid_file.write_text(graph_view.draw_mermaid(), encoding="utf-8")

    try:
        graph_png = graph_view.draw_mermaid_png(
            max_retries=5,
            retry_delay=2.0,
        )
        output_file.write_bytes(graph_png)
    except Exception as error:
        print(
            f"Could not generate the PNG diagram: {error}\n"
            f"Mermaid source was saved to {mermaid_file}."
        )
