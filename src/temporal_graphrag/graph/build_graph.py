
import json
from pathlib import Path

import networkx as nx

from temporal_graphrag.schemas.causal_chain import (
    CausalChain,
    validate_chain_references,
)


ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    ROOT / "data" / "processed" / "set_01000_llm_causal_chain.json"
)
OUTPUT_PATH = (
    ROOT / "data" / "processed" / "set_01000_temporal_graph.json"
)


def build_graph(chain: CausalChain) -> nx.MultiDiGraph:
    """Build a directed graph from extracted events and relations."""

    validate_chain_references(chain)

    graph = nx.MultiDiGraph(
        scenario_id=chain.scenario_id,
        graph_type="temporal_causal_decision_graph",
    )

    # Each extracted decision event becomes a node.
    for event in chain.events:
        graph.add_node(
            event.event_id,
            scenario_id=event.scenario_id,
            meeting_id=event.meeting_id,
            event_type=event.event_type.value,
            description=event.description,
            actor=event.actor,
            decision_option=event.decision_option,
            source_utterance_ids=event.source_utterance_ids,
            confidence=event.confidence,
        )

    # Each relation becomes a directed edge.
    for relation in chain.relations:
        graph.add_edge(
            relation.source_event_id,
            relation.target_event_id,
            relation_type=relation.relation_type.value,
            explanation=relation.explanation,
            source_utterance_ids=relation.source_utterance_ids,
        )

    return graph


def export_graph(graph: nx.MultiDiGraph, output_path: Path) -> None:
    """Export nodes and edges to JSON without losing graph attributes."""

    data = {
        "scenario_id": graph.graph["scenario_id"],
        "graph_type": graph.graph["graph_type"],
        "nodes": [
            {"event_id": node_id, **attributes}
            for node_id, attributes in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source_event_id": source,
                "target_event_id": target,
                **attributes,
            }
            for source, target, attributes in graph.edges(data=True)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    chain = CausalChain.model_validate(raw_data)
    graph = build_graph(chain)
    export_graph(graph, OUTPUT_PATH)

    print("Temporal decision graph built successfully.")
    print(f"Scenario: {chain.scenario_id}")
    print(f"Nodes (events): {graph.number_of_nodes()}")
    print(f"Edges (relations): {graph.number_of_edges()}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nEvent types:")
    for event_type in sorted({
        attributes["event_type"]
        for _, attributes in graph.nodes(data=True)
    }):
        count = sum(
            attributes["event_type"] == event_type
            for _, attributes in graph.nodes(data=True)
        )
        print(f"  {event_type}: {count}")

    print("\nRelation types:")
    for relation_type in sorted({
        attributes["relation_type"]
        for _, _, attributes in graph.edges(data=True)
    }):
        count = sum(
            attributes["relation_type"] == relation_type
            for _, _, attributes in graph.edges(data=True)
        )
        print(f"  {relation_type}: {count}")


if __name__ == "__main__":
    main()
