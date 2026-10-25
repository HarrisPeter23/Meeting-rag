
import argparse
import json
import re
from collections import deque
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[3]
GRAPH_PATH = ROOT / "data" / "processed" / "set_01000_temporal_graph.json"

STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "to", "of", "in", "on",
    "and", "or", "for", "with", "did", "do", "does", "what", "why",
    "how", "which", "who", "about", "by", "from", "it", "they",
    "he", "she", "their", "this", "that", "decision", "event",
}


def tokenize(text: str) -> set[str]:
    """Convert text into lowercase keywords."""
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 1}


def intent_bonus(question: str, attributes: dict) -> float:
    """Prefer event types that match the question's intent."""
    q = question.lower()
    event_type = attributes.get("event_type", "")

    if any(term in q for term in ("reopen", "reopened", "reconsider")):
        if event_type == "decision_reopened":
            return 2.0
        if event_type in {"objection", "evidence"}:
            return 1.0

    if any(term in q for term in ("final decision", "finally decide", "finalize")):
        if event_type == "final_decision":
            return 2.0

    if any(term in q for term in ("what evidence", "which evidence", "what data")):
        if event_type == "evidence":
            return 1.5
        if event_type == "support":
            return 0.5

    return 0.0

def load_graph(path: Path = GRAPH_PATH) -> nx.MultiDiGraph:
    """Load the exported graph JSON into a directed graph."""
    data = json.loads(path.read_text(encoding="utf-8"))

    graph = nx.MultiDiGraph(
        scenario_id=data["scenario_id"],
        graph_type=data["graph_type"],
    )

    for node in data["nodes"]:
        event_id = node["event_id"]
        graph.add_node(event_id, **{
            key: value for key, value in node.items()
            if key != "event_id"
        })

    for edge in data["edges"]:
        graph.add_edge(
            edge["source_event_id"],
            edge["target_event_id"],
            relation_type=edge["relation_type"],
            explanation=edge.get("explanation", ""),
            source_utterance_ids=edge.get("source_utterance_ids", []),
        )

    return graph


def score_event(query_tokens: set[str], attributes: dict) -> float:
    """Score an event by keyword overlap with its metadata."""
    searchable_text = " ".join([
        attributes.get("description", ""),
        attributes.get("actor") or "",
        attributes.get("event_type", ""),
        attributes.get("decision_option") or "",
    ])

    event_tokens = tokenize(searchable_text)
    if not query_tokens:
        return 0.0

    overlap = query_tokens & event_tokens
    return len(overlap) / len(query_tokens)


def retrieve_lineage(
    graph: nx.MultiDiGraph,
    question: str,
    top_k: int = 3,
    hops: int = 1,
) -> dict:
    """
    Retrieve matching events and nearby causal relations.

    Traversal follows edges in either direction to discover context,
    while preserving the original direction of every returned relation.
    """
    query_tokens = tokenize(question)

    scored_events = [
        (
        node_id,
        score_event(query_tokens, attrs)
        + intent_bonus(question, attrs),
        )
        for node_id, attrs in graph.nodes(data=True)
    ]
    scored_events.sort(key=lambda item: (-item[1], item[0]))

    seeds = [
        node_id for node_id, score in scored_events[:top_k]
        if score > 0
    ]

    if not seeds:
        return {
            "question": question,
            "matched_event_ids": [],
            "events": [],
            "relations": [],
            "message": "No relevant events found. Try rephrasing the question.",
        }

    # Expand from matched events using an undirected view of connectivity.
    
    # Use the highest-ranked event as the retrieval anchor.
    # Expand from that event so unrelated seed matches do not pull
    # entire neighboring parts of the graph into the result.
    anchor = seeds[0]
    selected = {anchor}
    queue = deque([(anchor, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth >= hops:
            continue

        neighbors = set(graph.predecessors(current))
        neighbors.update(graph.successors(current))

        # Prefer neighbors with stronger lexical relevance.
        ranked_neighbors = sorted(
            neighbors - selected,
            key=lambda node_id: (
                -(
                    score_event(query_tokens, graph.nodes[node_id])
                    + intent_bonus(question, graph.nodes[node_id])
                ),
                node_id,
            ),
        )

        for neighbor in ranked_neighbors:
            selected.add(neighbor)
            queue.append((neighbor, depth + 1))

    # Only report seed matches that are present in the retrieved context.
    retrieved_seeds = [node_id for node_id in seeds if node_id in selected]

    events = []
    for node_id in graph.nodes:
        if node_id in selected:
            events.append({
                "event_id": node_id,
                **graph.nodes[node_id],
            })

    relations = []
    for source, target, attributes in graph.edges(data=True):
        if source in selected and target in selected:
            relations.append({
                "source_event_id": source,
                "target_event_id": target,
                **attributes,
            })

    return {
        "question": question,
        "matched_event_ids": retrieved_seeds,
        "events": events,
        "relations": relations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve decision lineage from a temporal decision graph."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to search for in the decision graph.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--hops", type=int, default=1)
    args = parser.parse_args()

    if args.top_k < 1 or args.hops < 0:
        parser.error("--top-k must be at least 1 and --hops cannot be negative")

    graph = load_graph()
    result = retrieve_lineage(
        graph,
        question=args.question,
        top_k=args.top_k,
        hops=args.hops,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
