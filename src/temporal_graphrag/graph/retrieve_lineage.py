
import argparse
import json
import re
from collections import deque
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRAPH_PATH = ROOT / "data" / "processed" / "set_01000_temporal_graph.json"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by",
    "did", "do", "does", "for", "from", "how", "in", "into", "is",
    "it", "of", "on", "or", "that", "the", "their", "them", "then",
    "there", "these", "this", "to", "was", "were", "what", "when",
    "which", "who", "why", "with", "would",
}


def tokenize(text: str) -> set[str]:
    """Normalize text into a set of searchable tokens."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS
    }


def intent_bonus(question: str, event: dict) -> float:
    """Boost events whose type matches the question's intent."""
    question_lower = question.lower()
    event_type = str(event.get("event_type", "")).lower()
    description = str(event.get("description", "")).lower()

    bonus = 0.0

    asks_final_decision = any(
        phrase in question_lower
        for phrase in (
            "final decision",
            "ultimately decide",
            "final outcome",
            "what was decided",
            "what did they decide",
        )
    )

    asks_reconsideration = any(
        phrase in question_lower
        for phrase in (
            "reopen",
            "reconsider",
            "change the decision",
            "decision change",
            "why did the decision change",
        )
    )

    asks_for_evidence = any(
        phrase in question_lower
        for phrase in (
            "evidence",
            "data",
            "numbers",
            "metrics",
            "why did",
            "why was",
            "reason",
            "justify",
        )
    )

    if asks_final_decision and event_type == "final_decision":
        bonus += 0.35

    if asks_reconsideration and (
        "reopen" in event_type
        or "reopen" in description
        or "pending" in description
    ):
        bonus += 0.25

    if asks_for_evidence and event_type in {
        "evidence",
        "objection",
        "decision_reopened",
    }:
        bonus += 0.15

    return bonus


def score_event(query_tokens: set[str], event: dict) -> float:
    """Calculate lexical relevance between the question and an event."""
    searchable_fields = (
        "event_id",
        "event_type",
        "description",
        "summary",
        "topic",
        "speaker",
        "participants",
        "decision",
        "content",
    )

    text_parts = []

    for field in searchable_fields:
        value = event.get(field)

        if isinstance(value, list):
            text_parts.extend(str(item) for item in value)
        elif value is not None:
            text_parts.append(str(value))

    event_tokens = tokenize(" ".join(text_parts))

    if not query_tokens:
        return 0.0

    overlap = query_tokens.intersection(event_tokens)
    return len(overlap) / len(query_tokens)


def event_relevance(
    question: str,
    query_tokens: set[str],
    event: dict,
) -> float:
    """Combine lexical relevance and question-intent relevance."""
    return (
        score_event(query_tokens, event)
        + intent_bonus(question, event)
    )


def get_event_type(graph: nx.MultiDiGraph, node_id: str) -> str:
    """Return the normalized event type for a graph node."""
    return str(
        graph.nodes[node_id].get("event_type", "")
    ).lower()


def traversal_neighbors(
    graph: nx.MultiDiGraph,
    node_id: str,
) -> set[str]:
    """Return incoming and outgoing neighbors for a directed graph."""
    return set(graph.predecessors(node_id)) | set(graph.successors(node_id))


def expand_from_seeds(
    graph: nx.MultiDiGraph,
    seeds: list[str],
    hops: int,
) -> set[str]:
    """Collect seed nodes and their neighbors up to the hop limit."""
    selected = set(seeds)
    queue = deque((node_id, 0) for node_id in seeds)
    visited_depth = {node_id: 0 for node_id in seeds}

    while queue:
        current, depth = queue.popleft()

        if depth >= hops:
            continue

        neighbors = sorted(traversal_neighbors(graph, current))

        for neighbor in neighbors:
            next_depth = depth + 1

            if (
                neighbor not in visited_depth
                or next_depth < visited_depth[neighbor]
            ):
                visited_depth[neighbor] = next_depth
                selected.add(neighbor)
                queue.append((neighbor, next_depth))

    return selected


def decision_lineage_priority(
    graph: nx.MultiDiGraph,
    node_id: str,
) -> float:
    """
    Give important decision-lineage events a small selection bonus.

    This is a ranking heuristic, not a claim that a relation is causal.
    """
    event_type = get_event_type(graph, node_id)
    bonus = 0.0

    if event_type == "final_decision":
        bonus += 0.30
    elif event_type in {"decision_reopened", "reopened_decision"}:
        bonus += 0.25
    elif event_type in {"evidence", "objection"}:
        bonus += 0.12

    return bonus


def rank_candidates(
    graph: nx.MultiDiGraph,
    question: str,
    query_tokens: set[str],
    candidates: set[str],
) -> list[str]:
    """Rank candidate events using relevance and event-type priority."""
    return sorted(
        candidates,
        key=lambda node_id: (
            -(
                event_relevance(
                    question,
                    query_tokens,
                    graph.nodes[node_id],
                )
                + decision_lineage_priority(graph, node_id)
            ),
            node_id,
        ),
    )


def preserve_decision_lineage(
    graph: nx.MultiDiGraph,
    ranked_candidates: list[str],
    seeds: list[str],
    max_events: int,
) -> list[str]:
    """
    Select a bounded set of events while prioritizing decision lineage.

    Start with the highest-ranked candidates, then prefer paths connected
    to final-decision events. Remaining slots are filled by relevance.
    """
    if len(ranked_candidates) <= max_events:
        return ranked_candidates

    selected = []
    selected_set = set()

    def add(node_id: str) -> None:
        if node_id in selected_set or len(selected) >= max_events:
            return
        selected.append(node_id)
        selected_set.add(node_id)

    # Keep the most relevant matched seeds first.
    for seed in seeds:
        if seed in ranked_candidates:
            add(seed)

    # Identify final-decision events among the candidates.
    final_nodes = [
        node_id
        for node_id in ranked_candidates
        if get_event_type(graph, node_id) == "final_decision"
    ]

    # Prefer direct incoming lineage to final decisions, followed by
    # incoming lineage of those supporting events.
    for final_node in final_nodes:
        add(final_node)

        direct_predecessors = sorted(
            graph.predecessors(final_node),
            key=lambda node_id: (
                ranked_candidates.index(node_id)
                if node_id in ranked_candidates
                else len(ranked_candidates),
                node_id,
            ),
        )

        for predecessor in direct_predecessors:
            if predecessor in ranked_candidates:
                add(predecessor)

            for earlier in sorted(
                graph.predecessors(predecessor),
                key=lambda node_id: (
                    ranked_candidates.index(node_id)
                    if node_id in ranked_candidates
                    else len(ranked_candidates),
                    node_id,
                ),
            ):
                if earlier in ranked_candidates:
                    add(earlier)

    # Fill any remaining slots by overall ranking.
    for node_id in ranked_candidates:
        add(node_id)

    return selected


def retrieve_lineage(
    graph: nx.MultiDiGraph,
    question: str,
    top_k: int = 3,
    hops: int = 1,
    max_events: int = 10,
) -> dict:
    """
    Retrieve relevant events and relations for a question.

    Args:
        graph: Directed multigraph containing event nodes and relations.
        question: Natural-language question.
        top_k: Number of initial relevance seeds.
        hops: Maximum graph distance expanded from the seeds.
        max_events: Maximum number of events in the returned context.

    Returns:
        Dictionary containing matched IDs, events, and relations.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if hops < 0:
        raise ValueError("hops cannot be negative.")

    if max_events < 1:
        raise ValueError("max_events must be at least 1.")

    query_tokens = tokenize(question)

    scored_nodes = []

    for node_id, attributes in graph.nodes(data=True):
        score = event_relevance(question, query_tokens, attributes)
        scored_nodes.append((node_id, score))

    # Select the strongest lexical/intent matches as traversal seeds.
    scored_nodes.sort(key=lambda item: (-item[1], item[0]))

    seeds = [
        node_id
        for node_id, score in scored_nodes[:top_k]
        if score > 0
    ]

    # Fallback when the question has no lexical overlap with the graph.
    if not seeds and scored_nodes:
        seeds = [scored_nodes[0][0]]

    selected = expand_from_seeds(
        graph=graph,
        seeds=seeds,
        hops=hops,
    )

    ranked_candidates = rank_candidates(
        graph=graph,
        question=question,
        query_tokens=query_tokens,
        candidates=selected,
    )

    selected_order = preserve_decision_lineage(
        graph=graph,
        ranked_candidates=ranked_candidates,
        seeds=seeds,
        max_events=max_events,
    )

    selected_set = set(selected_order)

    events = []
    for node_id in selected_order:
        events.append({
            "event_id": node_id,
            **graph.nodes[node_id],
        })

    relations = []

    for source, target, key, attributes in graph.edges(
        keys=True,
        data=True,
    ):
        if source not in selected_set or target not in selected_set:
            continue

        relation = {
            "source_event_id": source,
            "target_event_id": target,
            **attributes,
        }

        # Include the edge key only if the relation has no explicit ID.
        if "relation_id" not in relation:
            relation["edge_key"] = key

        relations.append(relation)

    relations.sort(
        key=lambda relation: (
            relation["source_event_id"],
            relation["target_event_id"],
            str(relation.get("relation_type", "")),
        )
    )

    matched_event_ids = [
        seed for seed in seeds if seed in selected_set
    ]

    return {
        "question": question,
        "matched_event_ids": matched_event_ids,
        "events": events,
        "relations": relations,
        "metadata": {
            "top_k": top_k,
            "hops": hops,
            "max_events": max_events,
            "returned_event_count": len(events),
            "returned_relation_count": len(relations),
        },
    }



def load_graph(graph_path: Path) -> nx.MultiDiGraph:
    """Load the custom exported temporal graph JSON."""
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    with graph_path.open("r", encoding="utf-8") as file:
        graph_data = json.load(file)

    graph = nx.MultiDiGraph()

    for node in graph_data.get("nodes", []):
        node_data = node.copy()
        node_id = node_data.pop("event_id", node_data.pop("id", None))

        if node_id is None:
            raise ValueError(f"Graph node is missing an event ID: {node}")

        graph.add_node(str(node_id), **node_data)

    edges = graph_data.get("links", graph_data.get("edges", []))

    for edge in edges:
        edge_data = edge.copy()
        source = edge_data.pop("source_event_id", None)
        target = edge_data.pop("target_event_id", None)

        if source is None or target is None:
            raise ValueError(f"Graph edge is missing source or target: {edge}")

        graph.add_edge(str(source), str(target), **edge_data)

    return graph



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve temporal decision lineage from a graph."
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Question to answer using the graph.",
    )

    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH_PATH,
        help="Path to the exported temporal graph JSON.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of initial event matches to use as seeds.",
    )

    parser.add_argument(
        "--hops",
        type=int,
        default=1,
        help="Maximum graph distance to expand from the seeds.",
    )

    parser.add_argument(
        "--max-events",
        type=int,
        default=10,
        help="Maximum number of events to return.",
    )

    args = parser.parse_args()

    if args.top_k < 1 or args.max_events < 1:
        parser.error(
            "--top-k and --max-events must be at least 1."
        )

    if args.hops < 0:
        parser.error("--hops cannot be negative.")

    try:
        graph = load_graph(args.graph)

        result = retrieve_lineage(
            graph=graph,
            question=args.question,
            top_k=args.top_k,
            hops=args.hops,
            max_events=args.max_events,
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
