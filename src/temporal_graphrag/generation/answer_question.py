import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from temporal_graphrag.graph.retrieve_lineage import (
    load_graph,
    retrieve_lineage,
)

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_GRAPH_PATH = (
    ROOT / "data" / "processed" / "set_01000_temporal_graph.json"
)

load_dotenv(ROOT / ".env")


def build_prompt(question: str, context: dict) -> str:
    """Build a grounded answer-generation prompt from retrieved evidence."""
    evidence = json.dumps(
        {
            "events": context["events"],
            "relations": context["relations"],
        },
        indent=2,
        ensure_ascii=False,
    )

    return f"""
You are an evidence-grounded assistant for temporal meeting intelligence.

Answer the user's question using ONLY the retrieved events and relations
provided below.

USER QUESTION:
{question}

RETRIEVED EVENTS AND RELATIONS:
{evidence}

INSTRUCTIONS:
1. Explain the causal sequence that answers the question, not just a list
   of meeting summaries.
2. Distinguish earlier decisions from later decisions. A later decision
   supersedes an earlier one only when the retrieved evidence supports it.
3. Use explicit relations to explain causal connections. Do not assume that
   temporal order alone proves causation.
4. Cite evidence using event IDs, for example [e9], and include source
   utterance IDs when available, for example [e9; utterances: m2_u0010].
5. Attribute statements to the correct actors and preserve the meaning of
   the evidence.
6. Do not invent facts, actors, events, relations, dates, or citations.
7. If the evidence does not establish a causal connection, state that
   limitation explicitly.
8. If the retrieved context is insufficient to answer the question, say so.
9. Keep the answer clear and concise.

Use this format:

Answer:
A direct answer to the question.

Causal reasoning:
Explain the important steps in chronological or causal order, citing
the supporting events and relations.

Evidence limitations:
Mention any important uncertainty or missing evidence. If there are no
material limitations, state that the answer is supported by the retrieved
context.

Return plain text, not JSON.
""".strip()


def generate_answer(
    question: str,
    context: dict,
    model_name: str,
) -> str:
    """Generate an answer from retrieved temporal graph evidence."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Check the project-root .env file."
        )

    client = genai.Client(api_key=api_key)

    prompt = build_prompt(question, context)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response. Check model access and quota."
        )

    return response.text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Answer questions using temporal graph retrieval and Gemini."
        )
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Question to answer using the temporal graph.",
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
        help="Number of initial event matches.",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Maximum graph distance expanded from the matches.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=10,
        help="Maximum number of retrieved events.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model name.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the retrieved evidence before the answer.",
    )

    args = parser.parse_args()

    if args.top_k < 1 or args.max_events < 1:
        parser.error("--top-k and --max-events must be at least 1.")

    if args.hops < 0:
        parser.error("--hops cannot be negative.")

    try:
        graph = load_graph(args.graph)

        context = retrieve_lineage(
            graph=graph,
            question=args.question,
            top_k=args.top_k,
            hops=args.hops,
            max_events=args.max_events,
        )

        if not context["events"]:
            raise ValueError(
                "No events were retrieved. Check the graph and question."
            )

        if args.show_context:
            print("\n=== RETRIEVED EVIDENCE ===")
            print(
                json.dumps(
                    context,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        print("\n=== GENERATED ANSWER ===\n")

        answer = generate_answer(
            question=args.question,
            context=context,
            model_name=args.model,
        )

        print(answer)

        print("\n=== RETRIEVAL METADATA ===")
        print(
            json.dumps(
                context["metadata"],
                indent=2,
                ensure_ascii=False,
            )
        )

    except (
        FileNotFoundError,
        json.JSONDecodeError,
        ValueError,
        RuntimeError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
