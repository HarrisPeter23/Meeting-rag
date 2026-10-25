
import json
from pathlib import Path

from temporal_graphrag.schemas.causal_chain import (
    CausalChain,
    ChainLink,
    validate_chain_references,
)


ROOT = Path(__file__).resolve().parents[3]

EVENTS_PATH = ROOT / "data" / "processed" / "set_01000_events.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "set_01000_causal_chain.json"


def main():
    with EVENTS_PATH.open("r", encoding="utf-8") as file:
        event_data = json.load(file)

    relations = [
        ChainLink(
            source_event_id="e2",
            target_event_id="e1",
            relation_type="challenges",
            explanation=(
                "Sven's pilot results and projected compute costs "
                "raise concerns about the economics of the proposed "
                "no-card trial."
            ),
            source_utterance_ids=[
                "m2_u0086",
                "m2_u0087",
                "m2_u0094",
                "m2_u0096",
            ],
        ),
        ChainLink(
            source_event_id="e3",
            target_event_id="e1",
            relation_type="challenges",
            explanation=(
                "Ingrid challenges the economics of the no-card trial "
                "and suggests requiring a credit card to filter out "
                "users who never use the product."
            ),
            source_utterance_ids=["m2_u0102", "m2_u0103"],
        ),
        ChainLink(
            source_event_id="e4",
            target_event_id="e2",
            relation_type="responds_to",
            explanation=(
                "Henrik responds to the earlier cost concern by "
                "distinguishing uncapped sandbox costs from production "
                "trial-tier costs."
            ),
            source_utterance_ids=["m3_u0196"],
        ),
        ChainLink(
            source_event_id="e4",
            target_event_id="e3",
            relation_type="responds_to",
            explanation=(
                "Henrik provides updated cost information during the "
                "reconsideration of Ingrid's objection."
            ),
            source_utterance_ids=["m3_u0194", "m3_u0196"],
        ),
        ChainLink(
            source_event_id="e5",
            target_event_id="e1",
            relation_type="confirms",
            explanation=(
                "Diego's final launch decision confirms the original "
                "proposal for a free 14-day trial without a credit card."
            ),
            source_utterance_ids=["m3_u0236"],
        ),
    ]

    chain = CausalChain(
        scenario_id="set_01000",
        events=event_data,
        relations=relations,
    )

    validate_chain_references(chain)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            chain.model_dump(mode="json"),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("Causal chain validation passed.")
    print(f"Events: {len(chain.events)}")
    print(f"Relations: {len(chain.relations)}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
