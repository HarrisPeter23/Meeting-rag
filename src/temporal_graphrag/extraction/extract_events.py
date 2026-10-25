
import json
from pathlib import Path

from temporal_graphrag.schemas.causal_chain import DecisionEvent, EventType


ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = ROOT / "data" / "processed" / "set_01000_model_input.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "set_01000_events.json"


def main():
    # Load the cleaned transcript utterances.
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        utterances = json.load(file)

    # Create a lookup so every event can be checked against its source.
    utterance_by_id = {
        utterance["utterance_id"]: utterance
        for utterance in utterances
    }

    # Manually curated events grounded in the current Forecast Studio transcript.
    event_definitions = [
        {
            "event_id": "e1",
            "scenario_id": "set_01000",
            "meeting_id": "m1",
            "event_type": EventType.PROPOSAL,
            "description": (
                "Henrik proposes a free 14-day trial with no credit card "
                "required and the full feature set enabled."
            ),
            "actor": "Henrik",
            "decision_option": "option_A",
            "source_utterance_ids": ["m1_u0017"],
            "confidence": 1.0,
        },
        {
            "event_id": "e2",
            "scenario_id": "set_01000",
            "meeting_id": "m2",
            "event_type": EventType.EVIDENCE,
            "description": (
                "Sven reports that the pilot had 610 signups, a 4.1% paid "
                "conversion rate, 31% of signups never ran a forecast, "
                "and trial compute cost $11.20 per account."
            ),
            "actor": "Sven",
            "decision_option": None,
            "source_utterance_ids": [
                "m2_u0086",
                "m2_u0087",
            ],
            "confidence": 1.0,
        },
        
{
    "event_id": "e3",
    "scenario_id": "set_01000",
    "meeting_id": "m2",
    "event_type": EventType.OBJECTION,
    "description": (
        "Ingrid argues that projected trial compute costs of $56,000 "
        "against $16,200 in new monthly revenue are economically "
        "unfavorable. She suggests that requiring a credit card could "
        "filter out users who never use the product."
    ),
    "actor": "Ingrid",
    "decision_option": "option_B",
    "source_utterance_ids": [
        "m2_u0102",
        "m2_u0103",
    ],
    "confidence": 1.0,
}
,
        {
            "event_id": "e4",
            "scenario_id": "set_01000",
            "meeting_id": "m3",
            "event_type": EventType.RESPONSE,
            "description": (
                "Henrik responds that the $11.20 cost came from an uncapped "
                "sandbox; the cached production trial tier costs $3.60 per "
                "account, reducing projected monthly compute costs to $18,000 "
                "for 5,000 signups."
            ),
            "actor": "Henrik",
            "decision_option": "option_A",
            "source_utterance_ids": ["m3_u0196"],
            "confidence": 1.0,
        },
        {
            "event_id": "e5",
            "scenario_id": "set_01000",
            "meeting_id": "m3",
            "event_type": EventType.FINAL_DECISION,
            "description": (
                "Diego confirms the final launch decision: a free 14-day "
                "trial with no credit card and all features enabled."
            ),
            "actor": "Diego",
            "decision_option": "option_A",
            "source_utterance_ids": ["m3_u0236"],
            "confidence": 1.0,
        },
    ]

    events = []

    for definition in event_definitions:
        # Validate that every cited utterance exists.
        for utterance_id in definition["source_utterance_ids"]:
            if utterance_id not in utterance_by_id:
                raise ValueError(
                    f"Source utterance not found: {utterance_id}"
                )

            utterance = utterance_by_id[utterance_id]

            # Ensure the event references the correct meeting.
            if utterance["meeting_id"] != definition["meeting_id"]:
                raise ValueError(
                    f"Meeting mismatch for {utterance_id}: "
                    f"expected {definition['meeting_id']}, "
                    f"found {utterance['meeting_id']}"
                )

        events.append(DecisionEvent(**definition))

    # Save the validated event records.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_data = [event.model_dump(mode="json") for event in events]

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)

    print(f"Successfully extracted {len(events)} manually curated events.")
    print(f"Output saved to: {OUTPUT_PATH}")

    for event in events:
        print(
            f"{event.event_id}: {event.event_type.value} | "
            f"{event.meeting_id} | {event.actor}"
        )


if __name__ == "__main__":
    main()
