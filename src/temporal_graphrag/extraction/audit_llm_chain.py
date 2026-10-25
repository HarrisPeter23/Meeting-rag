
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = ROOT / "data" / "processed" / "set_01000_model_input.json"
CHAIN_PATH = ROOT / "data" / "processed" / "set_01000_llm_causal_chain.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    model_input = load_json(INPUT_PATH)
    chain_data = load_json(CHAIN_PATH)

    utterances = (
        model_input
        if isinstance(model_input, list)
        else model_input.get("utterances", [])
    )

    utterance_map = {
        item["utterance_id"]: item
        for item in utterances
    }

    events = (
        chain_data.get("events", [])
        if isinstance(chain_data, dict)
        else []
    )

    print(f"Events to audit: {len(events)}")
    print("=" * 80)

    for event in events:
        print(f"\nEvent: {event.get('event_id')}")
        print(f"Type: {event.get('event_type')}")
        print(f"Actor: {event.get('actor')}")
        print(f"Description: {event.get('description')}")

        source_ids = event.get("source_utterance_ids", [])

        if not source_ids:
            print("  WARNING: No source utterances cited.")

        for source_id in source_ids:
            utterance = utterance_map.get(source_id)

            if utterance is None:
                print(f"  INVALID SOURCE: {source_id}")
                continue

            print(
                f"  [{source_id}] "
                f"{utterance.get('speaker_name', 'Unknown')}: "
                f"{utterance.get('text', '')}"
            )

            if utterance.get("meeting_id") != event.get("meeting_id"):
                print("    WARNING: Meeting ID mismatch.")

    print("\nAudit complete. Review event descriptions against their sources.")


if __name__ == "__main__":
    main()
