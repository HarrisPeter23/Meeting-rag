
import json
import re
from pathlib import Path

INPUT_PATH = Path("data/processed/set_01000_model_input.json")
REFERENCE_PATH = Path("data/processed/set_01000_reference_chain.json")

UTTERANCE_ID_PATTERN = re.compile(r"m\d+_u\d+")


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def collect_utterances(data):
    """Recursively index transcript utterances by their utterance ID."""
    utterances = {}

    def walk(obj):
        if isinstance(obj, dict):
            # Prefer explicit utterance ID fields.
            for key in ("utterance_id", "utteranceId", "id"):
                uid = obj.get(key)
                if (
                    isinstance(uid, str)
                    and UTTERANCE_ID_PATTERN.fullmatch(uid)
                ):
                    utterances[uid] = obj
                    break

            for value in obj.values():
                if isinstance(value, (dict, list)):
                    walk(value)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return utterances


def main():
    data = load_json(INPUT_PATH)
    reference = load_json(REFERENCE_PATH)

    events = reference.get("events", [])
    relations = reference.get("relations", [])

    utterances = collect_utterances(data)

    print(f"Transcript utterances indexed: {len(utterances)}")
    print(f"Reference events: {len(events)}")
    print(f"Reference relations: {len(relations)}")

    if not utterances:
        raise SystemExit(
            "ERROR: No utterances found. Check the transcript JSON structure."
        )

    # Check required event fields and event citation IDs.
    missing = []
    event_ids = set()

    print("\n" + "=" * 75)
    print("REFERENCE EVENTS AND SOURCE UTTERANCES")
    print("=" * 75)

    required_event_fields = [
        "event_id",
        "scenario_id",
        "meeting_id",
        "event_type",
        "description",
        "actor",
        "source_utterance_ids",
    ]

    for event in events:
        event_id = event.get("event_id", "<missing event_id>")
        event_ids.add(event_id)

        absent_fields = [
            field for field in required_event_fields
            if field not in event
        ]

        print(
            f"\n[{event_id}] "
            f"{event.get('event_type', 'UNKNOWN')} | "
            f"{event.get('meeting_id', 'UNKNOWN MEETING')} | "
            f"{event.get('actor', 'UNKNOWN ACTOR')}"
        )
        print(f"Description: {event.get('description', '<missing>')}")

        if absent_fields:
            print(f"ERROR: Missing fields: {absent_fields}")

        source_ids = event.get("source_utterance_ids", [])

        if not source_ids:
            print("WARNING: Event has no source utterance IDs.")

        for uid in source_ids:
            utterance = utterances.get(uid)
            print(f"\n  {uid}:")

            if utterance is None:
                print("    SOURCE NOT FOUND")
                missing.append((event_id, uid))
            else:
                print(
                    "    " + json.dumps(
                        utterance,
                        ensure_ascii=False,
                    )
                )

    # Check relation fields, endpoint references, and citations.
    print("\n" + "=" * 75)
    print("REFERENCE RELATIONS")
    print("=" * 75)

    required_relation_fields = [
        "source_event_id",
        "target_event_id",
        "relation_type",
        "explanation",
        "source_utterance_ids",
    ]

    invalid_relations = []

    for index, relation in enumerate(relations, start=1):
        source = relation.get("source_event_id")
        target = relation.get("target_event_id")
        relation_type = relation.get("relation_type", "UNKNOWN")

        print(f"\n{index}. {source} --{relation_type}--> {target}")
        print(f"Explanation: {relation.get('explanation', '<missing>')}")

        absent_fields = [
            field for field in required_relation_fields
            if field not in relation
        ]

        if absent_fields:
            print(f"ERROR: Missing fields: {absent_fields}")
            invalid_relations.append(
                (index, f"Missing fields: {absent_fields}")
            )

        if source not in event_ids:
            print(f"ERROR: Source event does not exist: {source}")
            invalid_relations.append(
                (index, f"Unknown source event: {source}")
            )

        if target not in event_ids:
            print(f"ERROR: Target event does not exist: {target}")
            invalid_relations.append(
                (index, f"Unknown target event: {target}")
            )

        if source == target:
            print("ERROR: Relation links an event to itself.")
            invalid_relations.append((index, "Self-referencing relation"))

        source_ids = relation.get("source_utterance_ids", [])

        print(f"Evidence IDs: {source_ids}")

        if not source_ids:
            print("WARNING: Relation has no evidence citations.")

        for uid in source_ids:
            if uid not in utterances:
                print(f"  SOURCE NOT FOUND: {uid}")
                missing.append((f"relation {index}", uid))

    # Summary
    print("\n" + "=" * 75)
    print("AUDIT SUMMARY")
    print("=" * 75)

    if missing:
        print(f"FAIL: {len(missing)} missing citation(s):")
        for owner, uid in missing:
            print(f"  {owner}: {uid}")
    else:
        print("PASS: All cited utterance IDs exist.")

    if invalid_relations:
        print(f"FAIL: {len(invalid_relations)} relation structure issue(s):")
        for index, issue in invalid_relations:
            print(f"  Relation {index}: {issue}")
    else:
        print("PASS: Relation endpoints and required fields are valid.")

    print(
        "\nIMPORTANT: Existing IDs do not prove that an event description "
        "or causal relation is semantically correct. Review the printed "
        "transcript evidence manually."
    )

    if missing or invalid_relations:
        raise SystemExit("\nAudit failed. Fix the reported issues first.")

    print("\nStructural citation audit passed.")


if __name__ == "__main__":
    main()
