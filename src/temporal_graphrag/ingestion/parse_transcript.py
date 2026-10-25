
import json
import re
from pathlib import Path


SCENARIO_ID = "set_01000"

# Matches meeting separators such as "== m1" or "\== m1"
MEETING_PATTERN = re.compile(r"^\s*\\?==\s*(m\d+)\s*$")

# Matches dialogue lines such as "Henrik: Can we discuss pricing?"
SPEAKER_PATTERN = re.compile(r"^([^:]+):\s*(.*)$")

# Finds annotation tags such as @B1 and @D3
TAG_PATTERN = re.compile(r"@([A-Z]\d+)")


def normalize_speaker(name: str) -> str:
    """Create a stable speaker ID from a speaker name."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def parse_transcript(input_path: Path) -> list[dict]:
    records = []
    current_meeting = None
    sequence = 0

    text = input_path.read_text(encoding="utf-8-sig")

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            continue

        # Detect the start of a new meeting.
        meeting_match = MEETING_PATTERN.match(line)

        if meeting_match:
            current_meeting = meeting_match.group(1)
            continue

        # Ignore dataset metadata before the first meeting.
        if current_meeting is None:
            continue

        # Parse a speaker and their utterance.
        speaker_match = SPEAKER_PATTERN.match(line)

        if not speaker_match:
            print(
                f"Warning: unparsed line {line_number}: {line[:80]}"
            )
            continue

        speaker_name = speaker_match.group(1).strip()
        utterance_text = speaker_match.group(2).strip()

        if not utterance_text:
            continue

        sequence += 1

        records.append({
            "scenario_id": SCENARIO_ID,
            "meeting_id": current_meeting,
            "utterance_id": f"{current_meeting}_u{sequence:04d}",
            "speaker_id": normalize_speaker(speaker_name),
            "speaker_name": speaker_name,
            "text": utterance_text,
            "sequence_index": sequence,
            "source_line": line_number,
            "timestamp_seconds": None,
            "annotation_tags": TAG_PATTERN.findall(utterance_text)
        })

    if not records:
        raise ValueError(
            "No dialogue was parsed. Check the meeting separators "
            "and speaker format in the input file."
        )

    return records


def main():
    project_root = Path(__file__).resolve().parents[3]

    input_path = (
        project_root / "data" / "raw" / "transcripts"
        / "set_01000.txt"
    )

    output_path = (
        project_root / "data" / "processed"
        / "set_01000_utterances.json"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Transcript not found: {input_path}"
        )

    records = parse_transcript(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Parsed utterances: {len(records)}")
    print(f"Meetings: {sorted(set(r['meeting_id'] for r in records))}")
    print(f"Speakers: {sorted(set(r['speaker_name'] for r in records))}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
