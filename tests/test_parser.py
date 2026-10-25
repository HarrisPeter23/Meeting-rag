
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSER = ROOT / "src" / "temporal_graphrag" / "ingestion" / "parse_transcript.py"
OUTPUT = ROOT / "data" / "processed" / "set_01000_utterances.json"


def test_parser_runs_successfully():
    result = subprocess.run(
        [sys.executable, str(PARSER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert OUTPUT.exists()


def test_utterances_have_required_fields():
    records = json.loads(OUTPUT.read_text(encoding="utf-8"))

    assert len(records) > 0

    required_fields = {
        "scenario_id",
        "meeting_id",
        "utterance_id",
        "speaker_id",
        "speaker_name",
        "text",
        "sequence_index",
        "source_line",
        "timestamp_seconds",
        "annotation_tags",
    }

    for record in records:
        assert required_fields.issubset(record.keys())
        assert record["scenario_id"] == "set_01000"
        assert record["meeting_id"] in {"m1", "m2", "m3"}
        assert record["text"].strip()
        assert record["speaker_id"]
        assert isinstance(record["sequence_index"], int)


def test_utterance_ids_are_unique():
    records = json.loads(OUTPUT.read_text(encoding="utf-8"))
    ids = [record["utterance_id"] for record in records]

    assert len(ids) == len(set(ids))


def test_all_three_meetings_are_present():
    records = json.loads(OUTPUT.read_text(encoding="utf-8"))
    meeting_ids = {record["meeting_id"] for record in records}

    assert meeting_ids == {"m1", "m2", "m3"}


def test_utterances_follow_source_order():
    records = json.loads(OUTPUT.read_text(encoding="utf-8"))
    sequence_numbers = [r["sequence_index"] for r in records]

    assert sequence_numbers == list(
        range(1, len(records) + 1)
    )
