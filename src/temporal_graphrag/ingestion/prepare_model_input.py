
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = ROOT / "data" / "processed" / "set_01000_utterances.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "set_01000_model_input.json"


def prepare_model_input():
    records = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    cleaned_records = []

    for record in records:
        clean_record = record.copy()

        # Remove annotation markers from the text shown to the model.
        clean_record["text"] = re.sub(
            r"\s*@(?:B|D)\d+\b",
            "",
            clean_record["text"]
        ).strip()

        # Keep annotation tags only in the separate research record.
        clean_record.pop("annotation_tags", None)
        clean_record.pop("source_line", None)

        cleaned_records.append(clean_record)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(cleaned_records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Prepared {len(cleaned_records)} utterances.")
    print(f"Saved model input to: {OUTPUT_PATH}")


if __name__ == "__main__":
    prepare_model_input()
