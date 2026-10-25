
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from temporal_graphrag.schemas.causal_chain import (
    CausalChain,
    validate_chain_references,
)


ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    ROOT / "data" / "processed" / "set_01000_model_input.json"
)
OUTPUT_PATH = (
    ROOT / "data" / "processed" / "set_01000_llm_causal_chain.json"
)

load_dotenv(ROOT / ".env")


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Check your project-root .env file."
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        utterances = json.load(file)

    if not utterances:
        raise ValueError("The model-input transcript is empty.")

    utterance_by_id = {
        item["utterance_id"]: item for item in utterances
    }

    transcript = json.dumps(utterances, ensure_ascii=False, indent=2)

    
    
    prompt = f"""
You are extracting decision events and causal relations from meeting
transcripts for a Temporal GraphRAG research system.

Scenario ID: set_01000

Read the transcript and produce one structured CausalChain object.

Requirements:
1. Extract important decision-related events, including proposals,
   supporting evidence, objections, reopened decisions, responses,
   and final decisions.
2. Extract meaningful relations between events. Use only relation types
   allowed by the supplied schema.
3. Use the exact utterance_id values from the transcript as source
   references. Never invent utterance IDs.
4. Every event must use scenario_id "set_01000".
5. Use the actual meeting_id and speaker_name from the transcript.
6. Keep event descriptions faithful to what speakers said.
7. Do not confuse correlation, temporal order, or disagreement with
   proven causation. Choose the relation that best fits the evidence.
8. Include the final decision and relevant evidence or objections that
   led to its reconsideration.
9. Ignore unrelated discussion unless it materially affects the decision.
10. Return only the structured object matching the requested schema.
11. Do not assume the manually curated reference events are available.
    Extract independently from this transcript.

Transcript JSON:
{transcript}
"""



    client = genai.Client(api_key=api_key)

    print(f"Sending {len(utterances)} utterances to Gemini ({model_name})...")

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": CausalChain,
        },
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response. Check model access and quota."
        )

    # Parse the structured response using the existing Pydantic schema.
    chain = CausalChain.model_validate_json(response.text)

    # Validate event IDs and event-to-event references.
    validate_chain_references(chain)

    # Validate all event source IDs and meeting attribution.
    for event in chain.events:
        for utterance_id in event.source_utterance_ids:
            source = utterance_by_id.get(utterance_id)

            if source is None:
                raise ValueError(
                    f"Gemini invented or referenced an unknown utterance: "
                    f"{utterance_id}"
                )

            if source["meeting_id"] != event.meeting_id:
                raise ValueError(
                    f"Meeting mismatch: event {event.event_id} says "
                    f"{event.meeting_id}, but {utterance_id} belongs to "
                    f"{source['meeting_id']}."
                )

    # Validate relation evidence references.
    for relation in chain.relations:
        for utterance_id in relation.source_utterance_ids:
            if utterance_id not in utterance_by_id:
                raise ValueError(
                    f"Unknown relation source utterance: {utterance_id}"
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            chain.model_dump(mode="json"),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("Gemini extraction and structural validation succeeded.")
    print(f"Events extracted: {len(chain.events)}")
    print(f"Relations extracted: {len(chain.relations)}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(
        "Note: structural validation does not prove that the "
        "extracted relations are factually correct."
    )


if __name__ == "__main__":
    main()
