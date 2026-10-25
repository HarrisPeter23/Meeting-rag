
import json
from pathlib import Path

from temporal_graphrag.schemas.causal_chain import (
    CausalChain,
    validate_chain_references,
)

OUTPUT_PATH = Path("data/processed/set_01000_reference_chain.json")

# Manually reviewed reference chain.
# Keep this separate from the raw Gemini output.
chain_data = {
    "scenario_id": "set_01000",
    "events": [
        {
            "event_id": "ref_01",
            "scenario_id": "set_01000",
            "event_type": "proposal",
            "meeting_id": "m1",
            "actor": "Henrik",
            "description": (
                "Proposes a free 14-day trial with no credit card required "
                "and full access to the features."
            ),
            "source_utterance_ids": [
                "m1_u0017",
                "m1_u0019",
                "m1_u0021",
            ],
            "confidence": 1.0,
        },
        {
            "event_id": "ref_02",
            "scenario_id": "set_01000",
            "event_type": "support",
            "meeting_id": "m1",
            "actor": "Henrik",
            "description": (
                "Supports the no-card trial by arguing that card requirements "
                "create friction and that users may invite teammates during a trial."
            ),
            "source_utterance_ids": [
                "m1_u0023",
                "m1_u0024",
                "m1_u0025",
            ],
            "confidence": 1.0,
        },
        {
            "event_id": "ref_03",
            "scenario_id": "set_01000",
            "event_type": "final_decision",
            "meeting_id": "m1",
            "actor": "Diego",
            "description": "Initially approves the proposed trial.",
            "source_utterance_ids": ["m1_u0037"],
            "confidence": 1.0,
        },
        {
            "event_id": "ref_04",
            "scenario_id": "set_01000",
            "event_type": "evidence",
            "meeting_id": "m2",
            "actor": "Sven",
            "description": (
                "Reports pilot results: 610 signups, 4.1% paid conversion, "
                "31% of users never ran a forecast, and $11.20 compute cost "
                "per trial account. The initial estimate projects $56,000 "
                "in monthly compute costs against $16,200 in new monthly revenue "
                "at 5,000 signups."
            ),
            "source_utterance_ids": [
                "m2_u0086",
                "m2_u0087",
                "m2_u0089",
                "m2_u0093",
                "m2_u0094",
                "m2_u0096",
            ],
            "confidence": 1.0,
        },
        {
            "event_id": "ref_05",
            "scenario_id": "set_01000",
            "event_type": "objection",
            "meeting_id": "m2",
            "actor": "Ingrid",
            "description": (
                "Challenges the no-card trial's economics and suggests requiring "
                "a credit card to filter out inactive trial users."
            ),
            "source_utterance_ids": [
                "m2_u0102",
                "m2_u0103",
                "m2_u0105",
            ],
            "decision_option": "option_B",
            "confidence": 1.0,
        },
        {
            "event_id": "ref_06",
            "scenario_id": "set_01000",
            "event_type": "decision_reopened",
            "meeting_id": "m2",
            "actor": "Diego",
            "description": (
                "Reopens the trial decision and requests staging compute-cost "
                "data and a landing-page experiment before reconsidering."
            ),
            "source_utterance_ids": [
                "m2_u0152",
                "m2_u0179",
            ],
            "confidence": 1.0,
        },
        {
            "event_id": "ref_07",
            "scenario_id": "set_01000",
            "event_type": "response",
            "meeting_id": "m3",
            "actor": "Henrik",
            "description": (
                "Provides revised evidence: cached production compute costs "
                "$3.60 per account, or $18,000 per month at 5,000 signups; "
                "the no-card landing page converts at 18% versus 6% for the "
                "card-required page; and 55% of paid accounts in an older "
                "product invited teammates during the trial."
            ),
            "source_utterance_ids": [
                "m3_u0196",
                "m3_u0197",
                "m3_u0198",
                "m3_u0199",
            ],
            "confidence": 1.0,
        },
        {
            "event_id": "ref_08",
            "scenario_id": "set_01000",
            "event_type": "evidence",
            "meeting_id": "m3",
            "actor": "Sven",
            "description": (
                "Calculates that the card-required trial would need 12.3% "
                "conversion to break even, above the cited industry-report "
                "ceiling of 9%."
            ),
            "source_utterance_ids": [
                "m3_u0212",
                "m3_u0214",
                "m3_u0216",
            ],
            "confidence": 1.0,
        },
        {
            "event_id": "ref_09",
            "scenario_id": "set_01000",
            "event_type": "support",
            "meeting_id": "m3",
            "actor": "Ingrid",
            "description": (
                "Changes her position and supports the no-card trial after "
                "considering the revised compute costs and conversion evidence."
            ),
            "source_utterance_ids": [
                "m3_u0217",
                "m3_u0219",
                "m3_u0230",
            ],
            "confidence": 1.0,
        },
        {
            "event_id": "ref_10",
            "scenario_id": "set_01000",
            "event_type": "final_decision",
            "meeting_id": "m3",
            "actor": "Diego",
            "description": (
                "Makes the final decision to offer a free 14-day trial without "
                "a credit card and with full feature access."
            ),
            "source_utterance_ids": ["m3_u0236"],
            "confidence": 1.0,
        },
    ],
    "relations": [
        {
            "source_event_id": "ref_02",
            "target_event_id": "ref_01",
            "relation_type": "supports",
            "explanation": (
                "Henrik's rationale about reducing signup friction and encouraging "
                "team invitations supports the proposed no-card trial."
            ),
            "source_utterance_ids": [
                "m1_u0023",
                "m1_u0024",
                "m1_u0025",
            ],
        },
        {
            "source_event_id": "ref_01",
            "target_event_id": "ref_03",
            "relation_type": "leads_to",
            "explanation": (
                "Diego initially approves the trial that Henrik proposed."
            ),
            "source_utterance_ids": ["m1_u0017", "m1_u0037"],
        },
        {
            "source_event_id": "ref_04",
            "target_event_id": "ref_03",
            "relation_type": "challenges",
            "explanation": (
                "The pilot economics raise concerns about the no-card trial "
                "that Diego initially approved."
            ),
            "source_utterance_ids": [
                "m2_u0086",
                "m2_u0087",
                "m2_u0093",
            ],
        },
        {
            "source_event_id": "ref_04",
            "target_event_id": "ref_05",
            "relation_type": "leads_to",
            "explanation": (
                "The unfavorable pilot economics provide the context for "
                "Ingrid's objection to the no-card trial."
            ),
            "source_utterance_ids": [
                "m2_u0086",
                "m2_u0087",
                "m2_u0102",
                "m2_u0103",
            ],
        },
        {
            "source_event_id": "ref_05",
            "target_event_id": "ref_06",
            "relation_type": "leads_to",
            "explanation": (
                "The disagreement over trial economics leads to Diego requesting "
                "additional evidence before reconsidering the decision."
            ),
            "source_utterance_ids": [
                "m2_u0102",
                "m2_u0103",
                "m2_u0152",
                "m2_u0179",
            ],
        },
        {
            "source_event_id": "ref_06",
            "target_event_id": "ref_03",
            "relation_type": "reopens",
            "explanation": (
                "Diego reopens the previously approved trial decision pending "
                "additional cost and conversion evidence."
            ),
            "source_utterance_ids": [
                "m1_u0037",
                "m2_u0152",
                "m2_u0179",
            ],
        },
        {
            "source_event_id": "ref_07",
            "target_event_id": "ref_05",
            "relation_type": "responds_to",
            "explanation": (
                "Henrik presents revised compute costs and conversion results "
                "that address the concerns behind Ingrid's objection."
            ),
            "source_utterance_ids": [
                "m2_u0102",
                "m2_u0103",
                "m3_u0196",
                "m3_u0197",
            ],
        },
        {
            "source_event_id": "ref_08",
            "target_event_id": "ref_09",
            "relation_type": "supports",
            "explanation": (
                "The break-even analysis supports Ingrid's change in position "
                "by showing the difficulty of making a card-required trial viable."
            ),
            "source_utterance_ids": [
                "m3_u0212",
                "m3_u0214",
                "m3_u0216",
                "m3_u0217",
            ],
        },
        {
            "source_event_id": "ref_07",
            "target_event_id": "ref_09",
            "relation_type": "supports",
            "explanation": (
                "The lower revised compute cost and higher no-card conversion "
                "support Ingrid's change in position."
            ),
            "source_utterance_ids": [
                "m3_u0196",
                "m3_u0197",
                "m3_u0217",
                "m3_u0219",
            ],
        },
        {
            "source_event_id": "ref_09",
            "target_event_id": "ref_10",
            "relation_type": "supports",
            "explanation": (
                "Ingrid's support for the no-card approach is consistent with "
                "Diego's final decision."
            ),
            "source_utterance_ids": ["m3_u0230", "m3_u0236"],
        },
        {
            "source_event_id": "ref_10",
            "target_event_id": "ref_03",
            "relation_type": "supersedes",
            "explanation": (
                "The final decision in the third meeting replaces the earlier "
                "approval with a decision made after reviewing additional evidence."
            ),
            "source_utterance_ids": ["m1_u0037", "m3_u0236"],
        },
    ],
}


def main():
    # Validate the chain against the project's Pydantic schema.
    validated_chain = CausalChain.model_validate(chain_data)
    validate_chain_references(validated_chain)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            validated_chain.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Reference chain saved to: {OUTPUT_PATH}")
    print(f"Events: {len(validated_chain.events)}")
    print(f"Relations: {len(validated_chain.relations)}")
    print("Schema and reference validation passed.")


if __name__ == "__main__":
    main()
