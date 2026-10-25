
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field



class EventType(str, Enum):
    PROPOSAL = "proposal"
    SUPPORT = "support"
    EVIDENCE = "evidence"
    OBJECTION = "objection"
    DECISION_REOPENED = "decision_reopened"
    RESPONSE = "response"
    POSITION_CHANGE = "position_change"
    FINAL_DECISION = "final_decision"



class RelationType(str, Enum):
    SUPPORTS = "supports"
    CHALLENGES = "challenges"
    RESPONDS_TO = "responds_to"
    REOPENS = "reopens"
    LEADS_TO = "leads_to"
    SUPERSEDES = "supersedes"
    CONFIRMS = "confirms"


class DecisionEvent(BaseModel):
    event_id: str
    scenario_id: str
    meeting_id: str

    event_type: EventType
    description: str
    actor: Optional[str] = None
    decision_option: Optional[str] = None

    # IDs of transcript utterances supporting this event.
    source_utterance_ids: list[str] = Field(min_length=1)

    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )


class ChainLink(BaseModel):
    source_event_id: str
    target_event_id: str

    relation_type: RelationType
    explanation: str

    # Utterances supporting this relationship.
    source_utterance_ids: list[str] = Field(min_length=1)


class CausalChain(BaseModel):
    scenario_id: str
    events: list[DecisionEvent] = Field(min_length=1)
    relations: list[ChainLink] = Field(default_factory=list)


def validate_chain_references(chain: CausalChain) -> None:
    """Validate event IDs and scenario consistency."""

    event_ids = [event.event_id for event in chain.events]

    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Duplicate event_id found.")

    event_id_set = set(event_ids)

    for event in chain.events:
        if event.scenario_id != chain.scenario_id:
            raise ValueError(
                f"Event {event.event_id} belongs to another scenario."
            )

    for link in chain.relations:
        if link.source_event_id not in event_id_set:
            raise ValueError(
                f"Unknown source event: {link.source_event_id}"
            )

        if link.target_event_id not in event_id_set:
            raise ValueError(
                f"Unknown target event: {link.target_event_id}"
            )

        if link.source_event_id == link.target_event_id:
            raise ValueError("An event cannot link to itself.")
