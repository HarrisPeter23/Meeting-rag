
import pytest
from pydantic import ValidationError

from temporal_graphrag.schemas.causal_chain import (
    CausalChain,
    ChainLink,
    DecisionEvent,
    EventType,
    RelationType,
    validate_chain_references,
)


def make_event(event_id, event_type, utterance_id):
    return DecisionEvent(
        event_id=event_id,
        scenario_id="set_01000",
        meeting_id="m1",
        event_type=event_type,
        description=f"Test event {event_id}",
        source_utterance_ids=[utterance_id],
    )


def test_valid_chain_passes():
    proposal = make_event(
        "e1", EventType.PROPOSAL, "m1_u0001"
    )
    objection = make_event(
        "e2", EventType.OBJECTION, "m2_u0002"
    )

    link = ChainLink(
        source_event_id="e1",
        target_event_id="e2",
        relation_type=RelationType.CHALLENGES,
        explanation="Test relationship",
        source_utterance_ids=["m2_u0002"],
    )

    chain = CausalChain(
        scenario_id="set_01000",
        events=[proposal, objection],
        relations=[link],
    )

    validate_chain_references(chain)


def test_invalid_event_type_is_rejected():
    with pytest.raises(ValidationError):
        DecisionEvent(
            event_id="e1",
            scenario_id="set_01000",
            meeting_id="m1",
            event_type="not_a_real_event",
            description="Invalid event",
            source_utterance_ids=["m1_u0001"],
        )


def test_unknown_event_reference_is_rejected():
    event = make_event(
        "e1", EventType.PROPOSAL, "m1_u0001"
    )

    link = ChainLink(
        source_event_id="e1",
        target_event_id="missing_event",
        relation_type=RelationType.LEADS_TO,
        explanation="Invalid target",
        source_utterance_ids=["m1_u0001"],
    )

    chain = CausalChain(
        scenario_id="set_01000",
        events=[event],
        relations=[link],
    )

    with pytest.raises(ValueError, match="Unknown target event"):
        validate_chain_references(chain)


def test_duplicate_event_ids_are_rejected():
    first = make_event(
        "e1", EventType.PROPOSAL, "m1_u0001"
    )
    second = make_event(
        "e1", EventType.OBJECTION, "m2_u0002"
    )

    chain = CausalChain(
        scenario_id="set_01000",
        events=[first, second],
    )

    with pytest.raises(ValueError, match="Duplicate event_id"):
        validate_chain_references(chain)
