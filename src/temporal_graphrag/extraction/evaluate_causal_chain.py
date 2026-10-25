import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]

REFERENCE_PATH = ROOT / "data" / "processed" / "set_01000_reference_chain.json"
PREDICTION_PATH = ROOT / "data" / "processed" / "set_01000_llm_causal_chain.json"
INPUT_PATH = ROOT / "data" / "processed" / "set_01000_model_input.json"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def tokens(value):
    return set(re.findall(r"\b\w+\b", normalize(value)))


def f1_scores(tp, predicted, reference):
    precision = tp / predicted if predicted else 0.0
    recall = tp / reference if reference else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def event_similarity(ref, pred):
    # Meeting and actor identify the likely event; description helps
    # distinguish multiple events by the same actor in one meeting.
    if normalize(ref.get("meeting_id")) != normalize(pred.get("meeting_id")):
        return 0.0

    ref_actor = normalize(ref.get("actor"))
    pred_actor = normalize(pred.get("actor"))
    if ref_actor and pred_actor and ref_actor != pred_actor:
        return 0.0

    a = tokens(ref.get("description", ""))
    b = tokens(pred.get("description", ""))
    overlap = len(a & b) / len(a | b) if a | b else 0.0

    score = 0.65 * overlap
    if ref_actor and ref_actor == pred_actor:
        score += 0.25
    if normalize(ref.get("event_type")) == normalize(pred.get("event_type")):
        score += 0.10

    return score


def match_events(reference_events, predicted_events):
    # Greedy one-to-one matching. IDs may differ between the two files.
    candidates = []
    for ri, ref in enumerate(reference_events):
        for pi, pred in enumerate(predicted_events):
            score = event_similarity(ref, pred)
            if score >= 0.30:
                candidates.append((score, ri, pi))

    candidates.sort(reverse=True)
    used_ref = set()
    used_pred = set()
    ref_to_pred = {}

    for score, ri, pi in candidates:
        if ri in used_ref or pi in used_pred:
            continue
        used_ref.add(ri)
        used_pred.add(pi)
        ref_to_pred[reference_events[ri]["event_id"]] = (
            predicted_events[pi]["event_id"]
        )

    return ref_to_pred, used_ref, used_pred


def main():
    reference = load_json(REFERENCE_PATH)
    prediction = load_json(PREDICTION_PATH)
    model_input = load_json(INPUT_PATH)

    ref_events = reference.get("events", [])
    pred_events = prediction.get("events", [])
    ref_relations = reference.get("relations", [])
    pred_relations = prediction.get("relations", [])

    utterances = (
        model_input if isinstance(model_input, list)
        else model_input.get("utterances", [])
    )
    valid_utterance_ids = {
        item.get("utterance_id") for item in utterances
        if item.get("utterance_id")
    }

    ref_to_pred, matched_ref, matched_pred = match_events(
        ref_events, pred_events
    )

    print("=" * 68)
    print("CAUSAL CHAIN EVALUATION")
    print("=" * 68)
    print(f"Reference events: {len(ref_events)}")
    print(f"Predicted events: {len(pred_events)}")
    print(f"Matched events: {len(matched_ref)}")

    p, r, f1 = f1_scores(
        len(matched_ref), len(pred_events), len(ref_events)
    )
    print(f"Event precision: {p:.3f}")
    print(f"Event recall:    {r:.3f}")
    print(f"Event F1:        {f1:.3f}")

    print("\nEVENT TYPE CHECK")
    ref_by_id = {e["event_id"]: e for e in ref_events}
    pred_by_id = {e["event_id"]: e for e in pred_events}
    type_correct = 0

    for ref_id, pred_id in ref_to_pred.items():
        ref = ref_by_id[ref_id]
        pred = pred_by_id[pred_id]
        correct = normalize(ref.get("event_type")) == normalize(
            pred.get("event_type")
        )
        type_correct += int(correct)
        status = "MATCH" if correct else "TYPE MISMATCH"
        print(
            f"{status}: {ref_id} <-> {pred_id}: "
            f"{ref.get('event_type')} / {pred.get('event_type')}"
        )

    print(
        "Event type accuracy among matched events: "
        f"{type_correct / len(matched_ref):.3f}"
        if matched_ref else "Event type accuracy: N/A"
    )

    # Convert predicted endpoints back to reference IDs so different ID
    # naming conventions do not affect relation matching.
    pred_to_ref = {pred: ref for ref, pred in ref_to_pred.items()}

    def relation_key(relation, endpoint_map=None):
        source = relation.get("source_event_id")
        target = relation.get("target_event_id")

        if endpoint_map is not None:
            source = endpoint_map.get(source)
            target = endpoint_map.get(target)

        if source is None or target is None:
            return None

        return (
            source,
            target,
            normalize(relation.get("relation_type")),
        )

    ref_relation_keys = [
        relation_key(rel) for rel in ref_relations
    ]
    pred_relation_keys = [
        relation_key(rel, pred_to_ref) for rel in pred_relations
    ]

    ref_counts = Counter(k for k in ref_relation_keys if k is not None)
    pred_counts = Counter(k for k in pred_relation_keys if k is not None)
    true_positives = sum((ref_counts & pred_counts).values())

    p, r, f1 = f1_scores(
        true_positives, len(pred_relations), len(ref_relations)
    )

    print("\nCAUSAL RELATION METRICS")
    print(f"Reference relations: {len(ref_relations)}")
    print(f"Predicted relations: {len(pred_relations)}")
    print(f"Exact matches (type + direction): {true_positives}")
    print(f"Relation precision: {p:.3f}")
    print(f"Relation recall:    {r:.3f}")
    print(f"Relation F1:        {f1:.3f}")

    unmapped_relations = sum(k is None for k in pred_relation_keys)
    print(f"Relations with unmatched endpoints: {unmapped_relations}")

    # Evidence evaluation: compare citations for matched events and
    # matched relations. This measures overlap, not entailment.
    ref_evidence = set()
    pred_evidence = set()
    event_evidence_scores = []

    for ref_id, pred_id in ref_to_pred.items():
        ref = ref_by_id[ref_id]
        pred = pred_by_id[pred_id]
        ref_ids = set(ref.get("source_utterance_ids", []))
        pred_ids = set(pred.get("source_utterance_ids", []))

        ref_evidence.update(ref_ids)
        pred_evidence.update(pred_ids)

        union = ref_ids | pred_ids
        overlap = len(ref_ids & pred_ids) / len(union) if union else 1.0
        event_evidence_scores.append(overlap)

    evidence_precision = (
        len(ref_evidence & pred_evidence) / len(pred_evidence)
        if pred_evidence else 0.0
    )
    evidence_recall = (
        len(ref_evidence & pred_evidence) / len(ref_evidence)
        if ref_evidence else 0.0
    )
    evidence_f1 = (
        2 * evidence_precision * evidence_recall
        / (evidence_precision + evidence_recall)
        if evidence_precision + evidence_recall else 0.0
    )

    print("\nEVENT EVIDENCE OVERLAP")
    print(f"Unique reference event citations: {len(ref_evidence)}")
    print(f"Unique predicted event citations: {len(pred_evidence)}")
    print(f"Citation precision: {evidence_precision:.3f}")
    print(f"Citation recall:    {evidence_recall:.3f}")
    print(f"Citation F1:        {evidence_f1:.3f}")
    print(
        "Mean per-event citation Jaccard: "
        f"{sum(event_evidence_scores) / len(event_evidence_scores):.3f}"
        if event_evidence_scores else
        "Mean per-event citation Jaccard: N/A"
    )

    all_predicted_citations = set()
    for event in pred_events:
        all_predicted_citations.update(
            event.get("source_utterance_ids", [])
        )
    for relation in pred_relations:
        all_predicted_citations.update(
            relation.get("source_utterance_ids", [])
        )

    invalid = sorted(all_predicted_citations - valid_utterance_ids)
    print("\nCITATION VALIDITY")
    print(f"Unique predicted citation IDs: {len(all_predicted_citations)}")
    print(f"Invalid citation IDs: {len(invalid)}")
    if invalid:
        for citation in invalid:
            print(f"  INVALID: {citation}")
    else:
        print("All predicted citation IDs exist in the transcript.")

    print("\nLIMITATIONS")
    print("- Event matching uses meeting, actor, description, and event type.")
    print("- Relation scoring requires mapped endpoints, correct type, and direction.")
    print("- Citation overlap does not prove that evidence entails a claim.")
    print("- Results cover one scenario and are preliminary.")
    print("=" * 68)


if __name__ == "__main__":
    main()
