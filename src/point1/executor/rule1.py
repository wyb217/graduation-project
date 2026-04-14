"""Rule 1 execution over extracted PPE predicates."""

from __future__ import annotations

from common.schemas.point1 import Point1Prediction
from point1.candidates.person import PersonCandidate
from point1.explanation.rule1 import build_rule1_reason_slots, build_rule1_reason_text
from point1.predicates.rule1 import Rule1PredicateSet

MISSING_ITEM_LABELS = {
    "hard_hat_visible": "hard_hat",
    "upper_body_covered": "upper_body_coverage",
    "lower_body_covered": "lower_body_coverage",
    "toe_covered": "toe_coverage",
}


def execute_rule1_candidate(
    candidate: PersonCandidate,
    predicates: Rule1PredicateSet,
) -> Point1Prediction:
    """Execute Rule 1 against one person candidate and its predicate bundle."""
    missing_items, unknown_items, no_violation_reason = _derive_rule1_decision_inputs(predicates)
    if no_violation_reason is not None:
        decision_state = "no_violation"
    elif missing_items:
        decision_state = "violation"
    elif unknown_items:
        decision_state = "unknown"
    else:
        decision_state = "no_violation"

    subject = _humanize_candidate_id(candidate.candidate_id)
    reason_slots = build_rule1_reason_slots(
        subject=subject,
        missing_items=missing_items,
        unknown_items=unknown_items,
        no_violation_reason=no_violation_reason,
    )
    confidence = _derive_confidence(predicates, decision_state)

    return Point1Prediction(
        rule_id=1,
        decision_state=decision_state,
        target_bbox=candidate.bbox if decision_state == "violation" else None,
        supporting_evidence_ids=tuple(
            f"{candidate.candidate_id}:{predicate_name}"
            for predicate_name, label in MISSING_ITEM_LABELS.items()
            if label in missing_items
        ),
        counter_evidence_ids=tuple(
            f"{candidate.candidate_id}:{predicate_name}"
            for predicate_name, label in MISSING_ITEM_LABELS.items()
            if getattr(predicates, predicate_name).state == "yes"
        ),
        unknown_items=unknown_items,
        reason_slots=reason_slots,
        reason_text=build_rule1_reason_text(
            subject=subject,
            decision_state=decision_state,
            missing_items=missing_items,
            unknown_items=unknown_items,
            no_violation_reason=no_violation_reason,
        ),
        confidence=confidence,
    )


def _humanize_candidate_id(candidate_id: str) -> str:
    suffix = candidate_id.split("-")[-1]
    return f"worker candidate {suffix}"


def _derive_confidence(predicates: Rule1PredicateSet, decision_state: str) -> float:
    states = {
        "person_visible": predicates.person_visible,
        "ppe_applicable": predicates.ppe_applicable,
        "head_region_visible": predicates.head_region_visible,
        "hard_hat_visible": predicates.hard_hat_visible,
        "upper_body_covered": predicates.upper_body_covered,
        "lower_body_covered": predicates.lower_body_covered,
        "toe_covered": predicates.toe_covered,
    }
    if decision_state == "violation":
        scores = [
            result.score
            for name, result in states.items()
            if name != "person_visible" and result.state == "no"
        ]
    elif decision_state == "no_violation":
        scores = [
            result.score
            for name, result in states.items()
            if name != "person_visible" and result.state == "yes"
        ]
    else:
        scores = [result.score for result in states.values() if result.state == "unknown"]
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def _derive_rule1_decision_inputs(
    predicates: Rule1PredicateSet,
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    if predicates.person_visible.state == "no":
        return (), (), "the candidate is not a clearly inspectable worker."
    if predicates.ppe_applicable.state == "no":
        return (), (), "the candidate is not an on-foot worker."

    missing_items: list[str] = []
    unknown_items: list[str] = []
    applicability_is_unknown = predicates.ppe_applicable.state == "unknown"

    if applicability_is_unknown:
        unknown_items.append("ppe_applicable")

    if predicates.person_visible.state == "unknown":
        unknown_items.append("person_visible")

    hard_hat_state = predicates.hard_hat_visible.state
    if hard_hat_state == "no":
        if applicability_is_unknown:
            unknown_items.append("hard_hat_visible")
        elif predicates.head_region_visible.state == "yes":
            missing_items.append(MISSING_ITEM_LABELS["hard_hat_visible"])
        else:
            unknown_items.append("head_region_visible")
    elif hard_hat_state == "unknown":
        unknown_items.append("hard_hat_visible")

    for predicate_name in ("upper_body_covered", "lower_body_covered", "toe_covered"):
        predicate = getattr(predicates, predicate_name)
        if predicate.state == "no":
            if applicability_is_unknown:
                unknown_items.append(predicate_name)
            else:
                missing_items.append(MISSING_ITEM_LABELS[predicate_name])
        elif predicate.state == "unknown":
            unknown_items.append(predicate_name)

    if (
        predicates.head_region_visible.state == "unknown"
        and "head_region_visible" not in unknown_items
    ):
        unknown_items.append("head_region_visible")

    return tuple(missing_items), tuple(dict.fromkeys(unknown_items)), None
