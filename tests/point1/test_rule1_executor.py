"""Tests for Rule 1 predicate execution and explanation mapping."""

from __future__ import annotations

from common.schemas.bbox import NormalizedBBox
from point1.candidates.person import PersonCandidate
from point1.executor.rule1 import execute_rule1_candidate
from point1.predicates.rule1 import Rule1PredicateResult, Rule1PredicateSet


def _predicate(state: str, score: float, reason: str) -> Rule1PredicateResult:
    return Rule1PredicateResult(
        state=state,
        score=score,
        reason=reason,
        evidence_bbox=None,
    )


def _candidate() -> PersonCandidate:
    return PersonCandidate(
        candidate_id="person-1",
        bbox=NormalizedBBox(0.1, 0.2, 0.3, 0.6),
        score=0.9,
    )


def test_execute_rule1_candidate_marks_violation_when_hard_hat_missing() -> None:
    """A missing hard hat should become a Rule 1 violation with explanation slots."""
    candidate = _candidate()
    predicates = Rule1PredicateSet(
        candidate_id=candidate.candidate_id,
        person_visible=_predicate("yes", 0.95, "person is clearly visible"),
        ppe_applicable=_predicate("yes", 0.9, "worker is on foot"),
        head_region_visible=_predicate("yes", 0.9, "head region is visible"),
        hard_hat_visible=_predicate("no", 0.9, "no hard hat is visible"),
        upper_body_covered=_predicate("yes", 0.8, "upper body is covered"),
        lower_body_covered=_predicate("yes", 0.8, "lower body is covered"),
        toe_covered=_predicate("yes", 0.8, "toe protection is visible"),
    )

    prediction = execute_rule1_candidate(candidate, predicates)

    assert prediction.rule_id == 1
    assert prediction.decision_state == "violation"
    assert prediction.target_bbox == candidate.bbox
    assert prediction.reason_slots["missing_item"] == "hard_hat"
    assert "hard hat" in prediction.reason_text
    assert prediction.supporting_evidence_ids == ("person-1:hard_hat_visible",)


def test_execute_rule1_candidate_marks_no_violation_when_all_visible_items_pass() -> None:
    """Visible compliant PPE should yield a no_violation decision."""
    candidate = _candidate()
    predicates = Rule1PredicateSet(
        candidate_id=candidate.candidate_id,
        person_visible=_predicate("yes", 0.95, "person is clearly visible"),
        ppe_applicable=_predicate("yes", 0.9, "worker is on foot"),
        head_region_visible=_predicate("yes", 0.9, "head region is visible"),
        hard_hat_visible=_predicate("yes", 0.9, "hard hat is visible"),
        upper_body_covered=_predicate("yes", 0.8, "upper body is covered"),
        lower_body_covered=_predicate("yes", 0.8, "lower body is covered"),
        toe_covered=_predicate("yes", 0.8, "toe protection is visible"),
    )

    prediction = execute_rule1_candidate(candidate, predicates)

    assert prediction.decision_state == "no_violation"
    assert prediction.target_bbox is None
    assert prediction.unknown_items == ()
    assert "compliant" in prediction.reason_text


def test_execute_rule1_candidate_marks_unknown_when_visibility_is_insufficient() -> None:
    """Unknown visibility should defer the Rule 1 decision instead of forcing a label."""
    candidate = _candidate()
    predicates = Rule1PredicateSet(
        candidate_id=candidate.candidate_id,
        person_visible=_predicate("unknown", 0.4, "person crop is too small"),
        ppe_applicable=_predicate("unknown", 0.4, "unable to confirm on-foot worker"),
        head_region_visible=_predicate("unknown", 0.4, "head region is unclear"),
        hard_hat_visible=_predicate("unknown", 0.4, "head region is unclear"),
        upper_body_covered=_predicate("unknown", 0.4, "upper body is unclear"),
        lower_body_covered=_predicate("unknown", 0.4, "lower body is unclear"),
        toe_covered=_predicate("unknown", 0.4, "feet are unclear"),
    )

    prediction = execute_rule1_candidate(candidate, predicates)

    assert prediction.decision_state == "unknown"
    assert prediction.target_bbox is None
    assert set(prediction.unknown_items) == {
        "ppe_applicable",
        "person_visible",
        "head_region_visible",
        "hard_hat_visible",
        "upper_body_covered",
        "lower_body_covered",
        "toe_covered",
    }
    assert "insufficient" in prediction.reason_text


def test_execute_rule1_candidate_marks_not_applicable_when_candidate_is_not_on_foot() -> None:
    """Rule 1 should not fire for candidates the VLM marks as non-applicable."""
    candidate = _candidate()
    predicates = Rule1PredicateSet(
        candidate_id=candidate.candidate_id,
        person_visible=_predicate("yes", 0.95, "person is clearly visible"),
        ppe_applicable=_predicate("no", 0.9, "operator in excavator cab"),
        head_region_visible=_predicate("yes", 0.9, "head region is visible"),
        hard_hat_visible=_predicate("no", 0.9, "no hard hat is visible"),
        upper_body_covered=_predicate("no", 0.8, "upper body appears uncovered"),
        lower_body_covered=_predicate("no", 0.8, "lower body appears uncovered"),
        toe_covered=_predicate("no", 0.8, "toes appear uncovered"),
    )

    prediction = execute_rule1_candidate(candidate, predicates)

    assert prediction.decision_state == "no_violation"
    assert prediction.target_bbox is None
    assert prediction.supporting_evidence_ids == ()
    assert "not an on-foot Rule 1 inspection target" in prediction.reason_text


def test_execute_rule1_candidate_requires_visible_head_before_hard_hat_violation() -> None:
    """A hidden head region should downgrade hard-hat absence to unknown, not violation."""
    candidate = _candidate()
    predicates = Rule1PredicateSet(
        candidate_id=candidate.candidate_id,
        person_visible=_predicate("yes", 0.95, "person is clearly visible"),
        ppe_applicable=_predicate("yes", 0.9, "worker is on foot"),
        head_region_visible=_predicate("unknown", 0.2, "head region is occluded"),
        hard_hat_visible=_predicate("no", 0.8, "no hard hat is visible"),
        upper_body_covered=_predicate("yes", 0.8, "upper body is covered"),
        lower_body_covered=_predicate("yes", 0.8, "lower body is covered"),
        toe_covered=_predicate("yes", 0.8, "toe protection is visible"),
    )

    prediction = execute_rule1_candidate(candidate, predicates)

    assert prediction.decision_state == "unknown"
    assert prediction.target_bbox is None
    assert "head_region_visible" in prediction.unknown_items


def test_execute_rule1_candidate_still_allows_rule1_violation_with_other_risk_context() -> None:
    """Nearby excavator/risk context should not suppress a true Rule 1 violation."""
    candidate = _candidate()
    predicates = Rule1PredicateSet(
        candidate_id=candidate.candidate_id,
        person_visible=_predicate("yes", 0.95, "worker is clearly visible"),
        ppe_applicable=_predicate(
            "yes",
            0.9,
            "on-foot worker near an excavator; Rule 1 still applies to this candidate",
        ),
        head_region_visible=_predicate("yes", 0.9, "head region is visible"),
        hard_hat_visible=_predicate("no", 0.9, "no hard hat is visible"),
        upper_body_covered=_predicate("yes", 0.8, "upper body is covered"),
        lower_body_covered=_predicate("yes", 0.8, "lower body is covered"),
        toe_covered=_predicate("yes", 0.8, "toe protection is visible"),
    )

    prediction = execute_rule1_candidate(candidate, predicates)

    assert prediction.decision_state == "violation"
    assert prediction.target_bbox == candidate.bbox
