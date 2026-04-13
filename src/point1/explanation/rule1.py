"""Rule 1 explanation-slot helpers."""

from __future__ import annotations


def build_rule1_reason_slots(
    *,
    subject: str,
    missing_items: tuple[str, ...],
    unknown_items: tuple[str, ...],
) -> dict[str, str]:
    """Return explanation slots for one Rule 1 prediction."""
    scene_condition = "on foot at the construction site"
    if missing_items:
        return {
            "subject": subject,
            "missing_item": ", ".join(missing_items),
            "scene_condition": scene_condition,
        }
    if unknown_items:
        return {
            "subject": subject,
            "missing_item": ", ".join(unknown_items),
            "scene_condition": scene_condition,
        }
    return {
        "subject": subject,
        "missing_item": "",
        "scene_condition": scene_condition,
    }


def build_rule1_reason_text(
    *,
    subject: str,
    decision_state: str,
    missing_items: tuple[str, ...],
    unknown_items: tuple[str, ...],
) -> str:
    """Return human-readable Rule 1 reasoning from explanation slots."""
    if decision_state == "violation":
        joined_items = " and ".join(item.replace("_", " ") for item in missing_items)
        return f"{subject} is on foot at the site without {joined_items}."
    if decision_state == "unknown":
        joined_items = ", ".join(item.replace("_", " ") for item in unknown_items)
        return f"Visibility is insufficient to judge {joined_items} for {subject}."
    return f"{subject} appears compliant with the visible Rule 1 PPE requirements."
