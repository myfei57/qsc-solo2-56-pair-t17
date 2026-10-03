"""间隙融合状态快照。"""

from __future__ import annotations

from .fusion import AUTO, MANUAL_BLOCKED, MANUAL_CLEAR, GapFusion


def gap_status(gaps: GapFusion, train_id: str, at: float | None = None) -> dict:
    """返回一列车的融合裁决状态。"""

    verdict = gaps.verdict(train_id, at)
    return {
        "train_id": train_id,
        "state": verdict.label,
        "auto_label": verdict.auto_label,
        "clear": verdict.can_depart,
        "review_required": verdict.review_required,
        "source": verdict.source,
        "candidate": verdict.candidate,
        "coverage": round(verdict.coverage, 4),
        "weights": {
            "clear": round(verdict.clear_weight, 4),
            "blocked": round(verdict.blocked_weight, 4),
        },
        "override": dict(verdict.override),
        "sources": [view.as_dict() for view in verdict.sources],
        "basis_hash": verdict.basis_hash,
    }


def gap_summary(gaps: GapFusion, train_ids: list[str], at: float | None = None) -> dict:
    """汇总多列车的融合裁决状态（探测、控制台、审计共用同一口径）。"""

    items: list[dict] = []
    review_pending: list[str] = []
    for train_id in train_ids:
        item = gap_status(gaps, train_id, at)
        items.append(item)
        if item["review_required"]:
            review_pending.append(train_id)
    blocked = [
        item["train_id"]
        for item in items
        if item["state"] == "blocked"
    ]
    manual = [
        item["train_id"]
        for item in items
        if item["source"] in (MANUAL_CLEAR, MANUAL_BLOCKED)
    ]
    return {
        "count": len(items),
        "clear": len([item for item in items if item["clear"]]),
        "blocked": blocked,
        "review_pending": review_pending,
        "manual_overridden": manual,
        "sources": [source.as_dict() for source in gaps.sources],
        "trains": items,
    }
