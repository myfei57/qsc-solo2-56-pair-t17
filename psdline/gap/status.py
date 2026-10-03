"""间隙状态快照。"""

from __future__ import annotations

from .fusion import CLEAR, PENDING, REVIEW
from .gap import GapRegistry


def gap_status(gaps: GapRegistry, train_id: str) -> dict:
    """返回一列车的间隙状态（含融合裁决与各路探测源健康度）。"""

    verdict = gaps.verdict(train_id)
    return {
        "train_id": train_id,
        "state": verdict.state,
        "clear": verdict.state == CLEAR,
        "origin": verdict.origin,
        "contested": verdict.contested,
        "verdict": verdict.as_dict(),
        "sources": gaps.source_health(train_id),
    }


def gap_summary(gaps: GapRegistry, train_ids: list[str]) -> dict:
    """汇总多列车的间隙状态。"""

    items = [gap_status(gaps, train_id) for train_id in train_ids]
    return {
        "count": len(items),
        "clear": len([item for item in items if item["clear"]]),
        "blocked": gaps.blocked_ids(),
        "pending": [item["train_id"] for item in items if item["state"] == PENDING],
        "review": [item["train_id"] for item in items if item["state"] == REVIEW],
        "trains": items,
    }
