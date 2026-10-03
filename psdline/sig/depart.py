"""发车放行。

发车前必须同时满足两个条件：间隙融合裁决为无异常，且关门记录已落盘。两个
条件缺一不放行，放行失败时返回具体是哪一条没满足（探测未确认、多源未收敛、
等待人工复核还是判定有异物），便于值班员判断该去复检探测还是去复检门控。
"""

from __future__ import annotations

import json

from ..gap import BLOCKED, CLEAR, PENDING, REVIEW, GapRegistry
from ..store import JsonStore
from .sig import SignalController

_GAP_VIOLATIONS = {
    BLOCKED: "间隙探测判定有异物",
    PENDING: "间隙多源探测未收敛",
    REVIEW: "间隙探测等待人工复核",
}


def close_doc_id(train_id: str) -> str:
    return f"door/{train_id}/close"


def close_durable(store: JsonStore, train_id: str) -> bool:
    """读取关门记录并判断是否已落盘。"""

    raw = store.read(close_doc_id(train_id))
    if isinstance(raw, dict):
        return bool(raw.get("durable", False))
    if isinstance(raw, str):
        try:
            return bool(json.loads(raw).get("durable", False))
        except json.JSONDecodeError:
            return False
    return False


def depart_checked(
    signals: SignalController,
    store: JsonStore,
    gaps: GapRegistry,
    train_id: str,
    at: float,
) -> dict:
    """按顺序核对间隙裁决与关门落盘，然后升起发车信号。"""

    verdict = gaps.verdict(train_id, at)
    if verdict.state != CLEAR:
        return {
            "ok": False,
            "reason": f"gap_{verdict.state or 'unknown'}",
            "train_id": train_id,
            "violations": [_GAP_VIOLATIONS.get(verdict.state, "间隙探测未确认")],
            "verdict": verdict.as_dict(),
        }
    if not close_durable(store, train_id):
        return {
            "ok": False,
            "reason": "close_not_durable",
            "train_id": train_id,
            "violations": ["关门未落盘"],
        }
    signals.raise_signal(train_id, "depart", at)
    return {"ok": True, "train_id": train_id, "aspect": "green", "violations": []}
