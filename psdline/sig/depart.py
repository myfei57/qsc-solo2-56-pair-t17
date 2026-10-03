"""发车放行。

发车前必须同时满足两个条件：间隙融合裁决放行，且关门记录已落盘。两个条件缺一
不放行。间隙这一条不再由发车模块自行判断，而是调用融合裁决器的统一裁决出口
:meth:`GapFusion.depart_decision`：该调用先推进裁决（必要时落定/升级、请求人工
复核），再把读到的不可变裁决快照连同放行决定一起追加进哈希链，审计重放时按
快照指纹逐次核对。任何非 clear 的自动状态（证据窗口、待复核、证据不足）一律
拦下；人工放行凭证仅在该调用里一次性消耗。
"""

from __future__ import annotations

import json

from ..gap import GapFusion
from ..store import JsonStore
from .sig import SignalController


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
    gaps: GapFusion,
    train_id: str,
    at: float,
) -> dict:
    """按同一裁决核对间隙与关门落盘，然后升起发车信号。"""

    # 先查不消耗凭证的条件（关门落盘），再读裁决做发车决定，避免人工放行凭证
    # 被一次注定失败的尝试白白消耗。
    if not close_durable(store, train_id):
        return {
            "ok": False,
            "reason": "close_not_durable",
            "train_id": train_id,
            "violations": ["关门未落盘"],
            "gap": gaps.verdict(train_id, at).as_dict(),
            "decision_seq": None,
        }
    verdict, decision = gaps.depart_decision(train_id, at)
    if not verdict.can_depart:
        return {
            "ok": False,
            "reason": "gap_not_clear",
            "train_id": train_id,
            "violations": [verdict.reason()],
            "gap": verdict.as_dict(),
            "decision_seq": decision.seq,
        }
    signals.raise_signal(train_id, "depart", at)
    return {
        "ok": True,
        "train_id": train_id,
        "aspect": "green",
        "violations": [],
        "gap": verdict.as_dict(),
        "decision_seq": decision.seq,
    }
