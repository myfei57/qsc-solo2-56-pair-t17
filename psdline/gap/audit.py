"""间隙裁决的审计视角。

审计不读发车模块的结论，也不另算一份：它直接从不可变事件链重放裁决，再把发车
决定事件里内嵌的裁决指纹与重放结果逐一核对。指纹对得上，才能证明"探测、放行、
审计读的是同一份裁决"，且放行之后历史没有被改过。
"""

from __future__ import annotations

import os
import tempfile

from ..store import JsonStore
from .events import DEPARTURE_DECISION, GapEventJournal
from .fusion import FusionPolicy, GapFusion
from .source import GapSource


def _verify_with(
    train_id: str,
    events: list,
    meta: dict,
    sources: tuple[GapSource, ...],
    policy: FusionPolicy,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="gap-audit-") as scratch:
        fusion = GapFusion(JsonStore(scratch), sources=sources, policy=policy)
        state = fusion._state(train_id)  # noqa: SLF001 - 审计模块允许装配内部状态
        state.events = list(events)
        decisions: list[dict] = []
        mismatches: list[int] = []
        for index, event in enumerate(events):
            if event.type != DEPARTURE_DECISION:
                continue
            # 决定事件内嵌的是"该事件之前那条链"算出的裁决，核对时截到前一事件。
            prior = events[:index]
            embedded = event.payload.get("verdict", {})
            replayed = fusion._compute_with(  # noqa: SLF001
                state, prior, train_id, event.at
            )
            same = (
                embedded.get("basis_hash") == replayed.basis_hash
                and embedded.get("tail", {}).get("hash") == replayed.tail_hash
            )
            if not same:
                mismatches.append(event.seq)
            decisions.append(
                {
                    "seq": event.seq,
                    "at": event.at,
                    "granted": bool(event.payload.get("granted", False)),
                    "reason": str(event.payload.get("reason", "")),
                    "basis_match": same,
                    "embedded_basis": embedded.get("basis_hash", ""),
                    "replayed_basis": replayed.basis_hash,
                }
            )
    return {
        "train_id": train_id,
        "events": len(events),
        "broken_chain": bool(meta.get("broken_chain")),
        "truncated_lines": int(meta.get("truncated_lines", 0)),
        "decisions": decisions,
        "mismatches": mismatches,
        "consistent": not mismatches and not meta.get("broken_chain"),
    }


def verify_train(
    train_id: str,
    root: str,
    sources: tuple[GapSource, ...],
    policy: FusionPolicy | None = None,
) -> dict:
    """重放一列车的完整事件链并核对每次发车决定的裁决指纹。"""

    events, meta = GapEventJournal(root, train_id).replay()
    return _verify_with(train_id, events, meta, sources, policy or FusionPolicy())


def verify_all(
    root: str,
    sources: tuple[GapSource, ...],
    policy: FusionPolicy | None = None,
) -> list[dict]:
    """校验存储目录下全部列车的间隙事件链。"""

    used_policy = policy or FusionPolicy()
    reports: list[dict] = []
    if not os.path.isdir(root):
        return reports
    for name in sorted(os.listdir(root)):
        if not (name.startswith("gap-") and name.endswith(".jsonl")):
            continue
        train_id = name[len("gap-") : -len(".jsonl")]
        events, meta = GapEventJournal(root, train_id).replay()
        reports.append(_verify_with(train_id, events, meta, sources, used_policy))
    return reports


def audit_timeline(gaps: GapFusion, train_id: str) -> list[dict]:
    """把一列车的不可变事件链渲染成审计时间线。"""

    return [
        {
            "seq": event.seq,
            "at": event.at,
            "type": event.type,
            "payload": dict(event.payload),
            "hash": event.hash[:12],
        }
        for event in gaps.replay_events(train_id)
    ]
