"""间隙探测：多源加权融合裁决。

发车前必须确认站台与列车之间没有异物。多路探测各带可信度权重接入融合裁决器，
结论按"权重 + 持续时间"裁决：一边倒且持续保持才落定为自动结论；不一致时先给
短证据窗口（窗口内保守拦停），仍定不下来就升级为人工复核。人工复核结论在有效
期内盖过自动结论：人工放行一次性、60 秒有效，人工拦停粘性保持。

裁决器是事件溯源的：探测上报、故障、复核、裁决落定、发车决定全部追加进每列车
一条的哈希链流水，探测接入、发车放行、审计重放读到的是同一份不可变裁决。
未对位的列车不接受人工复核。
"""

from __future__ import annotations

from .audit import audit_timeline, verify_all, verify_train
from .events import (
    ADJUDICATION,
    DEPARTURE_DECISION,
    OBSERVATION,
    REVIEW_RECORDED,
    REVIEW_REQUESTED,
    SOURCE_FAULT,
    GapEvent,
    GapEventJournal,
)
from .fusion import (
    AUTO,
    BLOCKED,
    CLEAR,
    HOLD,
    MANUAL_BLOCKED,
    MANUAL_CLEAR,
    REVIEW,
    UNKNOWN,
    FusionPolicy,
    GapFusion,
    Verdict,
)
from .source import (
    OBS_BLOCKED,
    OBS_CLEAR,
    OPERATOR,
    SUPERVISOR,
    GapSource,
    default_sources,
    parse_sources,
)
from .status import gap_status, gap_summary

# 旧调用方仍以 GapRegistry / ProbeResult 命名引用融合裁决器。
GapRegistry = GapFusion


class ProbeResult:
    """旧接口返回值的兼容外观（内部就是融合裁决快照）。"""

    __slots__ = ("train_id", "clear", "confirmed", "at", "verdict")

    def __init__(self, verdict: Verdict) -> None:
        self.verdict = verdict
        self.train_id = verdict.train_id
        self.clear = verdict.can_depart
        self.confirmed = verdict.can_depart
        self.at = verdict.at

    def as_dict(self) -> dict:
        return {
            "train_id": self.train_id,
            "clear": self.clear,
            "confirmed": self.confirmed,
            "at": self.at,
            "verdict": self.verdict.as_dict(),
        }


__all__ = [
    "AUTO",
    "ADJUDICATION",
    "BLOCKED",
    "CLEAR",
    "DEPARTURE_DECISION",
    "FusionPolicy",
    "HOLD",
    "MANUAL_BLOCKED",
    "MANUAL_CLEAR",
    "OBSERVATION",
    "OBS_BLOCKED",
    "OBS_CLEAR",
    "OPERATOR",
    "ProbeResult",
    "REVIEW",
    "REVIEW_RECORDED",
    "REVIEW_REQUESTED",
    "SOURCE_FAULT",
    "SUPERVISOR",
    "GapEvent",
    "GapEventJournal",
    "GapFusion",
    "GapRegistry",
    "GapSource",
    "Verdict",
    "UNKNOWN",
    "audit_timeline",
    "default_sources",
    "gap_status",
    "gap_summary",
    "parse_sources",
    "verify_all",
    "verify_train",
]
