"""多源间隙探测的融合裁决。

裁决是纯函数：输入探测流水、复核流水与当前时刻，输出唯一裁决，不碰任何
存储。登记表（``GapRegistry``）只负责把流水落盘、在裁决变化时追加裁决历史，
并在重启后从流水重建——因此恢复与重放走的都是这同一个函数，结论必然一致。

裁决规则（按优先级）：

1. 有效期内的人工复核直接定结论，盖过一切自动裁决；
2. 每路探测取最新一条未过期读数，过期（静默超时）的源退出融合；
3. 所有在场源结论一致时立即裁决；
4. 结论不一致时先给一个很短的证据窗口（``window_s``），窗口内裁决为
   ``pending``——不放行但也不锁死，等更多证据；
5. 窗口耗尽后按"源权重 × 持续时长"裁决：每路的有效权重随其当前读数
   持续的时间线性爬坡（``confirm_s`` 内爬满），权重和大的一侧胜出；
6. 两侧有效权重相等（实在定不下来）时裁决为 ``review``，要求人工复核。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

CLEAR = "clear"
BLOCKED = "blocked"
PENDING = "pending"
REVIEW = "review"
UNKNOWN = ""

_EPSILON = 1e-9


@dataclass(frozen=True)
class Observation:
    """一路探测源上报的一次原始读数（只追加，不修改）。"""

    train_id: str
    source: str
    clear: bool
    at: float
    seq: int

    def as_dict(self) -> dict:
        return {
            "train_id": self.train_id,
            "source": self.source,
            "clear": self.clear,
            "at": self.at,
            "seq": self.seq,
        }

    @staticmethod
    def from_dict(payload: dict) -> "Observation":
        return Observation(
            train_id=str(payload.get("train_id", "")),
            source=str(payload.get("source", "")),
            clear=bool(payload.get("clear", False)),
            at=float(payload.get("at", 0.0)),
            seq=int(payload.get("seq", 0)),
        )


@dataclass(frozen=True)
class Review:
    """一次人工复核结论；``until`` 之前盖过自动裁决。"""

    train_id: str
    clear: bool
    reviewer: str
    at: float
    until: float
    seq: int

    def as_dict(self) -> dict:
        return {
            "train_id": self.train_id,
            "clear": self.clear,
            "reviewer": self.reviewer,
            "at": self.at,
            "until": self.until,
            "seq": self.seq,
        }

    @staticmethod
    def from_dict(payload: dict) -> "Review":
        return Review(
            train_id=str(payload.get("train_id", "")),
            clear=bool(payload.get("clear", False)),
            reviewer=str(payload.get("reviewer", "")),
            at=float(payload.get("at", 0.0)),
            until=float(payload.get("until", 0.0)),
            seq=int(payload.get("seq", 0)),
        )


@dataclass(frozen=True)
class Verdict:
    """一次融合裁决的结果。"""

    train_id: str
    state: str
    origin: str
    reason: str
    at: float
    clear_weight: float = 0.0
    blocked_weight: float = 0.0
    contested: bool = False
    sources: dict = field(default_factory=dict)
    review: dict | None = None

    def as_dict(self) -> dict:
        return {
            "train_id": self.train_id,
            "state": self.state,
            "origin": self.origin,
            "reason": self.reason,
            "at": self.at,
            "clear_weight": round(self.clear_weight, 6),
            "blocked_weight": round(self.blocked_weight, 6),
            "contested": self.contested,
            "sources": {key: dict(value) for key, value in self.sources.items()},
            "review": dict(self.review) if self.review else None,
        }


def _sorted(observations: list[Observation]) -> list[Observation]:
    """按（时刻, 序号）排序，容忍多源时钟轻微歪斜。"""

    return sorted(observations, key=lambda item: (item.at, item.seq))


def _reading_start(history: list[Observation], current: Observation, now: float) -> float:
    """返回当前读数从什么时候开始持续（同值读数的最早时刻）。"""

    start = current.at
    for item in reversed(history):
        if item.at > now or item.seq >= current.seq:
            continue
        if item.clear != current.clear:
            break
        start = item.at
    return start


def adjudicate(
    train_id: str,
    observations: list[Observation],
    reviews: list[Review],
    now: float,
    *,
    weights: Mapping[str, float],
    default_weight: float,
    confirm_s: float,
    observation_ttl_s: float,
    window_s: float,
) -> Verdict:
    """对一列车在给定时刻做一次融合裁决（纯函数，可重放）。"""

    # 1. 有效期内的人工复核盖过一切自动结论。
    valid = [item for item in reviews if item.at <= now < item.until]
    if valid:
        latest = max(valid, key=lambda item: (item.at, item.seq))
        return Verdict(
            train_id=train_id,
            state=CLEAR if latest.clear else BLOCKED,
            origin="manual",
            reason="manual_review",
            at=now,
            review=latest.as_dict(),
        )

    # 2. 每源取最新一条读数，静默超时的源退出融合。
    by_source: dict[str, list[Observation]] = {}
    for item in _sorted(observations):
        by_source.setdefault(item.source, []).append(item)
    sources: dict[str, dict] = {}
    current: dict[str, Observation] = {}
    starts: dict[str, float] = {}
    for source, history in by_source.items():
        seen = [item for item in history if item.at <= now]
        if not seen:
            continue
        latest = seen[-1]
        age = now - latest.at
        stale = age > observation_ttl_s
        start = _reading_start(history, latest, now)
        weight = float(weights.get(source, default_weight))
        held = max(0.0, now - start)
        effective = weight if confirm_s <= 0 else weight * min(1.0, held / confirm_s)
        sources[source] = {
            "clear": latest.clear,
            "at": latest.at,
            "age_s": round(age, 6),
            "stale": stale,
            "weight": weight,
            "held_s": round(held, 6),
            "effective": round(0.0 if stale else effective, 6),
        }
        if not stale:
            current[source] = latest
            starts[source] = start

    if not current:
        reason = "all_stale" if sources else "no_evidence"
        return Verdict(
            train_id=train_id, state=UNKNOWN, origin="none", reason=reason, at=now,
            sources=sources,
        )

    clear_weight = sum(
        sources[source]["effective"] for source, item in current.items() if item.clear
    )
    blocked_weight = sum(
        sources[source]["effective"] for source, item in current.items() if not item.clear
    )

    # 3. 在场源结论一致：立即裁决。
    readings = {item.clear for item in current.values()}
    if len(readings) == 1:
        agreed = readings.pop()
        return Verdict(
            train_id=train_id,
            state=CLEAR if agreed else BLOCKED,
            origin="auto",
            reason="unanimous",
            at=now,
            clear_weight=clear_weight,
            blocked_weight=blocked_weight,
            sources=sources,
        )

    # 4. 结论不一致：先给短窗口等更多证据，窗口内不放行也不锁死。
    conflict_since = max(starts[source] for source in current)
    if now - conflict_since < window_s:
        return Verdict(
            train_id=train_id,
            state=PENDING,
            origin="auto",
            reason="disagree_window",
            at=now,
            clear_weight=clear_weight,
            blocked_weight=blocked_weight,
            contested=True,
            sources=sources,
        )

    # 5. 窗口耗尽：按权重 × 持续时长裁决；平局才要求人工复核。
    difference = clear_weight - blocked_weight
    if abs(difference) <= _EPSILON:
        return Verdict(
            train_id=train_id,
            state=REVIEW,
            origin="auto",
            reason="tie",
            at=now,
            clear_weight=clear_weight,
            blocked_weight=blocked_weight,
            contested=True,
            sources=sources,
        )
    return Verdict(
        train_id=train_id,
        state=CLEAR if difference > 0 else BLOCKED,
        origin="auto",
        reason="weighted",
        at=now,
        clear_weight=clear_weight,
        blocked_weight=blocked_weight,
        contested=True,
        sources=sources,
    )
