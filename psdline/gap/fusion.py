"""多源融合的间隙裁决引擎。

同一份事件流水喂出三种读者（探测接入、发车放行、审计）看到的同一个结论：

* **多路 + 可信度**：每路探测带权重；只有"在线、健康、上报未过期"的源参与融合，
  参与权重在这几路之间重新归一化。故障或静默的源被剔除而不是悄悄投出一票。
* **不一致时的取舍**：不按"最保守一路立即拦停"，也不无脑等满；先开一个短证据
  窗口（``hold_s``）让抖动沉淀，窗口内一律按拦停处理（宁误拦不漏判）。窗口结束
  仍定不下来（权重不够、低于一致阈值），升级为 *要求人工复核*，继续拦停。
* **持续时间**：瞬时一致还不够，同一个结论必须连续保持 ``confirm_s`` 才落定为
  自动结论，抑制单帧毛刺。
* **人工盖过自动**：自动判定 clear 时值班员即可确认；打架或可疑时必须督导级
  人工放行。人工放行是"一次性 + 60 秒有效"的凭证（只对紧接着的那一次发车有效，
  避免一令多用）；人工拦停粘性保持，直到下一次人工结论显式改变。

所有状态都是事件流水的纯函数：重放同一条链必然得到同一组结论（``state_hash``），
这是发车模块与审计模块读到同一份裁决的基础。
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Callable

from ..errors import InterlockError
from ..store import JsonStore
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
from .source import (
    OBS_BLOCKED,
    OBS_CLEAR,
    OPERATOR,
    SUPERVISOR,
    GapSource,
    default_sources,
)

# 自动裁决标签
UNKNOWN = "unknown"            # 还没有任何可用证据
HOLD = "hold"                  # 证据窗口内：结论可能翻转，先拦停
REVIEW = "review_required"     # 窗口结束仍定不下来，要求人工复核
CLEAR = "clear"
BLOCKED = "blocked"

# 顶层结论来源
AUTO = "auto"
MANUAL_CLEAR = "manual_clear"
MANUAL_BLOCKED = "manual_blocked"

# 发车被拦的原因（与旧调用方的中文提示保持一致）
REASON_BLOCKED = "间隙可疑"
REASON_REVIEW = "间隙判定打架，等待人工复核"
REASON_UNCONFIRMED = "间隙探测未确认"

_AUTO_LABELS = (CLEAR, BLOCKED, HOLD, REVIEW, UNKNOWN)
_DEFINITE = (CLEAR, BLOCKED)


@dataclass(frozen=True)
class FusionPolicy:
    """融合裁决参数（全部可在配置里调）。"""

    confirm_s: float = 4.0            # 结论需连续保持多久才落定
    hold_s: float = 6.0             # 不一致时给证据窗口的时长
    review_timeout_s: float = 120.0  # 复核请求挂多久算超时（只告警，不自动放行）
    freshness_s: float = 20.0       # 探测上报超过多久视为过期
    min_quorum: float = 0.6         # 参与融合的权重至少占总权重多少
    agree_ratio: float = 2.0 / 3.0  # 一致阈值：清/疑各自达到该份额才算一边倒
    manual_clear_ttl_s: float = 60.0  # 人工放行有效期
    late_grace_s: float = 30.0      # 比最新事件还晚到多久算迟到上报（仅标记）

    def validate(self) -> "FusionPolicy":
        if self.confirm_s <= 0 or self.hold_s <= 0 or self.freshness_s <= 0:
            raise ValueError("时间窗必须为正")
        if not 0 < self.min_quorum <= 1:
            raise ValueError("最低权重必须落在 (0, 1]")
        if not 0.5 < self.agree_ratio <= 1:
            raise ValueError("一致阈值必须落在 (0.5, 1]")
        if self.manual_clear_ttl_s <= 0:
            raise ValueError("人工放行有效期必须为正")
        return self


@dataclass(frozen=True)
class SourceView:
    """裁决快照里一路探测的视图。"""

    id: str
    name: str
    weight: float
    observation: str       # clear / blocked / ""
    age_s: float | None
    healthy: bool
    participating: bool
    share: float           # 在参与权重中的归一化份额

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "weight": round(self.weight, 6),
            "observation": self.observation,
            "age_s": None if self.age_s is None else round(self.age_s, 3),
            "healthy": self.healthy,
            "participating": self.participating,
            "share": round(self.share, 6),
        }


@dataclass(frozen=True)
class Verdict:
    """一次裁决的不可变快照——探测、发车、审计共读这一份。"""

    train_id: str
    at: float
    auto_label: str                      # clear/blocked/hold/review_required/unknown
    label: str                           # 顶层结论：clear / blocked（非放行即拦停）
    source: str                          # auto / manual_clear / manual_blocked
    review_required: bool
    review_request_seq: int | None
    review_overdue_s: float | None
    candidate: str                       # clear/blocked/split/insufficient/unknown
    held_for_s: float
    coverage: float                      # 参与权重 / 配置总权重
    clear_weight: float
    blocked_weight: float
    sources: tuple[SourceView, ...]
    override: dict                       # 生效中的人工结论（无则为空）
    tail_seq: int                        # 所依据的事件链位置
    tail_hash: str
    basis_hash: str = ""

    @property
    def can_depart(self) -> bool:
        """唯一的放行判据：顶层结论为 clear。"""

        return self.label == CLEAR

    def reason(self) -> str:
        """拦停时给值班员的具体原因。"""

        if self.label == BLOCKED:
            return REASON_REVIEW if self.source == AUTO and self.auto_label == REVIEW else REASON_BLOCKED
        if self.auto_label == REVIEW:
            return REASON_REVIEW
        return REASON_UNCONFIRMED

    def as_dict(self) -> dict:
        return {
            "train_id": self.train_id,
            "at": round(self.at, 6),
            "auto_label": self.auto_label,
            "label": self.label,
            "source": self.source,
            "review_required": self.review_required,
            "review_request_seq": self.review_request_seq,
            "review_overdue_s": None
            if self.review_overdue_s is None
            else round(self.review_overdue_s, 3),
            "candidate": self.candidate,
            "held_for_s": round(self.held_for_s, 3),
            "coverage": round(self.coverage, 6),
            "weights": {
                "clear": round(self.clear_weight, 6),
                "blocked": round(self.blocked_weight, 6),
            },
            "sources": [view.as_dict() for view in self.sources],
            "override": dict(self.override),
            "tail": {"seq": self.tail_seq, "hash": self.tail_hash},
            "basis_hash": self.basis_hash,
        }


@dataclass
class _TrainState:
    """一列车从事件流水重放出来的内部状态。"""

    events: list[GapEvent] = field(default_factory=list)
    last_auto_label: str = ""
    open_request_seq: int | None = None
    resolved_request_seq: int | None = None
    truncated: int = 0
    broken: bool = False


AuditHook = Callable[[str, str, dict], None]


class GapFusion:
    """多源融合间隙裁决器（每列车一条哈希链事件流水）。"""

    def __init__(
        self,
        store: JsonStore,
        sources: tuple[GapSource, ...] | None = None,
        policy: FusionPolicy | None = None,
        audit_hook: AuditHook | None = None,
    ) -> None:
        self._store = store
        self.sources: tuple[GapSource, ...] = tuple(sources or default_sources())
        if not self.sources:
            raise InterlockError("间隙融合至少需要一路探测源", component="gap")
        self.policy = (policy or FusionPolicy()).validate()
        self._source_ids = {source.id for source in self.sources}
        self._audit_hook = audit_hook
        self._lock = threading.RLock()
        self._states: dict[str, _TrainState] = {}
        self._load_all()

    # ---- 输入 ---------------------------------------------------------

    def report(
        self, train_id: str, source_id: str, clear: bool, at: float
    ) -> GapEvent:
        """登记一路探测上报；未知源、过期太久的迟到上报会被拒绝。"""

        if source_id not in self._source_ids:
            raise InterlockError(f"未知间隙探测源: {source_id}", component="gap")
        with self._lock:
            state = self._state(train_id)
            latest = max(
                (event.at for event in state.events if event.type == OBSERVATION),
                default=at,
            )
            late = latest - at > self.policy.late_grace_s
            event = self._append(
                state,
                train_id,
                OBSERVATION,
                {
                    "source_id": source_id,
                    "observation": OBS_CLEAR if clear else OBS_BLOCKED,
                    "late": bool(late),
                },
                at,
            )
            if late:
                self._audit(
                    "gap_late_observation",
                    train_id,
                    {"source_id": source_id, "lag_s": round(latest - at, 3)},
                )
            return event

    def mark_source(
        self, train_id: str, source_id: str, faulty: bool, at: float
    ) -> GapEvent:
        """标记一路探测故障（``faulty=False`` 表示恢复）。"""

        if source_id not in self._source_ids:
            raise InterlockError(f"未知间隙探测源: {source_id}", component="gap")
        with self._lock:
            state = self._state(train_id)
            return self._append(
                state, train_id, SOURCE_FAULT, {"source_id": source_id, "faulty": bool(faulty)}, at
            )

    def manual_review(
        self,
        train_id: str,
        clear: bool,
        at: float,
        *,
        reviewer: str = "operator",
        role: str = OPERATOR,
        note: str = "",
    ) -> GapEvent:
        """登记人工复核结论，结论在有效期内盖过自动裁决。

        自动判定不是 clear 时（打架、可疑、证据不足）必须由督导（``supervisor``）
        放行；值班员只能在自动已判 clear 时做常规人工确认。
        """

        if role not in (OPERATOR, SUPERVISOR):
            raise InterlockError(f"未知复核角色: {role}", component="gap")
        with self._lock:
            state = self._state(train_id)
            if not self._store.exists(f"train/{train_id}/align"):
                raise InterlockError(
                    f"列车 {train_id} 尚未对位，无法进行间隙人工复核", component="gap"
                )
            verdict = self._compute(state, train_id, at)
            if clear and verdict.auto_label != CLEAR and role != SUPERVISOR:
                raise InterlockError(
                    "自动间隙判定不是无异常，人工放行必须由督导确认",
                    component="gap",
                )
            payload: dict = {
                "decision": OBS_CLEAR if clear else OBS_BLOCKED,
                "reviewer": reviewer,
                "role": role,
                "note": note,
                "request_seq": self._open_request_seq(state.events),
            }
            if clear:
                payload["expires_at"] = round(at + self.policy.manual_clear_ttl_s, 6)
                payload["single_use"] = True
            event = self._append(state, train_id, REVIEW_RECORDED, payload, at)
            state.resolved_request_seq = event.seq
            self._audit(
                "gap_manual_review",
                train_id,
                {
                    "decision": payload["decision"],
                    "reviewer": reviewer,
                    "role": role,
                    "auto_label": verdict.auto_label,
                    "expires_at": payload.get("expires_at"),
                },
            )
            return event

    # ---- 裁决（唯一出口）----------------------------------------------

    def verdict(self, train_id: str, at: float | None = None) -> Verdict:
        """推进裁决并返回当前不可变快照（不产生新的事件）。"""

        with self._lock:
            return self._compute(self._state(train_id), train_id, at)

    def pump(self, train_id: str, at: float | None = None) -> Verdict:
        """推进裁决，并把"裁决落定/升级"和"请求人工复核"落进不可变流水。"""

        with self._lock:
            state = self._state(train_id)
            snapshot = self._compute(state, train_id, at)
            if snapshot.auto_label != state.last_auto_label:
                self._append(
                    state,
                    train_id,
                    ADJUDICATION,
                    {
                        "auto_label": snapshot.auto_label,
                        "candidate": snapshot.candidate,
                        "coverage": snapshot.coverage,
                        "clear_weight": snapshot.clear_weight,
                        "blocked_weight": snapshot.blocked_weight,
                    },
                    at,
                )
                state.last_auto_label = snapshot.auto_label
                self._audit(
                    "gap_adjudication",
                    train_id,
                    {"auto_label": snapshot.auto_label, "candidate": snapshot.candidate},
                )
                snapshot = self._compute(state, train_id, at)
            if snapshot.review_required and snapshot.review_request_seq is None:
                request = self._append(
                    state,
                    train_id,
                    REVIEW_REQUESTED,
                    {
                        "candidate": snapshot.candidate,
                        "coverage": snapshot.coverage,
                        "held_for_s": snapshot.held_for_s,
                    },
                    at,
                )
                self._audit(
                    "gap_review_requested",
                    train_id,
                    {"request_seq": request.seq, "candidate": snapshot.candidate},
                )
                snapshot = self._compute(state, train_id, at)
            return snapshot

    def depart_decision(self, train_id: str, at: float) -> tuple[Verdict, GapEvent]:
        """发车模块专用：先推进裁决，再把"读到的裁决"原样落一条决定事件。

        决定事件内嵌完整快照与快照指纹；审计重放时核对指纹，即可确认当时放行动作
        依据的就是这一份裁决。人工放行凭证一旦被放行决定使用即消耗。
        """

        with self._lock:
            state = self._state(train_id)
            snapshot = self.pump(train_id, at)
            consume_seq = None
            if snapshot.can_depart and snapshot.source == MANUAL_CLEAR:
                consume_seq = self._latest_review_seq(state)
            event = self._append(
                state,
                train_id,
                DEPARTURE_DECISION,
                {
                    "granted": snapshot.can_depart,
                    "reason": "" if snapshot.can_depart else snapshot.reason(),
                    "verdict": snapshot.as_dict(),
                    "consumed_override": consume_seq,
                },
                at,
            )
            return snapshot, event

    # ---- 兼容旧 GapRegistry 接口 --------------------------------------

    def check(self, train_id: str, at: float) -> "Verdict":
        """旧接口：值班员确认间隙无异常（等价于一次值班员人工复核 clear）。"""

        self.manual_review(train_id, True, at, reviewer="operator", role=OPERATOR)
        return self.pump(train_id, at)

    def probe(self, train_id: str, clear: bool, at: float, *, source_id: str = "") -> "Verdict":
        """旧接口：登记一路自动探测；未指定源时记到最高可信度的一路。"""

        source_id = source_id or self.sources[0].id
        self.report(train_id, source_id, clear, at)
        return self.pump(train_id, at)

    def clear(self, train_id: str, at: float | None = None) -> bool:
        """当前是否可放行（顶层结论为 clear）。"""

        return self.pump(train_id, at).can_depart

    def state(self, train_id: str, at: float | None = None) -> str:
        """顶层状态：``clear`` / ``blocked`` / 空串（尚无任何事件）。"""

        with self._lock:
            state = self._state(train_id)
            if not state.events:
                return ""
            return self.pump(train_id, at).label

    def blocked_ids(self, at: float | None = None) -> list[str]:
        """当前被拦停（顶层结论非 clear 且已有事件）的列车。"""

        with self._lock:
            found: list[str] = []
            for train_id in self._trains_with_events():
                verdict = self.pump(train_id, at)
                if verdict.label == BLOCKED:
                    found.append(train_id)
            return sorted(found)

    @property
    def checks(self) -> int:
        """累计人工复核次数。"""

        return self._count(REVIEW_RECORDED)

    @property
    def probes(self) -> int:
        """累计探测上报次数。"""

        return self._count(OBSERVATION)

    # ---- 审计 / 重放 ---------------------------------------------------

    def replay_events(self, train_id: str) -> list[GapEvent]:
        """返回一列车的完整不可变事件链（供审计逐页核对）。"""

        with self._lock:
            return list(self._state(train_id).events)

    def state_hash(self) -> str:
        """全部列车已落定事件链的指纹；重放前后必须一致。"""

        with self._lock:
            hasher = hashlib.sha256()
            for train_id in sorted(self._states):
                state = self._state(train_id)
                if state.events:
                    hasher.update(train_id.encode("utf-8"))
                    hasher.update(str(state.events[-1].seq).encode("ascii"))
                    hasher.update(state.events[-1].hash.encode("ascii"))
            return hasher.hexdigest()

    def integrity(self) -> dict:
        """各列车流水的完整性（哈希链校验、尾部截断恢复情况）。"""

        with self._lock:
            report: dict[str, dict] = {}
            for train_id in sorted(self._states):
                # 完整性巡检只读落盘流水，断链也必须能报出来而不是被拦下。
                events, meta = self._journal(train_id).replay()
                report[train_id] = {
                    "events": len(events),
                    "truncated_lines": meta["truncated_lines"],
                    "broken_chain": meta["broken_chain"],
                }
            return report

    # ---- 内部 ----------------------------------------------------------

    def _journal(self, train_id: str) -> GapEventJournal:
        return GapEventJournal(self._store.root, train_id)

    def _state(self, train_id: str) -> _TrainState:
        state = self._states.get(train_id)
        if state is None:
            state = self._load(train_id)
            self._states[train_id] = state
        if state.broken:
            raise InterlockError(
                f"列车 {train_id} 间隙裁决流水哈希链断裂，必须先做完整性处置",
                component="gap",
            )
        return state

    def _load(self, train_id: str) -> _TrainState:
        events, meta = self._journal(train_id).replay()
        state = _TrainState(
            events=events, truncated=meta["truncated_lines"], broken=meta["broken_chain"]
        )
        for event in events:
            if event.type == ADJUDICATION:
                state.last_auto_label = str(event.payload.get("auto_label", ""))
            elif event.type == REVIEW_RECORDED:
                # 最近一次复核结论（当前是否有挂起请求由事件链纯派生）。
                state.resolved_request_seq = event.seq
        state.open_request_seq = self._open_request_seq(events)
        return state

    def _load_all(self) -> None:
        """发现磁盘上已有的列车流水；实际校验在首次使用该车时惰性进行。"""

        import os

        root = self._store.root
        if os.path.isdir(root):
            for name in os.listdir(root):
                if name.startswith("gap-") and name.endswith(".jsonl"):
                    train_id = name[len("gap-") : -len(".jsonl")]
                    self._states.setdefault(train_id, None)

    def _trains_with_events(self) -> list[str]:
        return list(self._states)

    def _append(
        self, state: _TrainState, train_id: str, type_: str, payload: dict, at: float | None
    ) -> GapEvent:
        if at is None:
            raise InterlockError("裁决事件必须带时间戳", component="gap")
        event = self._journal(train_id).append(at, type_, payload)
        state.events.append(event)
        return event

    def _audit(self, kind: str, train_id: str, detail: dict) -> None:
        if self._audit_hook is not None:
            self._audit_hook(kind, train_id, detail)

    def _count(self, type_: str) -> int:
        with self._lock:
            return sum(
                1
                for state in self._states.values()
                if state is not None
                for event in state.events
                if event.type == type_
            )

    @staticmethod
    def _open_request_seq(events: list[GapEvent]) -> int | None:
        """从事件链纯派生当前挂起的复核请求（结论落盘即关闭此前请求）。"""

        open_seq: int | None = None
        for event in events:
            if event.type == REVIEW_REQUESTED:
                open_seq = event.seq
            elif event.type == REVIEW_RECORDED:
                open_seq = None
        return open_seq

    @staticmethod
    def _latest_review_seq(state: _TrainState) -> int | None:
        for event in reversed(state.events):
            if event.type == REVIEW_RECORDED:
                return event.seq
        return None

    # ---- 纯函数：从事件链计算裁决 --------------------------------------

    def _samples_at(self, events: list[GapEvent], t: float) -> dict[str, tuple[str, float, bool]]:
        """时刻 ``t`` 每路源的最新上报与健康状态。

        返回 ``observation / 上报时间 / 是否在保鲜窗口内``；故障源与从未上报的源
        不出现（``healthy`` 标记需要另由 :meth:`_health_at` 给出）。
        """

        latest: dict[str, tuple[str, float]] = {}
        faulty: dict[str, bool] = {}
        for event in events:
            if event.at > t:
                break
            if event.type == OBSERVATION:
                source_id = str(event.payload.get("source_id", ""))
                latest[source_id] = (str(event.payload.get("observation", "")), event.at)
                faulty[source_id] = False
            elif event.type == SOURCE_FAULT:
                source_id = str(event.payload.get("source_id", ""))
                faulty[source_id] = bool(event.payload.get("faulty", False))
        samples: dict[str, tuple[str, float, bool]] = {}
        for source in self.sources:
            if faulty.get(source.id, False):
                continue
            item = latest.get(source.id)
            if item is None:
                continue
            fresh = t - item[1] <= self.policy.freshness_s
            samples[source.id] = (item[0], item[1], fresh)
        return samples

    def _health_at(self, events: list[GapEvent], t: float) -> dict[str, bool]:
        """时刻 ``t`` 每路源的健康位（未被故障标记即健康）。"""

        faulty: dict[str, bool] = {}
        for event in events:
            if event.at > t:
                break
            if event.type == OBSERVATION:
                faulty[str(event.payload.get("source_id", ""))] = False
            elif event.type == SOURCE_FAULT:
                faulty[str(event.payload.get("source_id", ""))] = bool(
                    event.payload.get("faulty", False)
                )
        return {source.id: not faulty.get(source.id, False) for source in self.sources}

    def _candidate(
        self, samples: dict[str, tuple[str, float, bool]], has_evidence: bool = False
    ) -> tuple[str, float, float, float, dict[str, float]]:
        """返回 (候选, 覆盖率, clear 权重, blocked 权重, 各源参与份额)。

        ``has_evidence=False``（此刻之前没有任何一路上报过）与"有上报但权重不够"
        是两回事：前者是尚未探测（unknown），后者才是证据不足（insufficient）。
        """

        weights = {source.id: source.weight for source in self.sources}
        total = sum(weights.values())
        fresh = {sid: item for sid, item in samples.items() if item[2]}
        participating = {sid: weights[sid] for sid in fresh}
        active_weight = sum(participating.values())
        coverage = active_weight / total if total else 0.0
        shares = {
            sid: (participating[sid] / active_weight if active_weight else 0.0)
            for sid in participating
        }
        clear_w = sum(weights[sid] for sid, item in fresh.items() if item[0] == OBS_CLEAR)
        blocked_w = sum(weights[sid] for sid, item in fresh.items() if item[0] == OBS_BLOCKED)
        if not has_evidence:
            candidate = "unknown"
        elif coverage < self.policy.min_quorum:
            candidate = "insufficient"
        elif active_weight and clear_w / active_weight >= self.policy.agree_ratio:
            candidate = CLEAR
        elif active_weight and blocked_w / active_weight >= self.policy.agree_ratio:
            candidate = BLOCKED
        else:
            candidate = "split"
        return candidate, round(coverage, 9), clear_w, blocked_w, shares

    def _compute(self, state: _TrainState, train_id: str, at: float | None) -> Verdict:
        return self._compute_with(state, state.events, train_id, at)

    def _compute_with(
        self, state: _TrainState, events: list[GapEvent], train_id: str, at: float | None
    ) -> Verdict:
        moment = float(at) if at is not None else (events[-1].at if events else 0.0)
        observation_times = [
            event.at for event in events if event.type == OBSERVATION and event.at <= moment
        ]

        def candidate_at(t: float) -> str:
            samples_at = self._samples_at(events, t)
            has_evidence = any(ts <= t for ts in observation_times)
            return self._candidate(samples_at, has_evidence)[0]

        samples = self._samples_at(events, moment)
        candidate, coverage, clear_w, blocked_w, shares = self._candidate(
            samples, bool(observation_times)
        )

        # 在事件时间轴上回推：当前候选已连续保持多久。采样点除事件时刻和当前时刻
        # 外，还要包含每条上报"变陈旧"的时刻，否则覆盖率在两个事件之间静默变化时
        # （例如唯一一路上报过期）会把持续时长算过头。
        boundaries = {event.at for event in events if event.at <= moment}
        for event in events:
            if event.type != OBSERVATION:
                continue
            expiry = event.at + self.policy.freshness_s
            if expiry <= moment:
                boundaries.add(expiry)
        probe_times = sorted(boundaries | {moment})

        def duration_of(match: tuple[str, ...]) -> float:
            earliest = moment
            for ts in reversed(probe_times):
                if candidate_at(ts) in match:
                    earliest = ts
                else:
                    break
            return max(0.0, moment - earliest)

        if candidate in _DEFINITE:
            held_for = duration_of((candidate,))
            auto_label = candidate if held_for >= self.policy.confirm_s else HOLD
            auto_held = held_for
        elif candidate in ("split", "insufficient"):
            undetermined_for = duration_of(("split", "insufficient"))
            auto_held = undetermined_for
            auto_label = REVIEW if undetermined_for >= self.policy.hold_s else HOLD
        else:
            auto_held = 0.0
            auto_label = UNKNOWN

        # 人工结论（取最近一次复核记录）。
        override_view: dict = {}
        source = AUTO
        label = BLOCKED if auto_label != CLEAR else CLEAR
        review_record = self._effective_review(events, moment)
        if review_record is not None:
            event, consumed = review_record
            decision = str(event.payload.get("decision", ""))
            if decision == OBS_BLOCKED:
                # 人工拦停：粘性保持，直到下一次人工结论改变。
                source = MANUAL_BLOCKED
                label = BLOCKED
                override_view = self._override_view(event, consumed, moment)
            elif decision == OBS_CLEAR and not consumed and moment <= float(
                event.payload.get("expires_at", 0.0)
            ):
                # 人工放行：一次性、有效期内。
                source = MANUAL_CLEAR
                label = CLEAR
                override_view = self._override_view(event, consumed, moment)

        review_required = auto_label == REVIEW and source != MANUAL_CLEAR
        # 人工放行之后若仍未恢复自动一致，新的打架仍应再次升级：复核请求只在
        # 已被记录的结论覆盖时视为关闭。
        open_request_seq = self._open_request_seq(events)
        if source == MANUAL_CLEAR and auto_label == REVIEW and open_request_seq is not None:
            review_required = True

        overdue = None
        if open_request_seq is not None:
            request_at = events[open_request_seq].at
            overdue = max(0.0, moment - request_at - self.policy.review_timeout_s)

        source_views = self._source_views(events, samples, shares, moment)
        tail_seq = events[-1].seq if events else -1
        tail_hash = events[-1].hash if events else ""
        basis = {
            "train_id": train_id,
            "at": round(moment, 6),
            "auto_label": auto_label,
            "candidate": candidate,
            "coverage": round(coverage, 6),
            "clear_weight": round(clear_w, 6),
            "blocked_weight": round(blocked_w, 6),
            "source": source,
            "label": label,
            "override_seq": override_view.get("seq") if override_view else None,
            "tail_seq": tail_seq,
            "tail_hash": tail_hash,
        }
        basis_hash = hashlib.sha256(
            json.dumps(basis, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return Verdict(
            train_id=train_id,
            at=moment,
            auto_label=auto_label,
            label=label,
            source=source,
            review_required=review_required,
            review_request_seq=open_request_seq,
            review_overdue_s=overdue,
            candidate=candidate,
            held_for_s=auto_held,
            coverage=coverage,
            clear_weight=clear_w,
            blocked_weight=blocked_w,
            sources=tuple(source_views),
            override=override_view,
            tail_seq=tail_seq,
            tail_hash=tail_hash,
            basis_hash=basis_hash,
        )

    def _effective_review(
        self, events: list[GapEvent], moment: float
    ) -> tuple[GapEvent, bool] | None:
        """截至 ``moment`` 最近一次人工复核记录，以及它是否已被发车决定消耗。"""

        review: GapEvent | None = None
        for event in events:
            if event.at > moment:
                break
            if event.type == REVIEW_RECORDED:
                review = event
        if review is None:
            return None
        consumed = any(
            decision.type == DEPARTURE_DECISION
            and review.at <= decision.at <= moment
            and decision.payload.get("consumed_override") == review.seq
            for decision in events
        )
        return review, consumed

    def _override_view(self, event: GapEvent, consumed: bool, moment: float) -> dict:
        expires_at = event.payload.get("expires_at")
        return {
            "seq": event.seq,
            "decision": str(event.payload.get("decision", "")),
            "reviewer": str(event.payload.get("reviewer", "")),
            "role": str(event.payload.get("role", "")),
            "recorded_at": event.at,
            "expires_at": expires_at,
            "remaining_s": None
            if expires_at is None
            else round(max(0.0, float(expires_at) - moment), 3),
            "consumed": consumed,
        }

    def _source_views(
        self,
        events: list[GapEvent],
        samples: dict[str, tuple[str, float, bool]],
        shares: dict[str, float],
        moment: float,
    ) -> list[SourceView]:
        health = self._health_at(events, moment)
        views: list[SourceView] = []
        for source in self.sources:
            item = samples.get(source.id)
            if item is not None:
                observation, at, fresh = item
                age = moment - at
            else:
                # 故障源或已过期：保留最后一次上报值，仅用于展示，不参与融合。
                observation, age, fresh = "", None, False
                for event in events:
                    if event.at > moment:
                        break
                    if (
                        event.type == OBSERVATION
                        and str(event.payload.get("source_id", "")) == source.id
                    ):
                        observation = str(event.payload.get("observation", ""))
                        age = moment - event.at
            views.append(
                SourceView(
                    id=source.id,
                    name=source.name,
                    weight=source.weight,
                    observation=observation,
                    age_s=None if age is None else round(age, 3),
                    healthy=health.get(source.id, True),
                    participating=bool(item is not None and fresh),
                    share=shares.get(source.id, 0.0),
                )
            )
        return views


# 旧名字保留：历史调用方与审计脚本仍以 GapRegistry 引用融合裁决器。
GapRegistry = GapFusion
