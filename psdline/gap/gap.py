"""间隙探测登记与多源融合裁决。

每列车的探测与复核各存一条只追加的 JSONL 流水（历史不可变），当前裁决是
流水的派生缓存：任何时刻删掉缓存都能用 :func:`adjudicate` 从流水原样重算，
重放因此只是"再算一遍"。探测、发车放行与审计读的都是 ``verdict()`` 这一份
裁决；裁决状态变化时往裁决流水里追加一条，作为运行期实际所见的历史。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..clock import Clock
from ..errors import InterlockError
from ..store import JsonlJournal, JsonStore
from .fusion import (
    BLOCKED,
    CLEAR,
    PENDING,
    REVIEW,
    UNKNOWN,
    Observation,
    Review,
    Verdict,
    adjudicate,
)

DEFAULT_SOURCE = "detector"


@dataclass(frozen=True)
class ProbeResult:
    """一次间隙探测结果。"""

    train_id: str
    clear: bool
    confirmed: bool
    at: float

    def as_dict(self) -> dict:
        return {
            "train_id": self.train_id,
            "clear": self.clear,
            "confirmed": self.confirmed,
            "at": self.at,
        }


class GapRegistry:
    """间隙探测登记表：多源流水 + 融合裁决缓存。"""

    def __init__(
        self,
        store: JsonStore,
        clock: Clock | None = None,
        *,
        weights: Mapping[str, float] | None = None,
        default_weight: float = 1.0,
        confirm_s: float = 4.0,
        observation_ttl_s: float = 30.0,
        window_s: float = 3.0,
        review_ttl_s: float = 120.0,
    ) -> None:
        self._store = store
        self._clock = clock
        self._weights = dict(weights or {})
        self._default_weight = float(default_weight)
        self._confirm_s = float(confirm_s)
        self._observation_ttl_s = float(observation_ttl_s)
        self._window_s = float(window_s)
        self._review_ttl_s = float(review_ttl_s)
        self._checks = 0
        self._probes = 0

    @property
    def checks(self) -> int:
        """返回累计确认次数。"""

        return self._checks

    @property
    def probes(self) -> int:
        """返回累计探测次数。"""

        return self._probes

    # ---- 流水 -----------------------------------------------------------

    def _obs_journal(self, train_id: str) -> JsonlJournal:
        return JsonlJournal(self._store.root, f"gap-obs-{train_id}")

    def _review_journal(self, train_id: str) -> JsonlJournal:
        return JsonlJournal(self._store.root, f"gap-review-{train_id}")

    def _verdict_journal(self, train_id: str) -> JsonlJournal:
        return JsonlJournal(self._store.root, f"gap-verdict-{train_id}")

    def observations(self, train_id: str) -> list[Observation]:
        """按（时刻, 序号）返回一列车的全部探测流水。"""

        records = [Observation.from_dict(item) for item in self._obs_journal(train_id).read_all()]
        return sorted(records, key=lambda item: (item.at, item.seq))

    def reviews(self, train_id: str) -> list[Review]:
        """按（时刻, 序号）返回一列车的全部复核流水。"""

        records = [Review.from_dict(item) for item in self._review_journal(train_id).read_all()]
        return sorted(records, key=lambda item: (item.at, item.seq))

    # ---- 裁决 -----------------------------------------------------------

    def _now(self, train_id: str, at: float | None) -> float:
        if at is not None:
            return at
        if self._clock is not None:
            return self._clock.now()
        evidence = [item.at for item in self.observations(train_id)]
        evidence += [item.at for item in self.reviews(train_id)]
        return max(evidence) if evidence else 0.0

    def _adjudicate(self, train_id: str, now: float) -> Verdict:
        return adjudicate(
            train_id,
            self.observations(train_id),
            self.reviews(train_id),
            now,
            weights=self._weights,
            default_weight=self._default_weight,
            confirm_s=self._confirm_s,
            observation_ttl_s=self._observation_ttl_s,
            window_s=self._window_s,
        )

    @staticmethod
    def _verdict_doc_id(train_id: str) -> str:
        return f"gap/{train_id}/verdict"

    def verdict(self, train_id: str, at: float | None = None) -> Verdict:
        """返回当前裁决；状态变化时追加裁决流水并刷新缓存。"""

        now = self._now(train_id, at)
        verdict = self._adjudicate(train_id, now)
        cached = self._store.read(self._verdict_doc_id(train_id))
        changed = not (
            isinstance(cached, dict)
            and cached.get("state") == verdict.state
            and cached.get("origin") == verdict.origin
            and cached.get("reason") == verdict.reason
        )
        if changed:
            record = verdict.as_dict()
            record["observations"] = self._obs_journal(train_id).count()
            record["reviews"] = self._review_journal(train_id).count()
            self._verdict_journal(train_id).append(record)
            self._store.write(self._verdict_doc_id(train_id), record)
        return verdict

    def clear(self, train_id: str, at: float | None = None) -> bool:
        """查询间隙当前是否裁决为无异常。"""

        return self.verdict(train_id, at).state == CLEAR

    def state(self, train_id: str, at: float | None = None) -> str:
        """返回当前裁决状态（``clear``/``blocked``/``pending``/``review``/空串）。"""

        return self.verdict(train_id, at).state

    def _known_train_ids(self) -> list[str]:
        found: set[str] = set()
        for doc_id in self._store.list_ids():
            if doc_id.startswith("gap/") and doc_id.endswith("/verdict"):
                found.add(doc_id.split("/")[1])
        return sorted(found)

    def blocked_ids(self, at: float | None = None) -> list[str]:
        """返回当前被裁决为有异物（``blocked``）的列车。"""

        return [
            train_id
            for train_id in self._known_train_ids()
            if self.verdict(train_id, at).state == BLOCKED
        ]

    def held_ids(self, at: float | None = None) -> list[str]:
        """返回当前不能放行的列车（``blocked``/``pending``/``review``）。"""

        return [
            train_id
            for train_id in self._known_train_ids()
            if self.verdict(train_id, at).state in (BLOCKED, PENDING, REVIEW)
        ]

    def source_health(self, train_id: str, at: float | None = None) -> dict:
        """返回每路探测源的最新读数、权重与静默状态。"""

        now = self._now(train_id, at)
        return dict(self._adjudicate(train_id, now).sources)

    # ---- 重放与核对 -----------------------------------------------------

    def replay(self, train_id: str) -> list[dict]:
        """只从流水重放裁决时间线（不读缓存），返回状态变迁序列。

        按（时刻, 序号）逐条折叠探测与复核流水，每条证据到达后重新裁决，
        因此重放复现的是运行期实际经历过的变迁，与谁在何时读取无关。
        """

        observations = self.observations(train_id)
        reviews = self.reviews(train_id)
        events = [(item.at, 0, item.seq) for item in observations]
        events += [(item.at, 1, item.seq) for item in reviews]
        timeline: list[dict] = []
        last_key: tuple | None = None
        for moment, kind, seq in sorted(events):
            verdict = adjudicate(
                train_id,
                [item for item in observations if (item.at, 0, item.seq) <= (moment, kind, seq)],
                [item for item in reviews if (item.at, 1, item.seq) <= (moment, kind, seq)],
                moment,
                weights=self._weights,
                default_weight=self._default_weight,
                confirm_s=self._confirm_s,
                observation_ttl_s=self._observation_ttl_s,
                window_s=self._window_s,
            )
            key = (verdict.state, verdict.origin, verdict.reason)
            if key != last_key:
                timeline.append(verdict.as_dict())
                last_key = key
        return timeline

    def verify(self, train_id: str, at: float | None = None) -> dict:
        """从流水重算当前裁决并与缓存对照，检验缓存是否可重现。"""

        now = self._now(train_id, at)
        recomputed = self._adjudicate(train_id, now)
        cached = self._store.read(self._verdict_doc_id(train_id))
        issues: list[str] = []
        if not isinstance(cached, dict):
            issues.append("no_cache")
        else:
            for key in ("state", "origin", "reason"):
                if cached.get(key) != getattr(recomputed, key):
                    issues.append(f"{key}_mismatch")
            if int(cached.get("observations", -1)) != self._obs_journal(train_id).count():
                issues.append("observations_mismatch")
            if int(cached.get("reviews", -1)) != self._review_journal(train_id).count():
                issues.append("reviews_mismatch")
        return {
            "ok": not issues,
            "train_id": train_id,
            "issues": issues,
            "recomputed": recomputed.as_dict(),
            "cached": cached if isinstance(cached, dict) else None,
        }

    # ---- 登记 -----------------------------------------------------------

    def observe(self, train_id: str, source: str, clear: bool, at: float) -> Observation:
        """登记一路探测源的一次原始读数。"""

        if not source:
            raise InterlockError("探测源标识不能为空", component="gap")
        journal = self._obs_journal(train_id)
        observation = Observation(
            train_id=train_id, source=source, clear=clear, at=at, seq=journal.count() + 1
        )
        journal.append(observation.as_dict())
        self._probes += 1
        self.verdict(train_id, at)
        return observation

    def probe(self, train_id: str, clear: bool, at: float) -> ProbeResult:
        """登记一次默认源的探测结果（可含可疑）。"""

        self.observe(train_id, DEFAULT_SOURCE, clear, at)
        return ProbeResult(
            train_id=train_id, clear=clear, confirmed=self.clear(train_id, at), at=at
        )

    def review(self, train_id: str, clear: bool, reviewer: str, at: float) -> Review:
        """登记一次人工复核；有效期内盖过自动裁决。"""

        if not reviewer:
            raise InterlockError("复核人标识不能为空", component="gap")
        journal = self._review_journal(train_id)
        review = Review(
            train_id=train_id,
            clear=clear,
            reviewer=reviewer,
            at=at,
            until=at + self._review_ttl_s,
            seq=journal.count() + 1,
        )
        journal.append(review.as_dict())
        self.verdict(train_id, at)
        return review

    def check(self, train_id: str, at: float) -> ProbeResult:
        """人工确认间隙无异常；未对位的列车直接拒绝。"""

        if not self._store.exists(f"train/{train_id}/align"):
            raise InterlockError(f"列车 {train_id} 尚未对位，无法确认间隙", component="gap")
        self.review(train_id, True, "operator", at)
        self._checks += 1
        return ProbeResult(train_id=train_id, clear=True, confirmed=True, at=at)
