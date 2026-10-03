"""间隙裁决事件与不可变链式流水。

融合裁决器采用事件溯源：所有输入（探测上报、人工复核、裁决落定、放行决定）都只
追加到 ``gap-<train>.jsonl``，永不修改。每条事件携带前一条的 SHA-256，形成
哈希链；任何一条被篡改或删除都会在重放时被发现。

崩溃恢复：追加最后一条时掉电会留下"撕裂行"（不是合法 JSON）。它属于尚未落定的
写入，重放时安全截断（报告 ``truncated``）；而链中间断链（前序哈希对不上）说明
历史被改过，直接拒绝静默启动——历史不可变是审计底线。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field, fields
from typing import Any

from ..errors import StorageError

GENESIS = "0" * 64

# 事件类型
OBSERVATION = "observation"          # 一路探测上报
SOURCE_FAULT = "source_fault"        # 一路探测被标记故障/恢复
REVIEW_REQUESTED = "review_requested"  # 裁决器请求人工复核
REVIEW_RECORDED = "review_recorded"    # 人工复核结论
ADJUDICATION = "adjudication"        # 自动裁决落定/升级
DEPARTURE_DECISION = "departure_decision"  # 发车放行读到的裁决

_KNOWN_TYPES = {
    OBSERVATION,
    SOURCE_FAULT,
    REVIEW_REQUESTED,
    REVIEW_RECORDED,
    ADJUDICATION,
    DEPARTURE_DECISION,
}


@dataclass(frozen=True)
class GapEvent:
    """一条间隙裁决事件（追加后不可变）。"""

    seq: int
    at: float
    train_id: str
    type: str
    payload: dict = field(default_factory=dict)
    prev_hash: str = GENESIS
    hash: str = ""

    def as_dict(self) -> dict:
        return {
            "seq": self.seq,
            "at": self.at,
            "train_id": self.train_id,
            "type": self.type,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "GapEvent":
        names = {item.name for item in fields(cls)}
        return cls(**{key: raw[key] for key in names if key in raw})


def canonical(seq: int, at: float, train_id: str, type_: str, payload: dict, prev_hash: str) -> bytes:
    """事件的规范字节串；键序固定，保证重放算出的哈希一致。"""

    body = [
        seq,
        round(float(at), 6),
        train_id,
        type_,
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
        prev_hash,
    ]
    return ("␟".join(str(item) for item in body)).encode("utf-8")


def digest(seq: int, at: float, train_id: str, type_: str, payload: dict, prev_hash: str) -> str:
    return hashlib.sha256(canonical(seq, at, train_id, type_, payload, prev_hash)).hexdigest()


class GapEventJournal:
    """单列车的追加式哈希链事件流水。"""

    def __init__(self, root: str, train_id: str) -> None:
        self._root = os.path.abspath(root)
        self._train_id = train_id
        self._path = os.path.join(self._root, f"gap-{train_id}.jsonl")
        self._lock = threading.RLock()
        os.makedirs(self._root, exist_ok=True)

    @property
    def path(self) -> str:
        return self._path

    def append(self, at: float, type_: str, payload: dict) -> GapEvent:
        """追加一条事件并 fsync，返回带哈希的事件。"""

        if type_ not in _KNOWN_TYPES:
            raise StorageError(f"未知间隙事件类型: {type_}", component="gap")
        with self._lock:
            events, meta = self._read_locked()
            if meta["broken_chain"]:
                raise StorageError(
                    f"列车 {self._train_id} 间隙流水哈希链已断，拒绝追加",
                    component="gap",
                )
            seq = len(events)
            prev_hash = events[-1].hash if events else GENESIS
            clean = _json_clean(payload)
            new_hash = digest(seq, at, self._train_id, type_, clean, prev_hash)
            event = GapEvent(seq, float(at), self._train_id, type_, clean, prev_hash, new_hash)
            try:
                with open(self._path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as error:
                raise StorageError(f"追加间隙流水失败: {error}", component="gap") from error
            return event

    def replay(self) -> tuple[list[GapEvent], dict]:
        """读取并校验整条链；返回事件与校验元信息。"""

        with self._lock:
            return self._read_locked()

    def exists(self) -> bool:
        return os.path.isfile(self._path)

    # ---- 内部 ---------------------------------------------------------

    def _read_locked(self) -> tuple[list[GapEvent], dict]:
        meta = {
            "truncated_lines": 0,
            "truncated_at": None,
            "broken_chain": False,
            "broken_at": None,
            "verified": 0,
        }
        if not os.path.isfile(self._path):
            return [], meta
        lines: list[str] = []
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
        except OSError as error:
            raise StorageError(f"读取间隙流水失败: {error}", component="gap") from error
        events: list[GapEvent] = []
        for index, line in enumerate(lines):
            last_line = index == len(lines) - 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                # 只有最后一行允许是"追加途中掉电"的半截写入，安全截断。
                if last_line:
                    meta["truncated_lines"] += 1
                    meta["truncated_at"] = index + 1
                    break
                meta["broken_chain"] = True
                meta["broken_at"] = index + 1
                break
            if not isinstance(raw, dict):
                meta["broken_chain"] = True
                meta["broken_at"] = index + 1
                break
            try:
                event = GapEvent.from_dict(raw)
            except (TypeError, KeyError):
                meta["broken_chain"] = True
                meta["broken_at"] = index + 1
                break
            expected_prev = events[-1].hash if events else GENESIS
            recomputed = digest(
                event.seq, event.at, event.train_id, event.type, event.payload, event.prev_hash
            )
            if (
                event.seq != index
                or event.prev_hash != expected_prev
                or event.hash != recomputed
                or event.train_id != self._train_id
            ):
                meta["broken_chain"] = True
                meta["broken_at"] = event.seq
                break
            events.append(event)
            meta["verified"] += 1
        if meta["truncated_lines"]:
            self._truncate_to(len(events))
        return events, meta

    def _truncate_to(self, count: int) -> None:
        """把流水截断回前 ``count`` 条（丢弃未完成的尾部写入）。"""

        kept: list[GapEvent] = []
        if not os.path.isfile(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    kept.append(GapEvent.from_dict(json.loads(line)))
                    if len(kept) >= count:
                        break
            with open(self._path, "w", encoding="utf-8") as handle:
                for event in kept:
                    handle.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, json.JSONDecodeError, TypeError, KeyError) as error:
            raise StorageError(f"截断间隙流水失败: {error}", component="gap") from error


def _json_clean(value: Any) -> Any:
    """把负载规范成可 JSON 序列化、键序稳定的形式。"""

    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
