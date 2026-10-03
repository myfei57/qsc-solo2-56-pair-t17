"""间隙探测。

发车前必须先确认站台与列车之间没有异物。探测是多源的：每路探测源各自上报
原始读数并带自己的可信度（权重），融合裁决（:mod:`psdline.gap.fusion`）按
权重与读数持续时长给出唯一结论；结论不一致时先给一个短证据窗口，窗口耗尽
仍定不下来就要求人工复核，复核结论在有效期内盖过自动裁决。

探测流水与复核流水只追加不修改，裁决是流水的纯函数——探测、发车放行与审计
读的都是这同一份裁决，重启恢复与重放也从流水重建。未对位的列车不接受人工
确认。
"""

from __future__ import annotations

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
from .gap import DEFAULT_SOURCE, GapRegistry, ProbeResult
from .status import gap_status, gap_summary

__all__ = [
    "BLOCKED",
    "CLEAR",
    "DEFAULT_SOURCE",
    "PENDING",
    "REVIEW",
    "UNKNOWN",
    "GapRegistry",
    "Observation",
    "ProbeResult",
    "Review",
    "Verdict",
    "adjudicate",
    "gap_status",
    "gap_summary",
]
