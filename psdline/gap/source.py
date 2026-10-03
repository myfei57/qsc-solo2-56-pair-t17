"""间隙探测源配置。

每一路探测（激光雷达、红外对射、人工瞭望台……）都带自己的可信度权重。权重只在
融合裁决时按"在线且健康"的源重新归一化，因此某一路故障或上报过期不会把它的
权重白白丢掉，也不会让它静默左右结论。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigError

# 角色
OPERATOR = "operator"        # 值班员：只能在自动判定 clear 时人工确认
SUPERVISOR = "supervisor"  # 督导：可以在打架时人工放行（盖过自动判定）

# 观测值
OBS_CLEAR = "clear"
OBS_BLOCKED = "blocked"


@dataclass(frozen=True)
class GapSource:
    """一路探测源。"""

    id: str
    name: str
    weight: float
    kind: str = "sensor"

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "weight": self.weight, "kind": self.kind}


def parse_sources(payload: object) -> tuple[GapSource, ...]:
    """从配置构造探测源列表，权重必须全为正且不全为零。"""

    if not isinstance(payload, (list, tuple)) or not payload:
        raise ConfigError("间隙探测源至少配置一路", component="config")
    sources: list[GapSource] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ConfigError("探测源配置必须是对象", component="config")
        source_id = str(item.get("id", ""))
        if not source_id:
            raise ConfigError("探测源 id 不能为空", component="config")
        if source_id in seen:
            raise ConfigError(f"探测源 id 重复: {source_id}", component="config")
        weight = float(item.get("weight", 0.0))
        if weight <= 0:
            raise ConfigError(f"探测源 {source_id} 权重必须为正", component="config")
        sources.append(
            GapSource(
                id=source_id,
                name=str(item.get("name", source_id)),
                weight=weight,
                kind=str(item.get("kind", "sensor")),
            )
        )
        seen.add(source_id)
    return tuple(sources)


def default_sources() -> tuple[GapSource, ...]:
    """默认三路：激光雷达（高可信）、红外对射、人工瞭望台。"""

    return (
        GapSource("lidar", "激光雷达", 0.45),
        GapSource("ir", "红外对射", 0.35),
        GapSource("manual", "人工瞭望台", 0.20, kind="watch"),
    )
