"""测试夹具：所有用例都从干净存储和可推进时钟起步。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psdline.app import Platform, build_platform  # noqa: E402
from psdline.clock import ManualClock  # noqa: E402
from psdline.config import default_config  # noqa: E402


@pytest.fixture()
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture()
def platform(tmp_path, clock) -> Platform:
    return build_platform(default_config(), clock=clock, storage_root=str(tmp_path / "state"))


@pytest.fixture()
def online(platform, clock) -> Platform:
    """已到站、已开门、三路探测一致且结论已落定的平台。"""

    platform.start()
    platform.arrive("T01", "ST1", 0)
    for source_id in ("lidar", "ir", "manual"):
        platform.report_gap("T01", source_id, True)
    clock.advance(platform.config.gap.confirm_s)
    platform.gaps.pump("T01", clock.now())
    return platform
