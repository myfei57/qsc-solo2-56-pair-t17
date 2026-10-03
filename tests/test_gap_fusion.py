"""多源间隙探测的融合裁决：权重、持续时长、证据窗口、人工复核与重放。"""

from __future__ import annotations

from psdline.clock import ManualClock
from psdline.console.api import Api
from psdline.gap import BLOCKED, CLEAR, PENDING, REVIEW, GapRegistry
from psdline.store import JsonStore

WEIGHTS = {"lidar": 2.0, "vision": 1.0}


def _registry(tmp_path, clock, **overrides) -> GapRegistry:
    params = {
        "weights": dict(WEIGHTS),
        "confirm_s": 0.0,
        "observation_ttl_s": 30.0,
        "window_s": 3.0,
        "review_ttl_s": 120.0,
    }
    params.update(overrides)
    return GapRegistry(JsonStore(str(tmp_path)), clock=clock, **params)


def test_unanimous_sources_decide_immediately(tmp_path):
    clock = ManualClock()
    gaps = _registry(tmp_path, clock)
    gaps.observe("T01", "lidar", True, clock.now())
    gaps.observe("T01", "vision", True, clock.now())
    verdict = gaps.verdict("T01")
    assert verdict.state == CLEAR and verdict.reason == "unanimous"
    assert gaps.clear("T01") is True and gaps.held_ids() == []


def test_disagreement_waits_window_then_weights_decide(tmp_path):
    clock = ManualClock()
    gaps = _registry(tmp_path, clock)
    gaps.observe("T01", "lidar", True, clock.now())
    gaps.observe("T01", "vision", False, clock.now())

    verdict = gaps.verdict("T01")
    assert verdict.state == PENDING and verdict.reason == "disagree_window"
    assert gaps.clear("T01") is False  # 窗口内不放行，但也不锁死
    assert gaps.blocked_ids() == [] and gaps.held_ids() == ["T01"]

    clock.advance(4.0)  # 窗口（3s）耗尽：lidar 权重 2 压过 vision 权重 1
    verdict = gaps.verdict("T01")
    assert verdict.state == CLEAR and verdict.reason == "weighted"
    assert verdict.contested is True
    assert verdict.clear_weight > verdict.blocked_weight


def test_duration_ramps_effective_weight(tmp_path):
    clock = ManualClock()
    gaps = _registry(tmp_path, clock, confirm_s=4.0, window_s=1.0)
    gaps.observe("T01", "vision", True, clock.now())  # 低权重但读数持续得久
    clock.advance(10.0)
    gaps.observe("T01", "lidar", False, clock.now())  # 高权重但刚改口
    clock.advance(2.0)  # 窗口已过：lidar 有效权重 2×(2/4)=1，vision 爬满 1

    verdict = gaps.verdict("T01")
    assert verdict.state == REVIEW and verdict.reason == "tie"
    assert verdict.clear_weight == verdict.blocked_weight == 1.0

    clock.advance(2.0)  # lidar 持续 4s 爬满：2 对 1，blocked 胜出
    verdict = gaps.verdict("T01")
    assert verdict.state == BLOCKED and verdict.reason == "weighted"


def test_tie_requires_manual_review_then_review_expires(tmp_path):
    clock = ManualClock()
    gaps = _registry(tmp_path, clock, weights={"a": 1.0, "b": 1.0}, review_ttl_s=10.0)
    gaps.observe("T01", "a", True, clock.now())
    gaps.observe("T01", "b", False, clock.now())
    clock.advance(5.0)
    assert gaps.verdict("T01").state == REVIEW

    gaps.review("T01", True, "dispatcher", clock.now())
    verdict = gaps.verdict("T01")
    assert verdict.state == CLEAR and verdict.origin == "manual"
    assert gaps.clear("T01") is True

    clock.advance(11.0)  # 复核有效期（10s）过了，探测未静默，平局依旧，回到待复核
    assert gaps.verdict("T01").state == REVIEW
    assert gaps.held_ids() == ["T01"]


def test_review_overrides_automatic_blocked(tmp_path):
    clock = ManualClock()
    gaps = _registry(tmp_path, clock, review_ttl_s=10.0)
    gaps.observe("T01", "lidar", False, clock.now())
    gaps.observe("T01", "vision", False, clock.now())
    assert gaps.verdict("T01").state == BLOCKED

    gaps.review("T01", True, "dispatcher", clock.now())
    assert gaps.verdict("T01").state == CLEAR  # 复核盖过一致的自动结论

    clock.advance(11.0)
    assert gaps.verdict("T01").state == BLOCKED  # 过期后自动结论恢复


def test_stale_source_drops_out_of_fusion(tmp_path):
    clock = ManualClock()
    gaps = _registry(tmp_path, clock)
    gaps.observe("T01", "vision", False, clock.now())
    clock.advance(40.0)  # vision 静默超过 30s，退出融合
    gaps.observe("T01", "lidar", True, clock.now())

    verdict = gaps.verdict("T01")
    assert verdict.state == CLEAR and verdict.reason == "unanimous"
    health = gaps.source_health("T01")
    assert health["vision"]["stale"] is True and health["vision"]["effective"] == 0.0
    assert health["lidar"]["stale"] is False

    clock.advance(31.0)  # 全部静默：没有证据，裁决为未知
    verdict = gaps.verdict("T01")
    assert verdict.state == "" and verdict.reason == "all_stale"
    assert gaps.clear("T01") is False


def test_restart_recovers_verdict_from_journals(tmp_path):
    clock = ManualClock()
    gaps = _registry(tmp_path, clock, review_ttl_s=10.0)
    gaps.observe("T01", "lidar", False, clock.now())
    gaps.review("T01", True, "dispatcher", clock.now())

    restored = _registry(tmp_path, clock, review_ttl_s=10.0)  # 同一存储目录重建，模拟重启
    verdict = restored.verdict("T01")
    assert verdict.state == CLEAR and verdict.origin == "manual"
    assert [item.source for item in restored.observations("T01")] == ["lidar"]
    assert len(restored.reviews("T01")) == 1

    clock.advance(11.0)  # 重启后时序继续生效：复核过期，回落到 blocked
    assert restored.verdict("T01").state == BLOCKED


def test_replay_and_verify(tmp_path):
    clock = ManualClock()
    gaps = _registry(tmp_path, clock)
    gaps.observe("T01", "lidar", True, clock.now())
    gaps.observe("T01", "vision", False, clock.now())  # 进入证据窗口
    clock.advance(4.0)
    gaps.observe("T01", "vision", True, clock.now())  # 改口，恢复一致
    gaps.review("T01", True, "dispatcher", clock.now())

    timeline = gaps.replay("T01")
    states = [(item["state"], item["origin"]) for item in timeline]
    # 重放只依赖流水，与读取时刻无关；复核把最后一跳的来源换成人工。
    assert states == [(CLEAR, "auto"), (PENDING, "auto"), (CLEAR, "auto"), (CLEAR, "manual")]

    checked = gaps.verify("T01")
    assert checked["ok"] is True and checked["issues"] == []
    assert checked["cached"]["state"] == checked["recomputed"]["state"]


def test_pending_verdict_blocks_depart(platform, clock):
    platform.start()
    platform.arrive("T01", "ST1", 0)
    platform.observe_gap("T01", "lidar", True)
    platform.observe_gap("T01", "vision", False)

    attempt = platform.depart("T01", "ST1")
    assert attempt["ok"] is False and attempt["violations"] == ["间隙多源探测未收敛"]

    clock.advance(4.0)  # 窗口耗尽，lidar(2) 压过 vision(1)
    departure = platform.depart("T01", "ST1")
    assert departure["ok"] is True
    gap_steps = [step for step in departure["steps"] if step["component"] == "gap"]
    assert gap_steps[0]["detail"]["state"] == CLEAR


def test_review_unblocks_depart(platform, clock):
    platform.start()
    platform.arrive("T01", "ST1", 0)
    platform.observe_gap("T01", "lidar", False)
    attempt = platform.depart("T01", "ST1")
    assert attempt["ok"] is False and attempt["violations"] == ["间隙探测判定有异物"]

    reviewed = platform.review_gap("T01", True, "dispatcher")
    assert reviewed["verdict"]["state"] == CLEAR
    departure = platform.depart("T01", "ST1")
    assert departure["ok"] is True

    kinds = [entry.kind for entry in platform.audit.list()]
    assert "gap_observe" in kinds and "gap_review" in kinds


def test_missing_evidence_blocks_depart(platform):
    platform.start()
    platform.arrive("T01", "ST1", 0)
    attempt = platform.depart("T01", "ST1")
    assert attempt["ok"] is False and attempt["violations"] == ["间隙探测未确认"]


def test_gap_api_observes_reviews_and_reports(platform):
    api = Api(platform)
    api.handle("POST", "/api/arrive", {}, {"train_id": "T01", "station_id": "ST1", "position_m": 0})
    status, observed = api.handle(
        "POST", "/api/gap/observe", {}, {"train_id": "T01", "source": "lidar", "clear": False}
    )
    assert status == 200 and observed["verdict"]["state"] == BLOCKED

    status, view = api.handle("GET", "/api/gaps/T01", {}, {})
    assert status == 200 and view["state"] == BLOCKED
    assert view["sources"]["lidar"]["clear"] is False
    assert view["verify"]["ok"] is True

    status, reviewed = api.handle(
        "POST", "/api/gap/review", {}, {"train_id": "T01", "clear": True, "reviewer": "op1"}
    )
    assert status == 200 and reviewed["verdict"]["state"] == CLEAR
    status, view = api.handle("GET", "/api/gaps/T01", {}, {})
    assert view["clear"] is True and view["origin"] == "manual"

    status, payload = api.handle("POST", "/api/gap/observe", {}, {"train_id": "T01"})
    assert status == 400  # 缺少探测源标识
