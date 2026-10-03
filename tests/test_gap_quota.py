"""多源融合间隙裁决与运行配额。

融合裁决测试覆盖：加权一致、持续时间落定、冲突证据窗口与升级复核、部分失败
剔除、保鲜过期、人工复核盖过自动（角色、有效期、一次性、粘性拦停）、不可变
哈希链、撕裂恢复、篡改拒绝、重放一致与审计指纹核对。
"""

from __future__ import annotations

import glob
import json
import os

import pytest

from psdline.errors import InterlockError, RailError
from psdline.gap import (
    FusionPolicy,
    GapFusion,
    GapSource,
    OPERATOR,
    SUPERVISOR,
    audit_timeline,
    gap_status,
    gap_summary,
    verify_all,
    verify_train,
)
from psdline.quota import QuotaRegistry
from psdline.store import JsonStore

SOURCES = (
    GapSource("lidar", "激光雷达", 0.45),
    GapSource("ir", "红外对射", 0.35),
    GapSource("manual", "人工瞭望台", 0.20, kind="watch"),
)


@pytest.fixture()
def fusion(tmp_path):
    store = JsonStore(str(tmp_path / "state"))
    engine = GapFusion(store, sources=SOURCES, policy=FusionPolicy())
    return engine, store


def _align(store: JsonStore, train_id: str) -> None:
    store.write(
        f"train/{train_id}/align",
        {"train_id": train_id, "position_m": 0, "durable": True, "at": 0.0},
    )


def _report_clear_all(engine: GapFusion, train_id: str, at: float) -> None:
    for source in SOURCES:
        engine.report(train_id, source.id, True, at)


# ---- 加权与持续时间 ----------------------------------------------------


def test_no_evidence_is_unknown_and_blocks(fusion):
    engine, _store = fusion
    verdict = engine.pump("T01", 1.0)
    assert verdict.auto_label == "unknown"
    assert verdict.can_depart is False


def test_unanimous_needs_confirm_duration(fusion):
    engine, store = fusion
    _align(store, "T01")
    _report_clear_all(engine, "T01", 1.0)
    assert engine.pump("T01", 3.9).auto_label == "hold"
    assert engine.pump("T01", 3.9).can_depart is False
    verdict = engine.pump("T01", 5.0)
    assert verdict.auto_label == "clear"
    assert verdict.can_depart is True


def test_weighted_split_does_not_follow_single_conservative_source(fusion):
    # 激光（0.45）说有异物，红外（0.35）+ 瞭望（0.20）说无：0.55 对 0.45，
    # 不满足 2/3 一致阈值 -> 冲突（不是直接按最保守一路拦死，也不是放行）。
    engine, store = fusion
    _align(store, "T01")
    engine.report("T01", "lidar", False, 1.0)
    engine.report("T01", "ir", True, 1.0)
    engine.report("T01", "manual", True, 1.0)
    verdict = engine.pump("T01", 1.0)
    assert verdict.candidate == "split"
    assert verdict.auto_label == "hold"
    assert verdict.can_depart is False


def test_strong_consensus_blocks_immediately_after_confirm(fusion):
    engine, store = fusion
    _align(store, "T01")
    engine.report("T01", "lidar", False, 1.0)
    engine.report("T01", "ir", False, 1.0)
    verdict = engine.pump("T01", 1.0)
    # lidar+ir=0.80 >= 2/3，一边倒为 blocked，但仍需持续 4s 落定，窗口内先拦。
    assert verdict.candidate == "blocked"
    assert verdict.auto_label == "hold"
    assert engine.pump("T01", 5.0).auto_label == "blocked"


def test_conflict_escalates_to_review_after_hold_window(fusion):
    engine, store = fusion
    _align(store, "T01")
    _report_clear_all(engine, "T01", 1.0)
    engine.pump("T01", 5.0)
    engine.report("T01", "ir", False, 5.0)
    engine.report("T01", "manual", False, 5.0)
    assert engine.pump("T01", 5.0).auto_label == "hold"
    verdict = engine.pump("T01", 11.0)
    assert verdict.auto_label == "review_required"
    assert verdict.review_required is True
    assert verdict.review_request_seq is not None


# ---- 部分失败与过期 ----------------------------------------------------


def test_faulty_source_is_excluded_from_weights(fusion):
    engine, store = fusion
    _align(store, "T01")
    engine.mark_source("T01", "ir", True, 1.0)
    engine.report("T01", "lidar", True, 1.0)
    engine.report("T01", "manual", True, 1.0)
    verdict = engine.pump("T01", 5.0)
    # ir 被剔除后，参与权重 0.65 >= 0.6，清的份额 100%。
    assert round(verdict.coverage, 2) == 0.65
    assert verdict.can_depart is True
    ir_view = next(view for view in verdict.sources if view.id == "ir")
    assert ir_view.healthy is False and ir_view.participating is False


def test_two_sources_faulty_below_quorum_escalates(fusion):
    engine, store = fusion
    _align(store, "T01")
    engine.mark_source("T01", "ir", True, 1.0)
    engine.mark_source("T01", "manual", True, 1.0)
    engine.report("T01", "lidar", True, 1.0)
    assert engine.pump("T01", 8.0).auto_label == "review_required"


def test_stale_observations_drop_coverage(fusion):
    engine, store = fusion
    _align(store, "T01")
    _report_clear_all(engine, "T01", 1.0)
    verdict = engine.pump("T01", 1.0 + FusionPolicy().freshness_s + 1.0)
    assert verdict.candidate == "insufficient"
    assert verdict.can_depart is False


def test_unknown_source_rejected(fusion):
    engine, store = fusion
    _align(store, "T01")
    with pytest.raises(InterlockError):
        engine.report("T01", "sonar-x", True, 1.0)


# ---- 人工复核 ----------------------------------------------------------


def test_unaligned_train_rejects_manual_review(fusion):
    engine, _store = fusion
    with pytest.raises(InterlockError):
        engine.manual_review("T99", True, 1.0, role=SUPERVISOR)


def test_operator_cannot_override_conflict_supervisor_can(fusion):
    engine, store = fusion
    _align(store, "T01")
    engine.report("T01", "lidar", False, 1.0)
    engine.report("T01", "ir", True, 1.0)
    engine.report("T01", "manual", True, 1.0)
    engine.pump("T01", 8.0)
    with pytest.raises(InterlockError):
        engine.manual_review("T01", True, 8.5, role=OPERATOR)
    engine.manual_review("T01", True, 8.5, role=SUPERVISOR, reviewer="s1")
    verdict = engine.pump("T01", 8.5)
    assert verdict.can_depart is True
    assert verdict.source == "manual_clear"
    assert verdict.override["remaining_s"] == pytest.approx(60.0)


def test_operator_confirm_allowed_when_auto_clear(fusion):
    engine, store = fusion
    _align(store, "T01")
    _report_clear_all(engine, "T01", 1.0)
    engine.pump("T01", 5.0)
    event = engine.manual_review("T01", True, 5.5, role=OPERATOR, reviewer="op")
    assert event.type == "review_recorded"


def test_manual_clear_is_single_use(fusion):
    engine, store = fusion
    _align(store, "T01")
    engine.report("T01", "lidar", False, 1.0)
    engine.pump("T01", 8.0)
    engine.manual_review("T01", True, 8.5, role=SUPERVISOR)
    first, decision = engine.depart_decision("T01", 8.6)
    assert first.can_depart is True
    assert decision.payload["consumed_override"] is not None
    second = engine.pump("T01", 8.7)
    assert second.can_depart is False
    assert second.source == "auto"


def test_manual_clear_expires_after_ttl(fusion):
    engine, store = fusion
    _align(store, "T01")
    engine.report("T01", "lidar", False, 1.0)
    engine.pump("T01", 8.0)
    engine.manual_review("T01", True, 8.5, role=SUPERVISOR)
    assert engine.pump("T01", 68.5).can_depart is True
    assert engine.pump("T01", 68.5 + 0.01).can_depart is False


def test_manual_blocked_is_sticky_until_new_review(fusion):
    engine, store = fusion
    _align(store, "T01")
    _report_clear_all(engine, "T01", 1.0)
    engine.manual_review("T01", False, 1.1, role=SUPERVISOR, note="看到异物")
    verdict = engine.pump("T01", 10.0)
    assert verdict.source == "manual_blocked" and verdict.label == "blocked"
    # 后续三路自动一致 clear，人工拦停仍在。
    verdict = engine.pump("T01", 20.0)
    assert verdict.source == "manual_blocked"
    # 下一次人工结论改变。
    engine.manual_review("T01", True, 20.5, role=SUPERVISOR)
    assert engine.pump("T01", 20.5).source == "manual_clear"


# ---- 不可变、恢复、重放、审计 ------------------------------------------


def test_replay_rebuilds_identical_verdict_and_hash(fusion):
    engine, store = fusion
    _align(store, "T01")
    _report_clear_all(engine, "T01", 1.0)
    engine.pump("T01", 5.0)
    engine.manual_review("T01", False, 6.0, role=SUPERVISOR)
    before = engine.state_hash()
    rebuilt = GapFusion(store, sources=SOURCES, policy=FusionPolicy())
    assert rebuilt.state_hash() == before
    assert rebuilt.verdict("T01", 6.5).basis_hash == engine.verdict("T01", 6.5).basis_hash


def test_audit_replay_verifies_departure_basis(fusion):
    engine, store = fusion
    _align(store, "T01")
    _report_clear_all(engine, "T01", 1.0)
    engine.pump("T01", 5.0)
    engine.depart_decision("T01", 5.1)
    report = verify_train("T01", store.root, SOURCES, FusionPolicy())
    assert report["consistent"] is True
    assert len(report["decisions"]) == 1
    assert report["decisions"][0]["basis_match"] is True
    assert verify_all(store.root, SOURCES, FusionPolicy())[0]["consistent"] is True


def test_torn_tail_is_truncated_and_reappended(fusion, tmp_path):
    engine, store = fusion
    _align(store, "T01")
    _report_clear_all(engine, "T01", 1.0)
    engine.pump("T01", 1.0)
    path = glob.glob(os.path.join(store.root, "gap-T01.jsonl"))[0]
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"seq": 4, "at": 2.0, "typ')  # 模拟掉电半截写入

    rebuilt = GapFusion(store, sources=SOURCES, policy=FusionPolicy())
    state = rebuilt._state("T01")
    assert state.truncated == 1
    assert rebuilt.replay_events("T01")[-1].seq == 3
    event = rebuilt.report("T01", "lidar", True, 3.0)
    assert event.seq == 4  # 截断后从正确序号续写


def test_tampered_middle_breaks_chain_and_refuses_writes(fusion):
    engine, store = fusion
    _align(store, "T01")
    _report_clear_all(engine, "T01", 1.0)
    path = glob.glob(os.path.join(store.root, "gap-T01.jsonl"))[0]
    lines = open(path, encoding="utf-8").read().splitlines()
    row = json.loads(lines[1])
    row["payload"]["observation"] = "blocked"
    lines[1] = json.dumps(row, ensure_ascii=False)
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    tampered = GapFusion(store, sources=SOURCES, policy=FusionPolicy())
    assert tampered.integrity()["T01"]["broken_chain"] is True
    with pytest.raises(InterlockError):
        tampered.report("T01", "ir", True, 2.0)


def test_late_observation_is_marked_not_silently_accepted(fusion):
    engine, store = fusion
    _align(store, "T01")
    seen: list[tuple] = []
    engine._audit_hook = lambda kind, train, detail: seen.append((kind, detail))
    engine.report("T01", "lidar", True, 100.0)
    engine.report("T01", "lidar", False, 100.0 - FusionPolicy().late_grace_s - 1)
    assert any(kind == "gap_late_observation" for kind, _ in seen)


def test_status_helpers(fusion):
    engine, store = fusion
    _align(store, "T01")
    _report_clear_all(engine, "T01", 1.0)
    engine.pump("T01", 5.0)
    status = gap_status(engine, "T01", 5.0)
    assert status["state"] == "clear" and status["clear"] is True
    summary = gap_summary(engine, ["T01"], 5.0)
    assert summary["count"] == 1 and summary["clear"] == 1
    assert [source["id"] for source in summary["sources"]] == ["lidar", "ir", "manual"]
    assert len(audit_timeline(engine, "T01")) >= 4


# ---- 配额（原有能力不回归）--------------------------------------------


def test_quota_consume_and_reset(tmp_path):
    quotas = QuotaRegistry(JsonStore(str(tmp_path)))
    quotas.set_limit("arrival", 2)
    assert quotas.consume("arrival") is True
    assert quotas.consume("arrival") is True
    assert quotas.consume("arrival") is False
    assert quotas.remaining("arrival") == 0
    quotas.reset("arrival")
    assert quotas.remaining("arrival") == 2
    assert quotas.status("arrival")["configured"] is True
    assert quotas.status("depart")["configured"] is False
    with pytest.raises(RailError):
        quotas.consume("depart")
    with pytest.raises(RailError):
        quotas.set_limit("arrival", 0)
    assert quotas.reset("depart") is None
