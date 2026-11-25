"""显著门槛口径与可确认检验结果批次测试。

覆盖：
* ``0.95`` 与 ``95%`` 得到同一个 ``alpha = 0.05``；P 值落在 0.01~0.05 时判定为显著，
  且不会因为 ``float("0.95") / 100`` 这类错位而被判成 0.0095 的门槛。
* 实验评估只关联「已确认」的结果批次，不按任务当前置信水平重算显著标记。
* 同一结果重复登记（含登记中断后重试）不生成第二份结论。
* 没有可核数据时返回原因。
"""
from __future__ import annotations

import pytest

from app.routers.chains import chain_experiment_evaluate
from app.services.significance import SignificanceService
from app.services.significance_results import (
    confidence_to_alpha,
    parse_confidence,
    result_registry,
)
from app.store import store

SAMPLE = {
    "sample_batch_id": "SMP-TEST-001",
    "control": [100, 101, 99, 102, 98, 100, 101, 99, 100, 102] * 10,
    "treatment": [102, 103, 101, 104, 100, 102, 103, 101, 102, 104] * 10,
}


@pytest.fixture(autouse=True)
def clean_registry():
    store.reset()
    result_registry.clear()
    yield
    result_registry.clear()
    store.reset()


class TestConfidenceNormalization:
    @pytest.mark.parametrize(
        "value", [0.95, "0.95", "95%", "95", " 95 % ", "95.0%"]
    )
    def test_same_alpha_for_decimal_and_percent(self, value):
        assert parse_confidence(value) == pytest.approx(0.95)
        assert confidence_to_alpha(value) == pytest.approx(0.05)

    def test_ninety_nine(self):
        assert confidence_to_alpha("99%") == pytest.approx(0.01)
        assert confidence_to_alpha(0.99) == pytest.approx(0.01)

    def test_invalid(self):
        for bad in ["", "abc", 0, -0.1, None]:
            with pytest.raises(ValueError):
                parse_confidence(bad)

    def test_percent_style_out_of_range(self):
        with pytest.raises(ValueError):
            parse_confidence(150)

    def test_run_test_uses_real_threshold(self):
        """旧实现把 "0.95" 除以 100 得到 0.0095 门槛，0.03 会被误判为不显著。"""
        svc = SignificanceService()
        store.find("significance", 2)["置信水平"] = "0.95"
        result = svc.run_test(2, confidence="95%")
        assert result["alpha"] == pytest.approx(0.05)
        # 0.95 与 95% 在同一份样本上给出一致结论
        again = svc.run_test(2, confidence=0.95)
        assert again["batch_id"] == result["batch_id"]
        assert again["alpha"] == result["alpha"]


class TestNoDoubleConclusion:
    def test_retry_after_interrupted_registration(self):
        svc = SignificanceService()
        first = svc.register_result(2, {"p_value": 0.03, "sample": SAMPLE})
        assert first["ok"] and first["created"]
        retry = svc.register_result(2, {"p_value": 0.03, "sample": SAMPLE})
        assert retry["ok"]
        assert retry["batch_id"] == first["batch_id"]
        assert retry["deduplicated"] is True
        assert len(result_registry.for_task(2)) == 1

    def test_confidence_change_does_not_spawn_second_batch(self):
        """置信水平从 95% 改成 99% 后再重试同一结果，仍只有一份结论。"""
        svc = SignificanceService()
        first = svc.register_result(2, {"p_value": 0.03, "sample": SAMPLE, "confidence": "95%"})
        store.find("significance", 2)["置信水平"] = "99%"
        retry = svc.register_result(2, {"p_value": 0.03, "sample": SAMPLE})
        assert retry["batch_id"] == first["batch_id"]
        # 结论保留登记当时的口径
        assert retry["alpha"] == pytest.approx(0.05)
        assert retry["significant"] is True
        assert len(result_registry.for_task(2)) == 1

    def test_pending_then_confirm_keeps_original_threshold(self):
        """结果还在路上时按 95% 记录；管理员改成 99% 后确认，仍按 95% 判定。"""
        svc = SignificanceService()
        pending = svc.preview_pending(2, p_value=0.03, sample=SAMPLE, confidence="95%")
        assert pending["ok"] and pending["alpha"] == pytest.approx(0.05)
        store.find("significance", 2)["置信水平"] = "99%"
        confirmed = svc.register_result(2, {"p_value": 0.03, "sample": SAMPLE, "confirm": True})
        assert confirmed["alpha"] == pytest.approx(0.05)
        assert confirmed["significant"] is True
        # 0.03 > 0.01，如果按当前 99% 设置重算就会被判成不显著
        assert confirmed["deduplicated"] is False
        assert len(result_registry.for_task(2)) == 1


class TestPendingAfterCompute:
    """同一轮实验刚算完（95%）、结果还在路上，管理员改成 99% 后再评估/重试/复盘。"""

    def _scenario(self):
        svc = SignificanceService()
        first = svc.compute_pending(2)
        assert first["ok"] and first["status"] == "待回执"
        assert first["alpha"] == pytest.approx(0.05)
        # 管理员把置信水平更正为 99%
        store.find("significance", 2)["置信水平"] = "99%"
        return svc, first

    def test_no_second_conclusion_on_retry(self):
        svc, first = self._scenario()
        retry = svc.run_test(2)
        assert retry["deduplicated"] is True
        assert retry["status"] == "待回执"
        assert retry["alpha"] == pytest.approx(0.05)
        assert retry["confidence"] == pytest.approx(0.95)
        # 还没有正式批次，也不会有第二份结论
        assert len(result_registry.for_task(2)) == 0

    def test_evaluation_without_receipt_reports_reason(self):
        svc, _ = self._scenario()
        result = chain_experiment_evaluate(2)
        assert result["evaluation"]["confirmed_results"] == 0
        assert result["significance_tests"] == []
        assert result["evaluation"]["unresolved"][0]["status"] == "待回执"

    def test_retry_confirm_then_evaluate_and_review(self):
        svc, first = self._scenario()
        pending_record = result_registry.pending_for_task(2)[0]
        registered = svc.register_result(
            2,
            {
                "p_value": pending_record["p_value"],
                "sample": pending_record["sample"],
            },
        )
        assert registered["created"] is True
        # 重试登记沿用待回执时捕获的 95% 门槛，而不是当前任务的 99%
        assert registered["alpha"] == pytest.approx(0.05)
        confirmed = svc.confirm_result(registered["batch_id"], task_id=2)
        assert confirmed["status"] == "已确认"

        evaluation = chain_experiment_evaluate(2)["significance_tests"][0]
        assert evaluation["alpha"] == pytest.approx(0.05)
        assert evaluation["significant"] is True
        assert evaluation["sample_batch_id"] == pending_record["sample_batch_id"]

        review = svc.review_batch(batch_id=registered["batch_id"])
        assert review["alpha"] == pytest.approx(0.05)
        assert review["confidence"] == pytest.approx(0.95)
        assert review["significant"] is True
        assert review["sample_batch_id"] == pending_record["sample_batch_id"]

    def test_retry_without_payload_recovers_pending_result(self):
        """登记中断后只补一个空回执，也能从待回执记录恢复同一份结果。"""
        svc, first = self._scenario()
        recovered = svc.register_result(2, {})
        assert recovered["ok"] is True
        assert recovered["alpha"] == pytest.approx(0.05)
        assert recovered["p_value"] == pytest.approx(first["p_value"])
        assert recovered["sample_batch_id"] == first["sample_batch_id"]
        assert len(result_registry.for_task(2)) == 1
        again = svc.register_result(2, {})
        assert again["batch_id"] == recovered["batch_id"]
        assert len(result_registry.for_task(2)) == 1


class TestPendingBetweenPointZeroOneAndPointZeroFive:
    """P 落在 0.01~0.05：回执缺失时评估 → 重试确认 → 复盘，全程用同一个批次。"""

    def test_full_cycle(self):
        svc = SignificanceService()
        pending = svc.preview_pending(2, p_value=0.03, sample=SAMPLE, confidence="95%")
        assert pending["status"] == "待回执"
        # 管理员在结果落地前把置信水平改成 99%
        store.find("significance", 2)["置信水平"] = "99%"

        # 1) 回执缺失时评估：没有可确认结论，但要说明原因
        before = chain_experiment_evaluate(2)
        assert before["evaluation"]["confirmed_results"] == 0
        assert before["significance_tests"] == []
        assert before["evaluation"]["unresolved"][0]["status"] == "待回执"

        # 2) 重试登记同一检验结果
        retry = svc.register_result(2, {"p_value": 0.03, "sample": SAMPLE})
        assert retry["created"] is True
        assert retry["alpha"] == pytest.approx(0.05)
        assert retry["significant"] is True
        again = svc.register_result(2, {"p_value": 0.03, "sample": SAMPLE})
        assert again["batch_id"] == retry["batch_id"]
        assert again["deduplicated"] is True

        # 3) 补确认，再评估
        confirmed = svc.confirm_result(None, task_id=2)
        assert confirmed["batch_id"] == retry["batch_id"]
        assert confirmed["status"] == "已确认"
        after = chain_experiment_evaluate(2)
        item = after["significance_tests"][0]
        assert item["batch_id"] == retry["batch_id"]
        assert item["alpha"] == pytest.approx(0.05)
        assert item["confidence"] == pytest.approx(0.95)
        assert item["p_value"] == pytest.approx(0.03)
        assert item["significant"] is True
        assert item["sample_batch_id"] == "SMP-TEST-001"
        assert len(result_registry.for_task(2)) == 1

        # 4) 复盘读取同一份批次
        review = svc.review_batch(batch_id=retry["batch_id"])
        assert review["p_value"] == pytest.approx(0.03)
        assert review["alpha"] == pytest.approx(0.05)
        assert review["confidence"] == pytest.approx(0.95)
        assert review["significant"] is True
        assert review["sample_batch_id"] == "SMP-TEST-001"
        assert review["status"] == "已确认"


class TestEvaluationUsesConfirmedBatch:
    def test_pending_result_not_evaluated(self):
        svc = SignificanceService()
        svc.preview_pending(2, p_value=0.03, sample=SAMPLE, confidence="95%")
        result = chain_experiment_evaluate(2)
        assert result["evaluation"]["confirmed_results"] == 0
        assert result["significance_tests"] == []
        assert any(item.get("status") == "待回执" for item in result["evaluation"]["unresolved"])

    def test_confirmed_result_evaluated_with_its_own_alpha(self):
        svc = SignificanceService()
        registered = svc.register_result(2, {"p_value": 0.03, "sample": SAMPLE, "confidence": "95%"})
        svc.confirm_result(registered["batch_id"], task_id=2)
        store.find("significance", 2)["置信水平"] = "99%"
        result = chain_experiment_evaluate(2)
        assert result["evaluation"]["confirmed_results"] == 1
        item = result["significance_tests"][0]
        assert item["batch_id"] == registered["batch_id"]
        assert item["alpha"] == pytest.approx(0.05)
        assert item["p_value"] == pytest.approx(0.03)
        assert item["significant"] is True
        assert item["sample_batch_id"] == "SMP-TEST-001"

    def test_p_value_between_point_zero_one_and_point_zero_five(self):
        """0.01 < P < 0.05：95% 门槛下显著，99% 门槛下不显著。"""
        svc = SignificanceService()
        batch = svc.register_result(2, {"p_value": 0.02, "sample": SAMPLE, "confidence": "95%"})
        assert batch["significant"] is True
        svc.confirm_result(batch["batch_id"], task_id=2)
        result = chain_experiment_evaluate(2)
        item = result["significance_tests"][0]
        assert item["significant"] is True
        assert item["alpha"] == pytest.approx(0.05)

    def test_reported_p_value_not_significant(self):
        """报告里的 P=0.490625 在 95% 口径下必须判为不显著。"""
        svc = SignificanceService()
        batch = svc.register_result(
            2, {"p_value": 0.490625, "sample": SAMPLE, "confidence": "95%"}
        )
        assert batch["alpha"] == pytest.approx(0.05)
        assert batch["significant"] is False
        svc.confirm_result(batch["batch_id"], task_id=2)
        item = chain_experiment_evaluate(2)["significance_tests"][0]
        assert item["p_value"] == pytest.approx(0.490625)
        assert item["significant"] is False
        assert item["result_label"] == "不显著"

    def test_review_reads_same_sample_and_threshold(self):
        svc = SignificanceService()
        batch = svc.register_result(2, {"p_value": 0.03, "sample": SAMPLE})
        svc.confirm_result(batch["batch_id"], task_id=2)
        review = svc.review_batch(batch_id=batch["batch_id"], include_values=True)
        assert review["ok"]
        assert review["p_value"] == pytest.approx(0.03)
        assert review["alpha"] == pytest.approx(0.05)
        assert review["confidence"] == pytest.approx(0.95)
        assert review["significant"] is True
        assert review["sample_batch_id"] == "SMP-TEST-001"
        assert len(review["sample"]["control_values"]) == len(SAMPLE["control"])
        assert review["sample"]["control_values"] == [float(v) for v in SAMPLE["control"]]

    def test_review_without_batch_reports_reason(self):
        svc = SignificanceService()
        review = svc.review_batch(task_id=2)
        assert review["ok"] is False
        assert "没有可复盘" in review["reason"]


class TestNoDataToVerify:
    def test_register_without_sample_reports_reason(self):
        svc = SignificanceService()
        result = svc.register_result(2, {"p_value": 0.03})
        assert result["ok"] is False
        assert "没有数据可核" in result["reason"]
        assert result_registry.for_task(2) == []

    def test_register_without_p_value_reports_reason(self):
        svc = SignificanceService()
        result = svc.register_result(2, {"sample": SAMPLE})
        assert result["ok"] is False
        assert "P 值" in result["reason"]

    def test_run_test_with_unparseable_sample(self):
        svc = SignificanceService()
        result = svc.run_test(2, sample={"control": [], "treatment": []})
        assert result["ok"] is False
        assert "没有数据可核" in result["reason"]
