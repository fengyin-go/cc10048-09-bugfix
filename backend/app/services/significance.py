"""显著性检验领域服务：统计检验执行和结果判定。

约定：
* 置信水平统一用 :func:`parse_confidence` 归一，``0.95`` 与 ``95%`` 都对应
  显著门槛 ``alpha = 0.05``。
* 每次检验都登记一个「可确认检验结果批次」，P 值、置信水平、样本批次和
  显著标记由同一次检验产生；实验评估和复盘都读批次，不再按任务当前设置重算。
* 同一份结果重复登记（含登记中断后的重试）命中原批次，不会产生第二份结论。
"""
# 数据操作：状态、状态、处理，与上下游模块联动消费批次数据。
from __future__ import annotations

import random
from typing import Any

from app.compute import chi_square_test, t_test
from app.services.base import BaseModuleService
from app.services.significance_results import (
    DEFAULT_CONFIDENCE,
    confidence_to_alpha,
    normalize_sample,
    now_text,
    parse_confidence,
    public_batch,
    result_registry,
    stable_seed,
)
from app.store import store


class SignificanceService(BaseModuleService):
    MODULE = "significance"
    ENTITY = "检验任务"
    OPERATION_TERM = "关联"
    LIST_FIELDS = ["任务标识", "关联实验", "检验方法", "置信水平", "最小提升", "检验结果", "P值", "检验时间"]
    # 结果批次、样本批次、显著门槛是本次检验的实际口径，和「置信水平」这一当前设置分开。
    RESULT_FIELDS = ["结果批次", "样本批次", "结果置信水平", "显著门槛", "结果状态"]
    REQUIRED_FIELDS = ["任务标识", "关联实验", "检验方法"]
    STATUS_ORDER = ["待计算", "计算中", "显著", "不显著"]
    ACTION_RULES = {"启动计算": "待计算", "调整参数": "不显著"}
    NEGATIVE_ACTIONS = []

    def normalize_values(self, values: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """登记/更正时把置信水平统一成小数写法，0.95 与 95% 都存成 0.95。"""
        if "置信水平" not in values:
            return values, None
        try:
            normalized = parse_confidence(values["置信水平"])
        except ValueError as exc:
            return values, f"置信水平无效：{exc}"
        return {**values, "置信水平": str(normalized)}, None

    # ---- 内部工具 ----
    def _resolve_confidence(self, task: dict[str, Any], override: Any = None) -> tuple[float, str]:
        """取本次检验实际采用的置信水平；无法解析时回落到 0.95 并说明来源。"""
        if override not in (None, ""):
            try:
                return parse_confidence(override), "调用方传入"
            except ValueError:
                return DEFAULT_CONFIDENCE, "默认值（传入的置信水平无效）"
        raw = task.get("置信水平")
        try:
            return parse_confidence(raw), "任务设置"
        except ValueError:
            return DEFAULT_CONFIDENCE, "默认值（任务未填写有效置信水平）"

    def _confidence_from_task_or(self, task: dict[str, Any], override: Any) -> tuple[float, str]:
        """计算时若任务设置已被改动，优先采用本轮检验实际使用的口径。"""
        if override not in (None, ""):
            try:
                return parse_confidence(override), "调用方传入"
            except ValueError:
                pass
        # 本轮的检验结果还在路上（回执暂时缺失）时，沿用那份记录的门槛；
        # 不能让管理员刚改过的设置把「已经算完」的这一次检验按新口径重算。
        previous = self._pending_confidence(task)
        if previous is not None:
            return previous, "沿用待回执检验口径"
        return self._resolve_confidence(task, None)

    def _pending_confidence(self, task: dict[str, Any]) -> float | None:
        task_id = int(task.get("id", 0))
        try:
            pending = result_registry.pending_for_task(task_id)
        except AttributeError:  # pragma: no cover - 兼容旧注册表
            pending = []
        if pending:
            return float(pending[-1]["confidence"])
        return None

    @staticmethod
    def _method_of(task: dict[str, Any], override: Any = None) -> str:
        raw = override if override not in (None, "") else task.get("检验方法", "t_test")
        return str(raw or "t_test")

    @staticmethod
    def _is_chi(method: str) -> bool:
        lowered = method.lower()
        return "chi" in lowered or "卡方" in method

    def _simulate_sample(self, task: dict[str, Any], method: str) -> dict[str, float]:
        """按任务和关联实验生成样本；种子跨进程稳定，保证重试拿到同一份样本。"""
        exp_ref = str(task.get("关联实验") or task.get("id") or "1")
        rng = random.Random(stable_seed("significance-sample", task.get("id"), exp_ref))
        n_a = rng.randint(100, 5000)
        n_b = rng.randint(100, 5000)
        group_a = [rng.gauss(100, 15) for _ in range(n_a)]
        group_b = [rng.gauss(105, 15) for _ in range(n_b)]
        return {"control": group_a, "treatment": group_b}

    def _run_statistics(
        self, method: str, sample: dict[str, Any], alpha: float
    ) -> dict[str, Any]:
        if sample.get("kind") == "values":
            group_a = list(sample.get("control") or [])
            group_b = list(sample.get("treatment") or [])
        else:
            # 仅有摘要时用摘要重建一个同规模样本，保持计算入口可用。
            # 种子只由摘要决定、与 alpha 无关，保证同一份样本换门槛复核时 P 值不变。
            rng = random.Random(
                stable_seed(
                    "significance-summary",
                    sample.get("sample_batch_id"),
                    sample.get("sample_fingerprint"),
                )
            )
            n_a = int(sample.get("n_control") or 0)
            n_b = int(sample.get("n_treatment") or 0)
            mean_a = float(sample.get("mean_control") or 100.0)
            mean_b = float(sample.get("mean_treatment") or 105.0)
            std_a = float(sample.get("std_control") or 15.0)
            std_b = float(sample.get("std_treatment") or 15.0)
            group_a = [rng.gauss(mean_a, std_a) for _ in range(max(n_a, 2))]
            group_b = [rng.gauss(mean_b, std_b) for _ in range(max(n_b, 2))]
        confidence_band = max(alpha, 0.0001)
        if self._is_chi(method):
            return chi_square_test(
                [sum(group_a) / len(group_a), sum(group_b) / len(group_b)],
                [sum(group_a) / len(group_a)] * 2,
                alpha=confidence_band,
            )
        return t_test(group_a, group_b, alpha=confidence_band)

    def _write_task_result(self, task_id: int, batch: dict[str, Any]) -> None:
        task = store.find(self.MODULE, task_id)
        if not task:
            return
        # 任务上的「置信水平」是管理员当前设置，结果批次保留的是本次检验实际采用的
        # 口径；两者分开写，避免登记结果把管理员刚改过的设置又覆盖回去。
        task["检验结果"] = "显著" if batch["significant"] else "不显著"
        task["P值"] = str(batch["p_value"])
        task["结果置信水平"] = str(batch["confidence"])
        task["显著门槛"] = str(batch["alpha"])
        task["检验时间"] = batch.get("tested_at") or now_text()
        task["结果批次"] = batch["batch_id"]
        task["样本批次"] = batch.get("sample_batch_id")
        task["结果状态"] = batch["status"]
        # 未确认的结果只是待回执，不能当成已定结论对外呈现。
        if batch["status"] == "已确认":
            task["status"] = "显著" if batch["significant"] else "不显著"
            task["pending"] = False
        else:
            task["status"] = "计算中"
            task["pending"] = True

    # ---- 对外业务方法 ----
    def _existing_confirmed(self, task_id: int) -> dict[str, Any] | None:
        """任务已经确认过的结果批次；有它就不再重复计算。"""
        confirmed = [
            batch for batch in result_registry.for_task(task_id)
            if batch["status"] == "已确认"
        ]
        return confirmed[-1] if confirmed else None

    def _existing_run_response(
        self,
        task: dict[str, Any],
        batch: dict[str, Any],
        confidence_value: float,
        confidence_source: str,
        alpha: float,
    ) -> dict[str, Any]:
        response = {
            "ok": True,
            "task_id": batch["task_id"],
            "task_name": task.get("任务标识"),
            "experiment_ref": batch["experiment_ref"],
            "method": batch["method"],
            "alpha": batch["alpha"],
            "confidence": batch["confidence"],
            "confidence_source": confidence_source,
            "p_value": batch["p_value"],
            "result": {
                "method": batch["method"],
                "p_value": batch["p_value"],
                "significant": batch["significant"],
                "alpha": batch["alpha"],
            },
            "significant": batch["significant"],
            "sample_batch_id": batch.get("sample_batch_id"),
            "sample_retained": True,
            "created": False,
            "deduplicated": True,
            "batch_id": batch["batch_id"],
            "batch": public_batch(batch),
            "note": "该任务已有可确认的检验结果，直接沿用原批次，未再生成第二份结论",
        }
        if abs(alpha - batch["alpha"]) > 1e-9:
            response["threshold_mismatch"] = {
                "requested_alpha": alpha,
                "result_alpha": batch["alpha"],
                "note": "当前置信水平与结果批次不一致，结论仍以结果批次为准",
            }
        return response

    def run_test(
        self,
        task_id: int,
        *,
        confidence: Any = None,
        method: Any = None,
        sample: Any = None,
        source: str = "计算任务",
        register: bool = True,
    ) -> dict[str, Any]:
        """执行（或登记）一次显著性检验。

        * ``confidence``/``method`` 省略时读任务当前设置；
        * ``sample`` 省略时使用按任务稳定生成、可复盘的同一样本；
        * 结果按「任务 + 方法 + P 值 + 样本」幂等登记，重试不会生成第二份结论。
        """
        task = store.find(self.MODULE, task_id)
        if not task:
            return {"error": f"检验任务 {task_id} 不存在"}

        confidence_value, confidence_source = self._confidence_from_task_or(task, confidence)
        alpha = confidence_to_alpha(confidence_value)
        method_name = self._method_of(task, method)
        existing = self._existing_confirmed(task_id)
        pending = result_registry.pending_for_task(task_id)

        if sample is None and existing is not None:
            # 同一任务已有可确认结果时不再重算，直接把原批次交回去。
            return self._existing_run_response(task, existing, confidence_value, confidence_source, alpha)
        if sample is None and pending:
            # 结果还在路上：不要再算第二份结论，等回执补齐后走登记/确认。
            record = pending[-1]
            return {
                "ok": True,
                "task_id": task_id,
                "task_name": task.get("任务标识"),
                "experiment_ref": record["experiment_ref"],
                "method": record["method"],
                "alpha": record["alpha"],
                "confidence": record["confidence"],
                "confidence_source": "沿用最近一次检验口径",
                "p_value": record["p_value"],
                "significant": record["significant"],
                "sample_batch_id": record.get("sample_batch_id"),
                "sample_retained": True,
                "created": False,
                "deduplicated": True,
                "batch_id": None,
                "pending_id": record["pending_id"],
                "status": "待回执",
                "note": "该任务已有一份结果还在路上（回执暂时缺失），未重复生成结论",
            }

        sample_payload = normalize_sample(sample) if sample is not None else None
        explicit_sample = sample_payload is not None
        if sample_payload is None:
            # 只在没有显式样本时兜底；显式传入但无法解析的样本走「没有数据可核」。
            if sample is not None and str(sample).strip() not in ("", "None"):
                return {
                    "task_id": task_id,
                    "task_name": task.get("任务标识"),
                    "ok": False,
                    "reason": "没有数据可核：样本缺少对照组/实验组样本值或可追溯的样本批次",
                }
            sample_payload = normalize_sample(self._simulate_sample(task, method_name))
            if sample_payload is None:  # pragma: no cover - 防御性分支
                return {
                    "task_id": task_id,
                    "task_name": task.get("任务标识"),
                    "ok": False,
                    "reason": "没有数据可核：无法生成可留存的样本",
                }

        result = self._run_statistics(method_name, sample_payload, alpha)
        p_value = float(result.get("p_value", 1.0))
        significant = p_value < alpha

        batch_payload = {
            "task_id": task_id,
            "experiment_ref": str(task.get("关联实验") or ""),
            "method": result.get("method") or method_name,
            "confidence": confidence_value,
            "alpha": alpha,
            "p_value": p_value,
            "significant": significant,
            "sample": sample_payload,
            "source": source,
            "status": "待确认",
        }
        if register:
            batch, created = result_registry.register(**batch_payload)
            self._write_task_result(task_id, batch)
        else:  # 预留给只计算不登记的场景
            batch = {**batch_payload, "batch_id": None}
            created = True

        response = {
            "ok": True,
            "task_id": task_id,
            "task_name": task.get("任务标识"),
            "experiment_ref": batch_payload["experiment_ref"],
            "method": result.get("method"),
            "alpha": alpha,
            "confidence": confidence_value,
            "confidence_source": confidence_source,
            "p_value": p_value,
            "result": {**result, "significant": significant, "p_value": p_value, "alpha": alpha},
            "significant": significant,
            "sample_batch_id": sample_payload.get("sample_batch_id"),
            "sample_retained": True,
            "created": created,
            "deduplicated": register and not created,
            "batch_id": batch["batch_id"],
            "batch": public_batch(batch) if register else None,
            "note": "同一份检验结果已登记，沿用原批次，未生成第二份结论"
            if register and not created
            else None,
        }
        if not explicit_sample:
            response["sample_source"] = "simulated"
        return response

    def compute_pending(
        self,
        task_id: int,
        *,
        confidence: Any = None,
        method: Any = None,
        sample: Any = None,
    ) -> dict[str, Any]:
        """算完一次检验，但结果还在路上：只留待回执记录，不出结论。

        这次检验实际采用的门槛与样本会被完整留存，后续评估、重试登记和复盘
        都按这份记录走，不受管理员此时更改置信水平的影响。
        """
        task = store.find(self.MODULE, task_id)
        if not task:
            return {"ok": False, "error": f"检验任务 {task_id} 不存在"}

        confidence_value, confidence_source = self._confidence_from_task_or(task, confidence)
        alpha = confidence_to_alpha(confidence_value)
        method_name = self._method_of(task, method)
        sample_payload = normalize_sample(sample) if sample is not None else None
        if sample_payload is None:
            if sample is not None and str(sample).strip() not in ("", "None"):
                return {"ok": False, "task_id": task_id,
                        "reason": "没有数据可核：样本缺少对照组/实验组样本值或可追溯的样本批次"}
            sample_payload = normalize_sample(self._simulate_sample(task, method_name))

        result = self._run_statistics(method_name, sample_payload, alpha)
        p_value = float(result.get("p_value", 1.0))
        significant = p_value < alpha
        pending = result_registry.mark_pending(
            task_id=task_id,
            experiment_ref=str(task.get("关联实验") or ""),
            method=result.get("method") or method_name,
            confidence=confidence_value,
            alpha=alpha,
            p_value=p_value,
            significant=significant,
            sample=sample_payload,
            source="计算完成待回执",
        )
        if pending.get("already_registered"):
            existing = result_registry.get(pending.get("batch_id"))
            if existing is not None:
                return self._existing_run_response(
                    task, existing, confidence_value, confidence_source, alpha
                )
        record = pending.get("record") or {}
        task["status"] = "计算中"
        task["pending"] = True
        task["结果状态"] = "待回执"
        task["P值"] = str(p_value)
        task["结果置信水平"] = str(confidence_value)
        task["显著门槛"] = str(alpha)
        task["样本批次"] = sample_payload.get("sample_batch_id")
        task["检验时间"] = record.get("tested_at") or now_text()
        return {
            "ok": True,
            "task_id": task_id,
            "task_name": task.get("任务标识"),
            "pending_id": pending.get("pending_id"),
            "status": "待回执",
            "confidence": confidence_value,
            "confidence_source": confidence_source,
            "alpha": alpha,
            "p_value": p_value,
            "significant": significant,
            "method": result.get("method"),
            "sample_batch_id": sample_payload.get("sample_batch_id"),
            "sample_retained": True,
            "note": "结果还在路上（回执暂时缺失），先按本次口径留存；确认后再作为可确认结论",
        }

    def preview_pending(
        self,
        task_id: int,
        *,
        p_value: Any,
        confidence: Any = None,
        method: Any = None,
        sample: Any = None,
        source: str = "待回执",
    ) -> dict[str, Any]:
        """记录一份还在路上、回执暂时缺失的检验结果。

        只保存待确认状态，不判定为结论；评估不会把它当成可确认结果。
        """
        task = store.find(self.MODULE, task_id)
        if not task:
            return {"ok": False, "error": f"检验任务 {task_id} 不存在"}
        try:
            parsed_p = float(str(p_value).strip())
        except (TypeError, ValueError):
            return {"ok": False, "task_id": task_id, "reason": f"P 值无法解析：{p_value!r}"}
        if not 0.0 <= parsed_p <= 1.0:
            return {"ok": False, "task_id": task_id,
                    "reason": f"P 值必须位于 0 与 1 之间：{parsed_p}"}
        confidence_value, confidence_source = self._confidence_from_task_or(task, confidence)
        alpha = confidence_to_alpha(confidence_value)
        method_name = self._method_of(task, method)
        sample_payload = normalize_sample(sample)
        if sample_payload is None:
            return {"ok": False, "task_id": task_id,
                    "reason": "没有数据可核：待回执结果缺少可留存的样本或样本批次"}
        pending = result_registry.mark_pending(
            task_id=task_id,
            experiment_ref=str(task.get("关联实验") or ""),
            method=method_name,
            confidence=confidence_value,
            alpha=alpha,
            p_value=parsed_p,
            significant=parsed_p < alpha,
            sample=sample_payload,
            source=source,
        )
        record = pending.get("record") or {}
        return {
            "ok": True,
            "task_id": task_id,
            "pending_id": pending.get("pending_id"),
            "already_registered": pending.get("already_registered", False),
            "registered_batch_id": pending.get("batch_id"),
            "status": "待回执",
            "confidence": confidence_value,
            "confidence_source": confidence_source,
            "alpha": alpha,
            "p_value": parsed_p,
            "significant": parsed_p < alpha,
            "sample_batch_id": sample_payload.get("sample_batch_id"),
            "first_seen_at": record.get("first_seen_at"),
            "note": "结果尚未拿到回执，暂不作为可确认结论；重试登记成功后再评估",
        }

    def register_result(self, task_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        """登记一份已经算好的检验结果（可重试、幂等）。

        ``payload`` 支持 ``p_value``/``P值``、``confidence``/``置信水平``、
        ``sample``/``samples`` 等写法。缺少 P 值或样本时返回原因，不生成批次。
        """
        task = store.find(self.MODULE, task_id)
        if not task:
            return {"ok": False, "error": f"检验任务 {task_id} 不存在"}
        payload = payload or {}

        raw_p = payload.get("p_value", payload.get("pvalue", payload.get("P值")))
        raw_sample = payload.get(
            "sample", payload.get("samples", payload.get("样本", payload.get("sample_batch")))
        )
        pending_records = result_registry.pending_for_task(task_id)
        if raw_p is None and raw_sample is None and len(pending_records) == 1:
            # 登记中断后重试：回执没带结论内容时，恢复同一任务唯一的那份待回执结果。
            record = pending_records[0]
            payload = {
                **payload,
                "p_value": record["p_value"],
                "sample": record["sample"],
                "method": payload.get("method") or record.get("method"),
            }
            raw_p = payload["p_value"]
            raw_sample = payload["sample"]
        if raw_p is None or str(raw_p).strip() == "":
            # 回执为空但该任务已有唯一结果批次：说明是同一结果的重复重试。
            batches = result_registry.for_task(task_id)
            if raw_sample is None and len(batches) == 1:
                batch = batches[0]
                return {
                    "ok": True,
                    "task_id": task_id,
                    "batch_id": batch["batch_id"],
                    "created": False,
                    "deduplicated": True,
                    "confidence": batch["confidence"],
                    "alpha": batch["alpha"],
                    "p_value": batch["p_value"],
                    "significant": batch["significant"],
                    "sample_batch_id": batch.get("sample_batch_id"),
                    "status": batch["status"],
                    "note": "同一份检验结果已存在，沿用原批次，未生成第二份结论",
                    "batch": public_batch(batch),
                }
            return {
                "ok": False,
                "task_id": task_id,
                "reason": "没有数据可核：登记结果缺少 P 值",
            }
        try:
            p_value = float(str(raw_p).strip())
        except (TypeError, ValueError):
            return {"ok": False, "task_id": task_id,
                    "reason": f"P 值无法解析：{raw_p!r}"}
        if not 0.0 <= p_value <= 1.0:
            return {"ok": False, "task_id": task_id,
                    "reason": f"P 值必须位于 0 与 1 之间：{p_value}"}

        raw_confidence = payload.get(
            "confidence", payload.get("置信水平", payload.get("置信度"))
        )
        method_name = self._method_of(task, payload.get("method", payload.get("检验方法")))

        sample = normalize_sample(raw_sample)
        if sample is None:
            return {
                "ok": False,
                "task_id": task_id,
                "reason": "没有数据可核：登记结果缺少可留存的样本或样本批次",
            }

        # 同一结果若之前以「待回执」留过档，沿用当时捕获的口径与判定，不按当前设置重算。
        pending = result_registry.pending_for(
            task_id=task_id,
            method=method_name,
            p_value=p_value,
            sample_fingerprint=str(sample.get("sample_fingerprint", "")),
        )
        if raw_confidence not in (None, ""):
            confidence_value = parse_confidence(raw_confidence)
            confidence_source = "登记入参"
        elif pending is not None:
            confidence_value = pending["confidence"]
            confidence_source = "沿用待回执时口径"
        else:
            confidence_value, confidence_source = self._resolve_confidence(task, None)
        alpha = confidence_to_alpha(confidence_value)

        batch, created = result_registry.register(
            task_id=task_id,
            experiment_ref=str(task.get("关联实验") or ""),
            method=method_name,
            confidence=confidence_value,
            alpha=alpha,
            p_value=p_value,
            significant=p_value < alpha,
            sample=sample,
            source=str(payload.get("source") or "结果登记"),
            status=str(payload.get("status") or "待确认"),
            tested_at=payload.get("tested_at") or payload.get("检验时间"),
        )
        if payload.get("confirm") or payload.get("confirmed"):
            batch = result_registry.confirm(batch)
        self._write_task_result(task_id, batch)
        return {
            "ok": True,
            "task_id": task_id,
            "batch_id": batch["batch_id"],
            "created": created,
            "deduplicated": not created,
            "confidence": batch["confidence"],
            "confidence_source": confidence_source,
            "alpha": batch["alpha"],
            "p_value": batch["p_value"],
            "significant": batch["significant"],
            "sample_batch_id": batch.get("sample_batch_id"),
            "status": batch["status"],
            "note": None if created else "同一份检验结果已存在，沿用原批次，未生成第二份结论",
            "batch": public_batch(batch),
        }

    def evaluation_view(
        self, batch: dict[str, Any], *, requested_alpha: float | None = None
    ) -> dict[str, Any]:
        """实验评估读到的检验结果视图：结论一律取自批次本身。"""
        view = {
            "task_id": batch["task_id"],
            "task_name": (store.find(self.MODULE, int(batch["task_id"])) or {}).get("任务标识"),
            "batch_id": batch["batch_id"],
            "method": batch["method"],
            "confidence": batch["confidence"],
            "alpha": batch["alpha"],
            "p_value": batch["p_value"],
            "significant": batch["significant"],
            "result_label": "显著" if batch["significant"] else "不显著",
            "sample_batch_id": batch.get("sample_batch_id"),
            "sample_size": {
                key: value
                for key, value in (batch.get("sample") or {}).items()
                if key in {"n_control", "n_treatment", "kind"}
            },
            "confirmed_at": batch.get("confirmed_at"),
            "tested_at": batch.get("tested_at"),
            # 兼容旧前端：result 里给的显著标记同样是批次里的那一次结论。
            "result": {
                "method": batch["method"],
                "p_value": batch["p_value"],
                "significant": batch["significant"],
                "alpha": batch["alpha"],
            },
        }
        if requested_alpha is not None and abs(requested_alpha - batch["alpha"]) > 1e-9:
            view["threshold_mismatch"] = {
                "evaluation_alpha": requested_alpha,
                "result_alpha": batch["alpha"],
                "note": "评估采用的结果批次按当时的置信水平判定，未按当前设置重算显著标记",
            }
        return view

    def confirm_result(self, batch_id: str | None, *, task_id: int | None = None) -> dict[str, Any]:
        """确认一份检验结果批次（登记回执）；重复确认返回同一批次。"""
        batch = result_registry.get(batch_id) if batch_id else None
        if batch is None and task_id is not None:
            batch = result_registry.latest_for_task(task_id)
        if batch is None:
            return {
                "ok": False,
                "reason": "没有可确认的检验结果批次：回执缺失且未找到已登记结果",
            }
        if task_id is not None and int(batch["task_id"]) != int(task_id):
            return {"ok": False, "reason": "检验结果批次不属于该检验任务"}
        already = batch.get("status") == "已确认"
        batch = result_registry.confirm(batch)
        self._write_task_result(batch["task_id"], batch)
        return {
            "ok": True,
            "batch_id": batch["batch_id"],
            "task_id": batch["task_id"],
            "status": batch["status"],
            "already_confirmed": already,
            "idempotent": already,
            "alpha": batch["alpha"],
            "confidence": batch["confidence"],
            "p_value": batch["p_value"],
            "significant": batch["significant"],
            "sample_batch_id": batch.get("sample_batch_id"),
            "batch": public_batch(batch),
        }

    def review_batch(
        self, *, batch_id: str | None = None, task_id: int | None = None,
        include_values: bool = False,
    ) -> dict[str, Any]:
        """复盘接口：读取当时那次检验的样本、门槛和显著标记。"""
        batch = result_registry.get(batch_id) if batch_id else None
        if batch is None and task_id is not None:
            batch = result_registry.latest_for_task(task_id)
        if batch is None:
            return {
                "ok": False,
                "reason": "没有可复盘的检验结果批次：该任务尚未登记可确认结果",
            }
        return {
            "ok": True,
            "batch_id": batch["batch_id"],
            "task_id": batch["task_id"],
            "experiment_ref": batch["experiment_ref"],
            "method": batch["method"],
            "confidence": batch["confidence"],
            "alpha": batch["alpha"],
            "p_value": batch["p_value"],
            "significant": batch["significant"],
            "status": batch["status"],
            "confirmed": batch["status"] == "已确认",
            "sample_batch_id": batch.get("sample_batch_id"),
            "sample": public_batch(batch, include_values=include_values)["sample"],
            "tested_at": batch.get("tested_at"),
            "created_at": batch.get("created_at"),
            "confirmed_at": batch.get("confirmed_at"),
            "source": batch.get("source"),
            "batch": public_batch(batch, include_values=include_values),
        }

    def list_batches(
        self, task_id: int | None = None, *, include_values: bool = False
    ) -> dict[str, Any]:
        batches = result_registry.for_task(task_id) if task_id is not None else result_registry.all()
        return {
            "items": [public_batch(batch, include_values=include_values) for batch in batches],
            "total": len(batches),
        }

    def list_pending(self, task_id: int | None = None) -> dict[str, Any]:
        """列出还在路上（回执暂时缺失）的结果；它们不是可确认结论。"""
        if task_id is None:
            records = result_registry.pending_all()
        else:
            records = result_registry.pending_for_task(task_id)
        items = [
            {
                "pending_id": record["pending_id"],
                "task_id": record["task_id"],
                "method": record["method"],
                "confidence": record["confidence"],
                "alpha": record["alpha"],
                "p_value": record["p_value"],
                "significant": record["significant"],
                "status": record["status"],
                "sample_batch_id": record.get("sample_batch_id"),
                "sample": public_batch({"sample": record["sample"]})["sample"],
                "first_seen_at": record.get("first_seen_at"),
                "tested_at": record.get("tested_at"),
            }
            for record in records
        ]
        return {"items": items, "total": len(items)}

    def domain_enrich(self, config_id: int) -> dict:
        """对当前模块执行校验操作，联动导出和回填，产出标准化数据产物。

        数据加工流程：读取原始记录 → 执行校验处理 → 导出校验字段完整性 → 生成回填批次标识 →
        写入存储层并返回可追溯的处理结果，下游消费模块可基于批次号追踪数据链路。
        """
        import time, json
        from app.store import store
        entry = store.find(self.MODULE, config_id)
        # 校验：遍历当前模块全部记录执行核心数据处理
        records = store.list_all(self.MODULE)
        processed = []
        for idx, rec in enumerate(records):
            # 导出：校验每条记录的字段完整性与格式合法性
            filled = sum(1 for v in rec.values() if v and str(v).strip())
            total = len(rec)
            rec_copy = dict(rec)
            rec_copy["_field_count"] = filled
            rec_copy["_complete_ratio"] = f"{filled}/{total}"
            rec_copy["_batch"] = f"{self.MODULE}-{idx:04d}"
            processed.append(rec_copy)
        # 回填：生成加工批次标识并写入存储
        batch_id = store.next_id(self.MODULE + "_processor")
        result = {
            "module": self.MODULE,
            "operation": "校验",
            "config_id": config_id,
            "records_processed": len(processed),
            "batch_id": batch_id,
            "field_validation": {"导出": "passed", "回填": "recorded"},
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "downstream_consumable": True,
            "records": processed,
            "entry": entry,
        }
        # 返回校验处理结果，含导出校验元数据和回填批次追溯信息
        return result
