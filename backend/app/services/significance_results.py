"""可确认检验结果批次：置信水平归一、样本留存、幂等登记与复盘溯源。

设计要点：

* 置信水平统一走 :func:`parse_confidence`，``0.95`` / ``"0.95"`` / ``"95%"`` / ``95``
  都归一成 ``0.95``，对应显著门槛 ``alpha = 0.05``；不会再出现
  ``float("0.95") / 100 = 0.0095`` 这类口径错位。
* 每次检验产生一个「结果批次」，P 值、置信水平、显著门槛、样本批次和显著标记
  都在同一次检验里写入，后续评估 / 复盘只读批次，不再按任务当前设置重算。
* 登记按「任务 + 方法 + P 值 + 样本」做幂等：同一结果重复登记只会拿回原批次，
  不会生成第二份结论；登记中断后重试也走同一条去重路径。
* 批次保存在进程内注册表里（与示例数据仓库一致，重启后重置），
  完整的样本值只在批次内部留存，对外默认只返回样本摘要。
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import datetime
from typing import Any

DEFAULT_CONFIDENCE = 0.95


def now_text() -> str:
    """统一的时间文本，便于结果批次之间的先后比较。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_confidence(value: Any, *, default: float | None = None) -> float:
    """把各种写法的置信水平归一成 0~1 之间的小数。

    支持 ``0.95`` / ``"0.95"`` / ``"95%"`` / ``95`` / ``"95 %"`` / ``"95.0％"``。
    百分比写法与小数写法会得到完全相同的数值，因此显著门槛也完全一致。
    """
    if value is None or str(value).strip() == "":
        if default is not None:
            return default
        raise ValueError("置信水平为空，无法确定显著门槛")
    text = str(value).strip().replace("％", "%").replace(" ", "")
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法识别的置信水平：{value!r}") from exc
    if percent or number > 1:
        number /= 100.0
    if not math.isfinite(number) or not 0 < number < 1:
        raise ValueError(f"置信水平必须位于 0 与 1 之间：{value!r}")
    # 显著检验的置信水平低于 0.5（门槛 > 0.5）没有业务含义，按填写错误处理，
    # 不再像旧实现那样静默地夹到 0.01 或 0.5，避免悄悄换掉门槛。
    if number <= 0.5:
        raise ValueError(f"置信水平必须高于 50%：{value!r}")
    return round(number, 10)


def confidence_to_alpha(confidence: Any, *, default: float | None = None) -> float:
    """置信水平转显著门槛：0.95 → 0.05，0.99 → 0.01。"""
    value = parse_confidence(confidence, default=default)
    return round(1.0 - value, 10)


def format_confidence(confidence: float) -> str:
    """给展示层用的百分比写法。"""
    return f"{confidence * 100:g}%"


def stable_seed(*parts: Any) -> int:
    """跨进程稳定的随机种子（内置 hash() 带盐，不能用于可复盘的结果）。"""
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float_list(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("样本值必须是数组")
    result: list[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("样本值必须是有限数值")
        result.append(number)
    return result


def _describe_values(values: list[float], prefix: str) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n
    variance = (
        sum((value - mean) ** 2 for value in values) / (n - 1) if n > 1 else 0.0
    )
    return {
        f"n_{prefix}": n,
        f"mean_{prefix}": round(mean, 6),
        f"std_{prefix}": round(math.sqrt(variance), 6),
        f"min_{prefix}": round(min(values), 6),
        f"max_{prefix}": round(max(values), 6),
    }


def _option(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def normalize_sample(raw: Any) -> dict[str, Any] | None:
    """把登记/计算时传入的样本整理成可留存的样本批次。

    支持的形态：

    * ``{"control": [...], "treatment": [...]}``（也接受 对照组/实验组）；
    * ``{"n_control": 500, "n_treatment": 500, ...}`` 这类摘要；
    * 只带 ``sample_batch_id`` / ``批次`` 的样本引用。

    三种形态都会生成稳定的 ``sample_batch_id`` 与样本指纹；没有可用样本时返回
    ``None``，由调用方给出「没有数据可核」的原因。
    """
    if raw is None or raw == {} or raw == []:
        return None
    payload = raw if isinstance(raw, dict) else {"sample_batch_id": str(raw)}

    control = _option(payload, "control", "control_values", "对照组")
    treatment = _option(payload, "treatment", "treatment_values", "实验组")
    if (
        isinstance(control, (list, tuple))
        and isinstance(treatment, (list, tuple))
        and len(control) >= 2
        and len(treatment) >= 2
    ):
        control_values = _float_list(control)
        treatment_values = _float_list(treatment)
        rounded_control = [round(value, 6) for value in control_values]
        rounded_treatment = [round(value, 6) for value in treatment_values]
        sample_fp = _fingerprint(
            {"kind": "values", "control": rounded_control, "treatment": rounded_treatment}
        )
        sample = {
            **_describe_values(control_values, "control"),
            **_describe_values(treatment_values, "treatment"),
            "kind": "values",
            "sample_fingerprint": sample_fp,
            "control": control_values,
            "treatment": treatment_values,
        }
        sample["sample_batch_id"] = str(
            _option(payload, "sample_batch_id", "sampleBatchId", "批次", "批次标识")
            or f"SMP-{sample_fp[:10]}"
        )
        return sample

    n_control = _option(payload, "n_control", "control_n", "n_a", "sample_size_control")
    n_treatment = _option(
        payload, "n_treatment", "treatment_n", "n_b", "sample_size_treatment", "n_experiment"
    )
    if n_control is not None and n_treatment is not None:
        try:
            n_c = int(float(n_control))
            n_t = int(float(n_treatment))
        except (TypeError, ValueError):
            n_c, n_t = 0, 0
        if n_c > 0 and n_t > 0:
            summary: dict[str, Any] = {
                "kind": "summary",
                "n_control": n_c,
                "n_treatment": n_t,
            }
            mean_control = _option(payload, "mean_control", "mean_a")
            mean_treatment = _option(payload, "mean_treatment", "mean_b")
            std_control = _option(payload, "std_control", "std_a")
            std_treatment = _option(payload, "std_treatment", "std_b")
            if mean_control is not None:
                summary["mean_control"] = round(float(mean_control), 6)
            if mean_treatment is not None:
                summary["mean_treatment"] = round(float(mean_treatment), 6)
            if std_control is not None:
                summary["std_control"] = round(float(std_control), 6)
            if std_treatment is not None:
                summary["std_treatment"] = round(float(std_treatment), 6)
            sample_fp = _fingerprint({"kind": "summary", "sample": summary})
            summary["sample_fingerprint"] = sample_fp
            summary["sample_batch_id"] = str(
                _option(payload, "sample_batch_id", "sampleBatchId", "批次", "批次标识")
                or f"SMP-{sample_fp[:10]}"
            )
            return summary

    reference = _option(
        payload, "sample_batch_id", "sampleBatchId", "批次", "批次标识", "label", "reference"
    )
    if reference is not None:
        sample_fp = _fingerprint({"kind": "reference", "reference": str(reference)})
        return {
            "kind": "reference",
            "sample_batch_id": str(reference),
            "sample_fingerprint": sample_fp,
        }
    return None


def sample_summary(sample: dict[str, Any] | None) -> dict[str, Any] | None:
    """对外展示的样本摘要：不包含原始样本数组。"""
    if not sample:
        return None
    return {
        key: value
        for key, value in sample.items()
        if key not in {"control", "treatment"}
    }


class SignificanceResultRegistry:
    """可确认检验结果批次的进程内注册表。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._batches: dict[str, dict[str, Any]] = {}
        self._by_fingerprint: dict[str, str] = {}
        self._order: list[str] = []
        self._pending: dict[str, dict[str, Any]] = {}
        self._pending_order: list[str] = []
        self._sequence = 0

    def clear(self) -> None:
        """测试或重置用：清空全部批次。"""
        with self._lock:
            self._batches.clear()
            self._by_fingerprint.clear()
            self._order.clear()
            self._pending.clear()
            self._pending_order.clear()
            self._sequence = 0

    def register(
        self,
        *,
        task_id: int,
        experiment_ref: str,
        method: str,
        confidence: float,
        alpha: float,
        p_value: float,
        significant: bool,
        sample: dict[str, Any],
        source: str,
        status: str = "待确认",
        tested_at: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """登记一次检验结果；同一结果重复登记返回原批次。"""
        fingerprint = result_fingerprint(
            task_id=task_id,
            method=method,
            p_value=p_value,
            sample_fingerprint=str(sample.get("sample_fingerprint", "")),
        )
        with self._lock:
            existing_id = self._by_fingerprint.get(fingerprint)
            if existing_id:
                return self._batches[existing_id], False
            pending = self._pending.pop(fingerprint, None)
            if pending is not None:
                # 重试登记沿用「结果还在路上」时捕获的口径，不按当前设置再判一次。
                confidence = pending["confidence"]
                alpha = pending["alpha"]
                p_value = pending["p_value"]
                significant = pending["significant"]
                sample = pending["sample"]
                tested_at = pending.get("tested_at") or tested_at
                source = source or pending.get("source", "")
            self._sequence += 1
            batch_id = f"SIG-{task_id}-{self._sequence:04d}-{fingerprint[:8]}"
            batch = {
                "batch_id": batch_id,
                "sequence": self._sequence,
                "task_id": int(task_id),
                "experiment_ref": str(experiment_ref),
                "method": str(method),
                "confidence": round(float(confidence), 10),
                "alpha": round(float(alpha), 10),
                "p_value": round(float(p_value), 10),
                "significant": bool(significant),
                "status": status,
                "source": source,
                "sample_batch_id": sample.get("sample_batch_id"),
                "sample": sample,
                "sample_fingerprint": sample.get("sample_fingerprint", ""),
                "fingerprint": fingerprint,
                "created_at": now_text(),
                "confirmed_at": now_text() if status == "已确认" else None,
                "tested_at": tested_at or now_text(),
            }
            self._batches[batch_id] = batch
            self._by_fingerprint[fingerprint] = batch_id
            self._order.append(batch_id)
            return batch, True

    def mark_pending(
        self,
        *,
        task_id: int,
        experiment_ref: str,
        method: str,
        confidence: float,
        alpha: float,
        p_value: float,
        significant: bool,
        sample: dict[str, Any],
        source: str,
        tested_at: str | None = None,
    ) -> dict[str, Any]:
        """记录一份「结果还在路上」的检验结果（回执暂时缺失）。

        待确认结果不算结论、也不参与评估；重试登记成功后会转成正式批次。
        同一结果重复标记只更新时间，不会堆积多条。
        """
        fingerprint = result_fingerprint(
            task_id=task_id,
            method=method,
            p_value=p_value,
            sample_fingerprint=str(sample.get("sample_fingerprint", "")),
        )
        with self._lock:
            if fingerprint in self._by_fingerprint:
                return {
                    "fingerprint": fingerprint,
                    "batch_id": self._by_fingerprint[fingerprint],
                    "already_registered": True,
                }
            record = self._pending.get(fingerprint)
            if record is None:
                record = {
                    "pending_id": f"PEND-{int(task_id)}-{fingerprint[:8]}",
                    "task_id": int(task_id),
                    "experiment_ref": str(experiment_ref),
                    "method": str(method),
                    "confidence": round(float(confidence), 10),
                    "alpha": round(float(alpha), 10),
                    "p_value": round(float(p_value), 10),
                    "significant": bool(significant),
                    "sample": sample,
                    "sample_batch_id": sample.get("sample_batch_id"),
                    "sample_fingerprint": sample.get("sample_fingerprint", ""),
                    "fingerprint": fingerprint,
                    "source": source,
                    "status": "待回执",
                    "first_seen_at": now_text(),
                    "tested_at": tested_at or now_text(),
                }
                self._pending[fingerprint] = record
                self._pending_order.append(fingerprint)
            else:
                record["last_seen_at"] = now_text()
            return {
                "fingerprint": fingerprint,
                "pending_id": record["pending_id"],
                "record": record,
                "already_registered": False,
            }

    def pending_for_task(self, task_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._pending[fingerprint]
                for fingerprint in self._pending_order
                if fingerprint in self._pending
                and int(self._pending[fingerprint]["task_id"]) == int(task_id)
            ]

    def pending_for(
        self, *, task_id: int, method: str, p_value: float, sample_fingerprint: str
    ) -> dict[str, Any] | None:
        """按「任务 + 方法 + P 值 + 样本」找待回执记录。

        计算阶段记录的方法名可能是 ``welch_t_test``，登记时任务上写的是 ``t_test``；
        精确指纹找不到时，退回到「任务 + P 值 + 样本」匹配同一份结果。
        """
        fingerprint = result_fingerprint(
            task_id=task_id,
            method=method,
            p_value=p_value,
            sample_fingerprint=sample_fingerprint,
        )
        with self._lock:
            record = self._pending.get(fingerprint)
            if record is not None:
                return record
            for fingerprint in self._pending_order:
                candidate = self._pending.get(fingerprint)
                if candidate is None:
                    continue
                if (
                    int(candidate["task_id"]) == int(task_id)
                    and abs(float(candidate["p_value"]) - float(p_value)) < 1e-12
                    and candidate.get("sample_fingerprint") == str(sample_fingerprint)
                ):
                    return candidate
        return None

    def pending_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._pending[fingerprint]
                for fingerprint in self._pending_order
                if fingerprint in self._pending
            ]

    def get(self, batch_id: str | None) -> dict[str, Any] | None:
        if not batch_id:
            return None
        with self._lock:
            return self._batches.get(str(batch_id))

    def latest_for_task(self, task_id: int) -> dict[str, Any] | None:
        with self._lock:
            for batch_id in reversed(self._order):
                batch = self._batches[batch_id]
                if int(batch["task_id"]) == int(task_id):
                    return batch
        return None

    def for_task(self, task_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._batches[batch_id]
                for batch_id in self._order
                if int(self._batches[batch_id]["task_id"]) == int(task_id)
            ]

    def confirm(self, batch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            stored = self._batches.get(str(batch.get("batch_id")))
            if stored is None:
                return batch
            if stored.get("status") != "已确认":
                stored["status"] = "已确认"
                stored["confirmed_at"] = now_text()
            return stored

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._batches[batch_id] for batch_id in self._order]


def _canonical_method(method: Any) -> str:
    """结果身份里用的方法名：``t_test`` 与 ``welch_t_test``、卡方写法视为同一种检验。"""
    text = str(method).strip().lower()
    if "chi" in text or "卡方" in str(method):
        return "chi_square"
    if "t_test" in text or "welch" in text or "t检验" in str(method) or "t 检验" in str(method):
        return "t_test"
    return text


def result_fingerprint(
    *, task_id: int, method: str, p_value: float, sample_fingerprint: str
) -> str:
    """结果身份：任务 + 方法 + P 值 + 样本。

    刻意不包含置信水平：同一份检验结果在管理员改过置信水平后重试登记时，
    仍然命中原来的批次，不会按新门槛再生成一份结论。
    """
    return _fingerprint(
        {
            "task_id": int(task_id),
            "method": _canonical_method(method),
            "p_value": round(float(p_value), 10),
            "sample": str(sample_fingerprint),
        }
    )


def public_batch(batch: dict[str, Any], *, include_values: bool = False) -> dict[str, Any]:
    """对外的批次视图；默认只给样本摘要，需要复盘原始样本时可显式展开。"""
    view = {
        key: value
        for key, value in batch.items()
        if key not in {"sample", "sample_fingerprint"}
    }
    sample = batch.get("sample") or {}
    sample_view = sample_summary(sample) or {}
    if include_values:
        if "control" in sample:
            sample_view["control"] = sample["control"]
            sample_view["control_values"] = sample["control"]
        if "treatment" in sample:
            sample_view["treatment"] = sample["treatment"]
            sample_view["treatment_values"] = sample["treatment"]
    view["sample"] = sample_view
    return view


result_registry = SignificanceResultRegistry()
