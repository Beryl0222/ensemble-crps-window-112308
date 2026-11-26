"""带权经验 CRPS 及其对成员预报值的解析导数。

记号：成员 ``i`` 的概率权重 ``p_i``（中心和中心内均分合成，和为 1），
预报值 ``x_i``，观测 ``y``。

* 观测距离项 ``O = sum_i p_i |x_i - y|``；
* 成对项 ``S = sum_{i != j} p_i p_j |x_i - x_j|``，按不同成员总权
  ``D = sum_{i != j} p_i p_j = 1 - sum_i p_i^2`` 归一；
* ``CRPS = O - S / (2 D)``。

``i != j`` 是有序不同成员对。两个成员等权时就是通常的 fair CRPS；
单个成员时 ``D = 0``，成对项不可定义，调用方不得交付损失。

导数
----
``dCRPS/dx_i = p_i * (sgn(x_i - y) - (1/D) * sum_{j != i} p_j sgn(x_i - x_j))``

值相等的成员对贡献零差，双方按对称方式各得 0；观测恰好等于预报值
（``x_i == y``）时绝对值导数取 0。相同预报值的不同成员各自保留自己的
``p_i`` 与导数，不会因去重而合并。

实现按不同预报值分组、以有序前缀和求成对项，时间 O(n log n)、额外空间
O(n)，不构造任何成员对矩阵；大量重复值只占一个值组。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CrpsResult:
    """一个格点上带权 CRPS 的分解结果。"""

    loss: float
    observation_term: float
    pairwise_term: float
    pairwise_normalizer: float
    gradients: tuple[float, ...]
    ordered_pairs: int
    distinct_values: int


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def weighted_crps(
    values: Sequence[float],
    probabilities: Sequence[float],
    observation: float,
) -> CrpsResult:
    """计算带权 fair CRPS、各项分解与每个成员的解析导数。

    ``values`` 与 ``probabilities`` 一一对应且顺序即返回导数的顺序。要求
    至少两个正权重成员，否则成对项无定义。
    """

    values = [float(v) for v in values]
    probabilities = [float(p) for p in probabilities]
    if len(values) != len(probabilities):
        raise ValueError("values 与 probabilities 长度必须一致")
    if len(values) < 2:
        raise ValueError(
            "成对项至少需要两个成员；共同支持不足时不得交付校准损失"
        )
    if any(not np.isfinite(v) for v in values) or not np.isfinite(observation):
        raise ValueError("预报值与观测必须是有限数值")
    if any(not np.isfinite(p) or p < 0.0 for p in probabilities):
        raise ValueError("成员权重必须是非负有限数值")
    total = float(sum(probabilities))
    if total <= 0.0:
        raise ValueError("成员权重之和必须为正")

    p = [prob / total for prob in probabilities]
    normalized_total = 1.0
    n = len(p)

    # 按值分组。组内成员值相同，彼此的差为零，不贡献成对项和符号和。
    order = sorted(range(n), key=lambda i: values[i])
    groups: list[dict[str, object]] = []
    for index in order:
        value = values[index]
        if groups and groups[-1]["value"] == value:
            groups[-1]["indices"].append(index)  # type: ignore[union-attr]
            groups[-1]["weight"] = float(groups[-1]["weight"]) + p[index]  # type: ignore[arg-type]
        else:
            groups.append({"value": value, "indices": [index], "weight": p[index]})

    pairwise_normalizer = 1.0 - float(sum(pi * pi for pi in p))
    if pairwise_normalizer <= 0.0:
        raise ValueError(
            "成对项归一为零：不同成员的总权不足，不得交付校准损失"
        )

    observation_term = float(sum(pi * abs(xi - observation) for pi, xi in zip(p, values)))

    # S = sum_{i != j} p_i p_j |x_i - x_j|
    #   = 2 * sum_{a < b} q_a q_b (v_b - v_a)，q 为值组总权。
    pairwise_sum = 0.0
    weight_below = 0.0
    moment_below = 0.0
    for group in groups:
        value = float(group["value"])  # type: ignore[arg-type]
        weight = float(group["weight"])  # type: ignore[arg-type]
        pairwise_sum += 2.0 * weight * (value * weight_below - moment_below)
        weight_below += weight
        moment_below += weight * value

    pairwise_term = pairwise_sum / pairwise_normalizer
    loss = observation_term - 0.5 * pairwise_term

    gradients = [0.0] * n
    weight_below = 0.0
    for position, group in enumerate(groups):
        value = float(group["value"])  # type: ignore[arg-type]
        weight = float(group["weight"])  # type: ignore[arg-type]
        weight_above = normalized_total - weight_below - weight
        signed_pair_weight = weight_below - weight_above
        obs_sign = _sign(value - observation)
        for index in group["indices"]:  # type: ignore[union-attr]
            gradients[index] = p[index] * (
                obs_sign - signed_pair_weight / pairwise_normalizer
            )
        weight_below += weight

    return CrpsResult(
        loss=loss,
        observation_term=observation_term,
        pairwise_term=pairwise_term,
        pairwise_normalizer=pairwise_normalizer,
        gradients=tuple(gradients),
        ordered_pairs=n * (n - 1),
        distinct_values=len(groups),
    )


def weighted_crps_naive(
    values: Sequence[float], probabilities: Sequence[float], observation: float
) -> CrpsResult:
    """显式双重循环的参考实现，只用于核对，不可用于大成员集合。"""

    values = [float(v) for v in values]
    p = [float(x) for x in probabilities]
    total = float(sum(p))
    p = [x / total for x in p]
    n = len(p)
    if n < 2:
        raise ValueError("至少两个成员")
    normalizer = float(sum(p[i] * p[j] for i in range(n) for j in range(n) if i != j))
    pairwise_sum = float(
        sum(p[i] * p[j] * abs(values[i] - values[j]) for i in range(n) for j in range(n) if i != j)
    )
    observation_term = float(sum(p[i] * abs(values[i] - observation) for i in range(n)))
    gradients = []
    for i in range(n):
        pair_sign = sum(
            p[j] * _sign(values[i] - values[j]) for j in range(n) if j != i
        )
        gradients.append(
            p[i] * (_sign(values[i] - observation) - pair_sign / normalizer)
        )
    return CrpsResult(
        loss=observation_term - 0.5 * (pairwise_sum / normalizer),
        observation_term=observation_term,
        pairwise_term=pairwise_sum / normalizer,
        pairwise_normalizer=normalizer,
        gradients=tuple(gradients),
        ordered_pairs=n * (n - 1),
        distinct_values=len(set(values)),
    )
