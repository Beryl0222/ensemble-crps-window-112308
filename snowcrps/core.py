"""按格点的加权 CRPS 结算、增量更新与校准器交接。

设计要点
--------
* 结算只按格点进行。每个格点按 ``cutoff`` 取各成员的最新修订，质量标志为
  true 的成员才有效；参与中心的权重在参与中心之间重新归一，中心内再按有效
  成员均分。不同格点的有效成员集合可以完全不同。
* 成对项要求至少两个有效成员。共同支持不足时结算状态为
  ``insufficient_support``，不给出损失；``cutoff`` 早于 seal 记为
  ``pending``（尚可补齐），否则记为 ``formal``（按截止正式缺失）。
* 观测无效的格点状态为 ``no_observation``，不进入聚合分母。
* 每次状态变化（成员值、质量、成员集合、中心权重）都会改变结算的
  ``token``。校准器只能消费与当前结算 ``token`` 一致的成员梯度。
* 增量更新只重读受影响中心在受影响格点上的资料；其他中心使用项目端保留
  的归约切片，不产生任何原始成员读取，也不构造成员对矩阵。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .crps import weighted_crps
from .exchange import (
    Exchange,
    ReadRecord,
    ReadWindow,
    RevisionReader,
    summarize_revision,
)

# -- 状态 -------------------------------------------------------------------
DELIVERED = "delivered"
INSUFFICIENT_SUPPORT = "insufficient_support"
NO_OBSERVATION = "no_observation"

# -- 可得性 -----------------------------------------------------------------
PENDING = "pending"  # cutoff 早于 seal，后续成员可能补齐
FORMAL = "formal"  # cutoff 达到 seal，按承诺成员集合定稿
SEALED = "sealed"  # 已封存并回查过

# -- 暴露口径 ---------------------------------------------------------------
EXPOSURE_FORMAL = "formal"
EXPOSURE_PREVIEW = "preview"

EXPOSURE_FIELDS: Mapping[str, frozenset[str]] = {
    EXPOSURE_FORMAL: frozenset(
        {
            "grid",
            "cutoff",
            "loss",
            "observation_term",
            "pairwise_term",
            "pairwise_normalizer",
            "members",
            "centers",
            "gradients",
            "token",
            "status",
            "availability",
        }
    ),
    EXPOSURE_PREVIEW: frozenset(
        {
            "grid",
            "cutoff",
            "loss",
            "members",
            "centers",
            "token",
            "status",
            "availability",
        }
    ),
}


class LedgerError(RuntimeError):
    """结算或交接状态不允许当前操作。"""


class StaleGradientError(LedgerError):
    """成员梯度来自旧状态，不能用于当前轮次。"""


@dataclass(frozen=True)
class CenterShare:
    """一个参与中心在该格点结算中的权重归属。"""

    center: str
    raw_weight: float
    weight: float
    members: tuple[tuple[str, float], ...]  # (member, value)

    @property
    def effective_count(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class MemberShare:
    """一个有效成员在结算中的归属。"""

    center: str
    member: str
    value: float
    inter_center_weight: float
    weight: float
    rank: int
    group: int  # 相同预报值共享的组号

    @property
    def member_id(self) -> tuple[str, str]:
        return (self.center, self.member)


@dataclass(frozen=True)
class Settlement:
    """一个格点在一个轮次上的结算结果。

    ``loss`` 及 ``gradients`` 为 ``None`` 表示该状态不可交付校准损失；
    原因在 ``reason``，可得性在 ``availability``。``members`` 与 ``centers``
    无论是否可交付都会给出，便于核对覆盖和权重归属。
    """

    grid: str
    cutoff: int
    status: str
    availability: str
    loss: float | None
    observation_term: float | None
    pairwise_term: float | None
    pairwise_normalizer: float | None
    members: tuple[MemberShare, ...]
    centers: tuple[CenterShare, ...]
    gradients: tuple[float, ...] | None
    token: str
    epoch: int
    reason: str
    exposure: str

    @property
    def deliverable(self) -> bool:
        """损失是否可交付；预览可交付损失但不含梯度。"""

        return self.loss is not None

    @property
    def has_gradients(self) -> bool:
        return self.gradients is not None

    @property
    def member_ids(self) -> tuple[tuple[str, str], ...]:
        return tuple(m.member_id for m in self.members)

    @property
    def effective_count(self) -> int:
        return len(self.members)

    @property
    def center_coverage(self) -> tuple[str, ...]:
        return tuple(center.center for center in self.centers)

    @property
    def fillable(self) -> bool:
        """共同支持不足但截止未到：后续成员仍可能补齐。"""

        return self.status == INSUFFICIENT_SUPPORT and self.availability == PENDING

    @property
    def formally_missing(self) -> bool:
        """按截止正式缺失：共同支持不足且已到 seal。"""

        return self.status == INSUFFICIENT_SUPPORT and self.availability == FORMAL

    def gradient_for(self, center: str, member: str) -> float:
        if self.gradients is None:
            raise LedgerError(
                f"{self.grid}@{self.cutoff} 的 {self.status}/{self.availability} "
                "没有可回传的成员梯度"
            )
        for share, gradient in zip(self.members, self.gradients):
            if share.center == center and share.member == member:
                return gradient
        raise KeyError(f"{self.grid}@{self.cutoff} 没有成员 ({center}, {member})")

    def exposed(self) -> dict[str, object]:
        """按暴露口径返回可以交给校准器的量。"""

        allowed = EXPOSURE_FIELDS[self.exposure]
        values: dict[str, object] = {
            "grid": self.grid,
            "cutoff": self.cutoff,
            "status": self.status,
            "availability": self.availability,
            "loss": self.loss,
            "observation_term": self.observation_term,
            "pairwise_term": self.pairwise_term,
            "pairwise_normalizer": self.pairwise_normalizer,
            "members": self.members,
            "centers": self.centers,
            "gradients": self.gradients,
            "token": self.token,
        }
        return {key: values[key] for key in allowed}


@dataclass(frozen=True)
class Aggregate:
    """跨格点结算：只在可交付且有观测的格点上按 grid_area 加权。"""

    loss: float | None
    weighted_area: float
    used: tuple[str, ...]
    excluded: Mapping[str, str]


@dataclass(frozen=True)
class SettlementBook:
    """一个轮次上全部格点的结算，可按格点挑选。"""

    cutoff: int
    settlements: Mapping[str, Settlement]

    def __getitem__(self, grid: str) -> Settlement:
        return self.settlements[grid]

    @property
    def delivered(self) -> tuple[str, ...]:
        return tuple(g for g in sorted(self.settlements) if self.settlements[g].deliverable)

    def select(
        self, grids: Iterable[str] | None = None, *, require_deliverable: bool = False
    ) -> "SettlementBook":
        if grids is None:
            chosen = dict(self.settlements)
        else:
            wanted = list(grids)
            missing = set(wanted) - set(self.settlements)
            if missing:
                raise KeyError(f"结算中没有格点: {sorted(missing)}")
            chosen = {g: self.settlements[g] for g in wanted}
        if require_deliverable:
            for grid, settlement in chosen.items():
                if not settlement.deliverable:
                    raise LedgerError(
                        f"{grid} 不可交付: {settlement.status}/{settlement.availability}"
                    )
        return SettlementBook(self.cutoff, chosen)

    def aggregate(
        self, area: Mapping[str, float], *, allowed: Iterable[str] | None = None
    ) -> Aggregate:
        """按面积加权；分母只含实际使用的可交付格点。"""

        permitted = None if allowed is None else set(allowed)
        numerator = 0.0
        denominator = 0.0
        used: list[str] = []
        excluded: dict[str, str] = {}
        for grid in sorted(self.settlements):
            settlement = self.settlements[grid]
            if permitted is not None and grid not in permitted:
                excluded[grid] = "not_selected"
                continue
            if not settlement.deliverable:
                excluded[grid] = f"{settlement.status}/{settlement.availability}"
                continue
            weight = float(area.get(grid, 0.0))
            if weight <= 0.0:
                excluded[grid] = "zero_area"
                continue
            numerator += weight * float(settlement.loss)
            denominator += weight
            used.append(grid)
        loss = numerator / denominator if denominator > 0.0 else None
        return Aggregate(
            loss=loss,
            weighted_area=denominator,
            used=tuple(used),
            excluded=excluded,
        )


@dataclass(frozen=True)
class GradientPacket:
    """可回传给校准器的成员梯度，绑定产生它的结算状态。"""

    grid: str
    cutoff: int
    token: str
    epoch: int
    member_ids: tuple[tuple[str, str], ...]
    gradients: tuple[float, ...]


@dataclass(frozen=True)
class TrainingBatch:
    """一份可供校准器继续训练的结算集合。

    只包含可交付格点；每个 ``GradientPacket`` 都带 ``token``，消费前需经
    :meth:`Ledger.accept_packet` 核对当前状态。
    """

    cutoff: int
    packets: tuple[GradientPacket, ...]
    aggregate: Aggregate

    def packet(self, grid: str) -> GradientPacket:
        for packet in self.packets:
            if packet.grid == grid:
                return packet
        raise KeyError(f"训练批次中没有格点 {grid!r}")

    @property
    def grids(self) -> tuple[str, ...]:
        return tuple(packet.grid for packet in self.packets)


@dataclass(frozen=True)
class CenterSlice:
    """项目端保留的一个中心在一个格点上的有效成员归约。"""

    center: str
    grid: str
    cutoff: int
    members: tuple[tuple[str, float], ...]  # (member, value)，按成员名排序
    revisions: tuple[tuple[str, int], ...]
    epoch: int

    @property
    def member_ids(self) -> tuple[tuple[str, str], ...]:
        return tuple((self.center, member) for member, _ in self.members)


@dataclass(frozen=True)
class UpdateReport:
    """一次增量更新的结果与读取记账。"""

    grid: str
    center: str
    arrival: int
    cutoff: int
    settlement: Settlement
    previous: Settlement | None
    own_raw_members: int
    other_center_raw_members: int
    summary_page_reads: int
    log: tuple[ReadRecord, ...]

    @property
    def status_changed(self) -> bool:
        return self.previous is None or self.previous.status != self.settlement.status

    @property
    def membership_changed(self) -> bool:
        if self.previous is None:
            return True
        return set(self.previous.member_ids) != set(self.settlement.member_ids)

    @property
    def ranking_changed(self) -> bool:
        if self.previous is None:
            return self.membership_changed
        return tuple((m.center, m.member, m.rank) for m in self.previous.members) != tuple(
            (m.center, m.member, m.rank) for m in self.settlement.members
        )

    @property
    def weights_changed(self) -> bool:
        if self.previous is None:
            return True
        return tuple((c.center, c.weight) for c in self.previous.centers) != tuple(
            (c.center, c.weight) for c in self.settlement.centers
        )


@dataclass(frozen=True)
class SealedRound:
    """封存轮次：承诺成员集合及其结算快照。"""

    cutoff: int
    settlements: Mapping[str, Settlement]
    committed: Mapping[str, tuple[tuple[str, str, float], ...]]
    token: str


class Ledger:
    """一个交换资料上的格点结算账本。

    构造时不读取任何原始成员。``prepare`` / ``settle`` 只用项目端保留的
    归约资料；``apply_arrival`` / ``apply_change`` 只重读受影响中心在受影响
    格点上的原始资料，其余中心沿用保留切片。
    """

    def __init__(
        self,
        exchange: Exchange,
        *,
        page_size: int = 64,
        summary_page_budget: int | None = None,
    ) -> None:
        self.exchange = exchange
        self.page_size = max(1, int(page_size))
        budget = exchange.max_update_work.get("other_summary_page_reads")
        self.summary_page_budget = 128 if summary_page_budget is None else int(
            summary_page_budget
        )
        self.other_member_budget = int(exchange.max_update_work.get("other_member_scan", 0))
        self._slices: dict[tuple[str, str, int], CenterSlice] = {}
        self._epoch: dict[str, int] = {grid: 0 for grid in exchange.grids()}
        self._sealed: SealedRound | None = None
        self._history: dict[str, list[Settlement]] = {grid: [] for grid in exchange.grids()}
        self._last: dict[str, Settlement] = {}
        self._round: dict[str, int] = {grid: 0 for grid in exchange.grids()}
        # 记录已经应用过到达的 (格点, 中心, 轮次)，用于还原“本次变更之前”的状态。
        self._applied: set[tuple[str, str, int]] = set()

    # -- 内部：切片与结算 ---------------------------------------------
    def _material_slice(self, grid: str, center: str, cutoff: int) -> CenterSlice:
        """从项目端已归约资料构造切片，不产生校准器读取。"""

        if grid not in self.exchange.grid_area:
            raise KeyError(f"未登记格点: {grid!r}")
        if center not in self.exchange.center_weight:
            raise KeyError(f"未登记中心: {center!r}")
        key = (grid, center, cutoff)
        cached = self._slices.get(key)
        if cached is not None:
            return cached
        chosen = [
            r
            for r in self.exchange.selection(grid, cutoff)
            if r.center == center and r.quality
        ]
        slice_ = CenterSlice(
            center=center,
            grid=grid,
            cutoff=cutoff,
            members=tuple(sorted((r.member, r.value) for r in chosen)),
            revisions=tuple(sorted((r.member, r.revision) for r in chosen)),
            epoch=self._epoch[grid],
        )
        self._slices[key] = slice_
        return slice_

    def slices(self, grid: str, cutoff: int) -> tuple[CenterSlice, ...]:
        return tuple(
            self._material_slice(grid, center, cutoff) for center in self.exchange.centers()
        )

    def _record(
        self, grid: str, settlement: Settlement, cutoff: int | None = None
    ) -> Settlement:
        previous = self._last.get(grid)
        if previous is None or previous.token != settlement.token:
            self._history[grid].append(settlement)
        round_now = self._round.get(grid, 0)
        if cutoff is not None:
            self._round[grid] = max(round_now, cutoff)
        if cutoff is None or cutoff >= round_now:
            self._last[grid] = settlement
        return settlement

    def _compose(
        self,
        grid: str,
        cutoff: int,
        exposure: str,
        slices: Sequence[CenterSlice],
        *,
        epoch: int | None = None,
    ) -> Settlement:
        if grid not in self.exchange.grid_area:
            raise KeyError(f"未登记格点: {grid!r}")
        self._check_cutoff(cutoff, exposure)
        epoch = self._epoch[grid] if epoch is None else int(epoch)
        availability = PENDING if cutoff < self.exchange.seal else FORMAL

        observation = self.exchange.observation_value[grid]
        valid_observation = self.exchange.observation_valid[grid]

        populated = [s for s in slices if s.members]
        raw_total = sum(self.exchange.center_weight[s.center] for s in populated)
        centers: list[CenterShare] = []
        if raw_total > 0.0:
            for slice_ in sorted(populated, key=lambda s: s.center):
                raw = self.exchange.center_weight[slice_.center]
                centers.append(
                    CenterShare(
                        center=slice_.center,
                        raw_weight=raw,
                        weight=raw / raw_total,
                        members=tuple(slice_.members),
                    )
                )

        members: list[MemberShare] = []
        for center_share in centers:
            per_member = center_share.weight / len(center_share.members)
            for member, value in center_share.members:
                members.append(
                    MemberShare(
                        center=center_share.center,
                        member=member,
                        value=value,
                        inter_center_weight=center_share.weight,
                        weight=per_member,
                        rank=0,
                        group=0,
                    )
                )

        ordered = sorted(members, key=lambda m: (m.value, m.center, m.member))
        rank_of: dict[tuple[str, str], int] = {}
        group_of: dict[tuple[str, str], int] = {}
        group_index = -1
        rank = 0
        previous_value: float | None = None
        for share in ordered:
            if previous_value is None or share.value != previous_value:
                group_index += 1
                rank += 1
                previous_value = share.value
            group_of[share.member_id] = group_index
            rank_of[share.member_id] = rank
        ranked = tuple(
            MemberShare(
                center=m.center,
                member=m.member,
                value=m.value,
                inter_center_weight=m.inter_center_weight,
                weight=m.weight,
                rank=rank_of[m.member_id],
                group=group_of[m.member_id],
            )
            for m in ordered
        )

        base = {
            "grid": grid,
            "cutoff": cutoff,
            "members": ranked,
            "centers": tuple(centers),
            "epoch": epoch,
            "exposure": exposure,
        }

        if not valid_observation:
            return self._result(
                base,
                status=NO_OBSERVATION,
                availability=availability,
                reason="观测无效，不参加结算也不进入聚合分母",
            )

        if not ranked:
            return self._result(
                base,
                status=INSUFFICIENT_SUPPORT,
                availability=availability,
                reason="没有有效成员，无法定义分布项",
            )

        normalizer = 1.0 - sum(m.weight * m.weight for m in ranked)
        if normalizer <= 0.0:
            return self._result(
                base,
                status=INSUFFICIENT_SUPPORT,
                availability=availability,
                reason="只有一个有效成员，成对项共同支持不足",
            )

        result = weighted_crps(
            [m.value for m in ranked],
            [m.weight for m in ranked],
            observation,
        )
        return self._result(
            base,
            status=DELIVERED,
            availability=availability,
            reason="",
            loss=result.loss,
            observation_term=result.observation_term,
            pairwise_term=result.pairwise_term,
            pairwise_normalizer=result.pairwise_normalizer,
            gradients=result.gradients if exposure == EXPOSURE_FORMAL else None,
        )

    def _result(
        self,
        base: Mapping[str, object],
        *,
        status: str,
        availability: str,
        reason: str,
        loss: float | None = None,
        observation_term: float | None = None,
        pairwise_term: float | None = None,
        pairwise_normalizer: float | None = None,
        gradients: tuple[float, ...] | None = None,
    ) -> Settlement:
        members = base["members"]
        assert isinstance(members, tuple)
        token = _token(
            base["grid"],
            base["cutoff"],
            status,
            availability,
            loss,
            observation_term,
            pairwise_term,
            pairwise_normalizer,
            members,
        )
        return Settlement(
            grid=str(base["grid"]),
            cutoff=int(base["cutoff"]),
            status=status,
            availability=availability,
            loss=loss,
            observation_term=observation_term,
            pairwise_term=pairwise_term,
            pairwise_normalizer=pairwise_normalizer,
            members=members,
            centers=base["centers"],  # type: ignore[arg-type]
            gradients=gradients,
            token=token,
            epoch=int(base["epoch"]),
            reason=reason,
            exposure=str(base["exposure"]),
        )

    def _check_cutoff(self, cutoff: int, exposure: str) -> None:
        if cutoff < 1:
            raise LedgerError(f"轮次必须为正: {cutoff!r}")
        if exposure == EXPOSURE_FORMAL and cutoff > self.exchange.seal:
            raise LedgerError(
                f"正式结算不能越过 seal={self.exchange.seal}（请求 {cutoff}）；"
                "seal 之后的修订只可能在预览口径中看到"
            )
        if exposure not in EXPOSURE_FIELDS:
            raise ValueError(f"未知暴露口径: {exposure!r}")

    # -- 公开：准备与结算 ---------------------------------------------
    def prepare(self, cutoff: int | None = None) -> SettlementBook:
        """建立指定轮次的全部格点结算（只使用项目端归约资料）。"""

        return self.book(cutoff, exposure=EXPOSURE_FORMAL)

    def settle(
        self,
        grid: str,
        cutoff: int | None = None,
        *,
        exposure: str = EXPOSURE_FORMAL,
    ) -> Settlement:
        """结算一个格点；不改变保留切片，也不产生原始成员读取。"""

        cutoff = self.exchange.seal if cutoff is None else int(cutoff)
        settlement = self._compose(grid, cutoff, exposure, self.slices(grid, cutoff))
        if exposure == EXPOSURE_FORMAL:
            return self._record(grid, settlement, cutoff)
        return settlement

    def receipt_rounds(self, *, include_post_seal: bool = False) -> tuple[int, ...]:
        """校准系统的收件时点；默认只列 seals 之内的正式收件轮次。"""

        seal = self.exchange.seal
        return tuple(
            q for q in self.exchange.query if include_post_seal or q <= seal
        )

    def receipt_books(
        self, *, include_post_seal: bool = False
    ) -> dict[int, SettlementBook]:
        """按收件时点逐个建立结算，便于调用者挑选一份继续训练。"""

        books: dict[int, SettlementBook] = {}
        for cutoff in self.receipt_rounds(include_post_seal=include_post_seal):
            exposure = (
                EXPOSURE_FORMAL if cutoff <= self.exchange.seal else EXPOSURE_PREVIEW
            )
            books[cutoff] = self.book(cutoff, exposure=exposure)
        return books

    def book(
        self, cutoff: int | None = None, *, exposure: str = EXPOSURE_FORMAL
    ) -> SettlementBook:
        cutoff = self.exchange.seal if cutoff is None else int(cutoff)
        return SettlementBook(
            cutoff,
            {
                grid: self.settle(grid, cutoff, exposure=exposure)
                for grid in self.exchange.grids()
            },
        )

    # -- 公开：增量更新 -----------------------------------------------
    def _slice_before(
        self, grid: str, center: str, cutoff: int, arrival: int
    ) -> CenterSlice:
        """还原某次到达之前该中心在该格点上的有效成员，全部来自归约资料。"""

        chosen = [
            r
            for r in self.exchange.selection(grid, cutoff, before_arrival=arrival)
            if r.center == center and r.quality
        ]
        return CenterSlice(
            center=center,
            grid=grid,
            cutoff=cutoff,
            members=tuple(sorted((r.member, r.value) for r in chosen)),
            revisions=tuple(sorted((r.member, r.revision) for r in chosen)),
            epoch=self._epoch[grid],
        )

    def _refresh_slice(
        self,
        grid: str,
        center: str,
        cutoff: int,
        reader: RevisionReader | None,
        *,
        arrival: int,
        window: ReadWindow | None,
    ) -> tuple[CenterSlice, int]:
        """重建一个中心在一个格点上的切片。

        有授权读取窗口时只读取该到达涉及的成员，读取上限等于受影响成员数，
        不随该中心或其他中心的成员总数增长；没有窗口时沿用项目端归约资料。
        读取严格限定在 ``(center, grid)`` 内。
        """

        if not self._holds_slice(grid, center, cutoff):
            raise LedgerError(
                f"({center}, {grid})@{cutoff} 没有可用的归约切片；"
                "该中心在此轮次尚未交付，也没有首次到达或补发记录"
            )
        affected = sorted(
            {
                r.member
                for r in self.exchange.deliveries(arrival)
                if r.center == center and r.grid == grid
            }
        )
        own_members = 0
        if window is None:
            # 没有授权窗口：沿用项目端保留的归约切片。
            effective = [
                r
                for r in self.exchange.selection(grid, cutoff)
                if r.center == center and r.quality
            ]
            members = {r.member: (r.value, r.revision) for r in effective}
        else:
            if reader is None:
                raise LedgerError(
                    f"({center}, {grid}) 在到达 {arrival} 需要按窗口读取原始成员，"
                    "但没有提供读取器"
                )
            # 保留切片给出未受影响成员，窗口只读取本次受影响的成员。
            retained_slice = self._slice_before(grid, center, cutoff, arrival)
            revision_of = dict(retained_slice.revisions)
            retained = {
                member: (value, revision_of.get(member, 0))
                for member, value in retained_slice.members
            }
            mark = reader.mark()
            with reader.restricted(center, [grid]):
                readings = {
                    member: reader.read_member(center, member, grid)
                    for member in affected
                }
            records = reader.after(mark)
            own_members = len(affected)
            other_raw = sum(
                record.count
                for record in records
                if record.kind in {"member", "center_grid"} and record.center != center
            )
            if other_raw or reader.other_center_reads > self.other_member_budget:
                raise LedgerError("增量更新读取了其他中心的原始成员，违反预算约定")

            members = dict(retained)
            for member, revisions in readings.items():
                visible = [r for r in revisions if r.arrival <= cutoff]
                if not visible:
                    continue
                latest = max(visible, key=lambda r: r.revision)
                if latest.quality:
                    members[member] = (latest.value, latest.revision)
                else:
                    members.pop(member, None)

        slice_ = CenterSlice(
            center=center,
            grid=grid,
            cutoff=cutoff,
            members=tuple(sorted((member, value) for member, (value, _) in members.items())),
            revisions=tuple(
                sorted((member, revision) for member, (_, revision) in members.items())
            ),
            epoch=self._epoch[grid] + 1,
        )
        return slice_, own_members

    def _window_for(self, arrival: int, center: str, grid: str) -> ReadWindow | None:
        for window in self.exchange.windows_at(arrival):
            if window.center == center and grid in window.grids:
                return window
        return None

    def _holds_slice(self, grid: str, center: str, cutoff: int) -> bool:
        if (grid, center, cutoff) in self._slices:
            return True
        return any(
            r.arrival <= cutoff
            for r in self.exchange.revisions_for_center_grid(center, grid)
        )

    def apply_change(
        self,
        grid: str,
        center: str,
        cutoff: int,
        arrival: int,
        *,
        reader: RevisionReader | None = None,
        exposure: str = EXPOSURE_FORMAL,
        verify_other_centers: bool = False,
    ) -> UpdateReport:
        """应用一个到达对某中心某格点的影响，并只重算该格点结算。

        若有授权读取窗口，只读取该中心在该格点上的原始成员；否则使用保留
        的归约资料。值变更或单成员退出的读取量只与该中心在该格点上的成员
        数有关，不随其他中心成员总数增长。``verify_other_centers`` 时以摘要
        页核对其他中心的覆盖，页数受 ``max_update_work`` 约束。
        """

        if arrival > cutoff:
            raise LedgerError(
                f"到达 {arrival} 不在轮次 {cutoff} 的视野内（到达必须不晚于截止）"
            )
        epoch_before = self._epoch[grid]
        previous = self._compose(
            grid,
            cutoff,
            exposure,
            tuple(
                self._slice_before(grid, c, cutoff, arrival)
                if c == center or (grid, c, cutoff) not in self._applied
                else self._material_slice(grid, c, cutoff)
                for c in self.exchange.centers()
            ),
            epoch=self._epoch[grid],
        )
        self._epoch[grid] = epoch_before + 1
        record_mark = reader.mark() if reader is not None else None
        try:
            slice_, own_members = self._refresh_slice(
                grid,
                center,
                cutoff,
                reader,
                arrival=arrival,
                window=self._window_for(arrival, center, grid),
            )
            self._slices[(grid, center, cutoff)] = slice_
            self._applied.add((grid, center, cutoff))
            summary_reads = 0
            if verify_other_centers and reader is not None:
                summary_reads = self._verify_other_centers(center, reader)
            settlement = self._compose(
                grid, cutoff, exposure, self.slices(grid, cutoff)
            )
        except Exception:
            self._epoch[grid] = epoch_before
            raise

        log = reader.after(record_mark) if reader is not None and record_mark else ()
        other_raw = sum(
            record.count
            for record in log
            if record.kind in {"member", "center_grid"} and record.center != center
        )
        if exposure == EXPOSURE_FORMAL:
            # 预览更新不推进正式轮次，也不改变已发出的正式梯度包状态。
            settlement = self._record(grid, settlement, cutoff)
        return UpdateReport(
            grid=grid,
            center=center,
            arrival=arrival,
            cutoff=cutoff,
            settlement=settlement,
            previous=previous,
            own_raw_members=own_members,
            other_center_raw_members=other_raw,
            summary_page_reads=summary_reads,
            log=log,
        )

    def apply_arrival(
        self,
        arrival: int,
        cutoff: int | None = None,
        *,
        reader: RevisionReader | None = None,
        exposure: str = EXPOSURE_FORMAL,
        verify_other_centers: bool = False,
    ) -> tuple[UpdateReport, ...]:
        """处理一个到达序号涉及的全部 (格点, 中心)，各自独立限制读取范围。"""

        cutoff = self.exchange.seal if cutoff is None else int(cutoff)
        affected: list[tuple[str, str]] = []
        for revision in self.exchange.deliveries(arrival):
            if revision.arrival <= cutoff:
                pair = (revision.grid, revision.center)
                if pair not in affected:
                    affected.append(pair)
        return tuple(
            self.apply_change(
                grid,
                center,
                cutoff,
                arrival,
                reader=reader,
                exposure=exposure,
                verify_other_centers=verify_other_centers,
            )
            for grid, center in sorted(affected)
        )

    def _verify_other_centers(self, own_center: str, reader: RevisionReader) -> int:
        """以摘要页核对其他中心，单次更新页数不超过约定预算。"""

        reads = 0
        for center in self.exchange.centers():
            if center == own_center:
                continue
            total = sum(1 for r in self.exchange.formal_entries if r.center == center)
            pages = (total + self.page_size - 1) // self.page_size
            for page in range(pages):
                if reads >= self.summary_page_budget:
                    return reads
                reader.read_summary_page(center, page)
                reads += 1
        return reads

    # -- 公开：梯度回传 -----------------------------------------------
    def fetch_gradients(
        self,
        grid: str,
        cutoff: int | None = None,
        *,
        members: Iterable[tuple[str, str]] | None = None,
        center: str | None = None,
        reader: RevisionReader | None = None,
    ) -> tuple[tuple[tuple[str, str], float], ...]:
        """按成员和格点范围逐段取回梯度。

        只读取所请求成员所在中心的资料，并与增量更新的预算分开记账；读取量
        随请求的成员数增长，不随其他中心成员总数增长。
        """

        cutoff = self.exchange.seal if cutoff is None else int(cutoff)
        settlement = self._compose(
            grid, cutoff, EXPOSURE_FORMAL, self.slices(grid, cutoff)
        )
        if not settlement.deliverable:
            raise LedgerError(
                f"{grid}@{cutoff} 不可交付: "
                f"{settlement.status}/{settlement.availability}"
            )
        wanted = None if members is None else {(c, m) for c, m in members}
        if center is not None:
            available = {mid for mid in settlement.member_ids if mid[0] == center}
            wanted = available if wanted is None else {mid for mid in wanted if mid[0] == center}
        if wanted is None:
            wanted = set(settlement.member_ids)
        unknown = wanted - set(settlement.member_ids)
        if unknown:
            raise KeyError(f"{grid}@{cutoff} 没有成员: {sorted(unknown)}")

        if reader is not None:
            for group_center in sorted({c for c, _ in wanted}):
                group_members = sorted(m for c, m in wanted if c == group_center)
                with reader.restricted(group_center, [grid]):
                    reader.read_segment(
                        group_center, grid, group_members, for_gradient=True
                    )

        return tuple(
            (share.member_id, settlement.gradient_for(*share.member_id))
            for share in settlement.members
            if share.member_id in wanted
        )

    def fetch_gradient_segments(
        self,
        grids: Iterable[str],
        cutoff: int | None = None,
        *,
        members: Iterable[tuple[str, str]] | None = None,
        reader: RevisionReader | None = None,
    ) -> dict[str, tuple[tuple[tuple[str, str], float], ...]]:
        """按成员与格点范围逐段取回梯度，每格点只读该格点涉及的成员。

        读取量随 ``(格点, 请求成员)`` 数增长；不读其他中心、其他格点的原始
        成员，也不随其他中心成员总数增长。
        """

        cutoff = self.exchange.seal if cutoff is None else int(cutoff)
        requested = None if members is None else {(c, m) for c, m in members}
        result: dict[str, tuple[tuple[tuple[str, str], float], ...]] = {}
        for grid in grids:
            settlement = self._compose(
                grid, cutoff, EXPOSURE_FORMAL, self.slices(grid, cutoff)
            )
            if not settlement.deliverable:
                raise LedgerError(
                    f"{grid}@{cutoff} 不可交付: "
                    f"{settlement.status}/{settlement.availability}"
                )
            # 只索取该格点确实拥有的成员；其他格点的成员不在这里报错。
            present = set(settlement.member_ids)
            scoped = None if requested is None else (requested & present)
            result[grid] = self.fetch_gradients(
                grid, cutoff, members=scoped, reader=reader
            )
        return result

    def packet(self, settlement: Settlement) -> GradientPacket:
        if not settlement.deliverable or settlement.gradients is None:
            raise LedgerError(
                f"{settlement.grid}@{settlement.cutoff} 不可交付，没有梯度包"
            )
        return GradientPacket(
            grid=settlement.grid,
            cutoff=settlement.cutoff,
            token=settlement.token,
            epoch=settlement.epoch,
            member_ids=settlement.member_ids,
            gradients=settlement.gradients,
        )

    def accept_packet(
        self, packet: GradientPacket, *, cutoff: int | None = None
    ) -> GradientPacket:
        """确认梯度包对应当前结算状态，拒绝旧状态的成员梯度。"""

        cutoff = packet.cutoff if cutoff is None else int(cutoff)
        current_round = self._round.get(packet.grid, 0)
        if cutoff != current_round:
            raise StaleGradientError(
                f"{packet.grid} 的梯度包属于轮次 {cutoff}，当前轮次是 {current_round}"
            )
        current = self._compose(
            packet.grid, cutoff, EXPOSURE_FORMAL, self.slices(packet.grid, cutoff)
        )
        if not current.deliverable:
            raise StaleGradientError(
                f"{packet.grid}@{cutoff} 当前不可交付: "
                f"{current.status}/{current.availability}"
            )
        if current.token != packet.token or current.epoch != packet.epoch:
            raise StaleGradientError(
                f"{packet.grid}@{cutoff} 的梯度包来自旧状态，不能用于当前轮次"
            )
        return packet

    def build_batch(
        self,
        book: SettlementBook,
        *,
        grids: Iterable[str] | None = None,
        area: Mapping[str, float] | None = None,
    ) -> TrainingBatch:
        """从一份结算中挑选可交付格点组成训练批次。"""

        chosen = book.select(grids)
        packets = tuple(
            self.packet(settlement)
            for grid, settlement in sorted(chosen.settlements.items())
            if settlement.deliverable
        )
        aggregate = chosen.aggregate(area or self.exchange.grid_area)
        return TrainingBatch(cutoff=book.cutoff, packets=packets, aggregate=aggregate)

    # -- 公开：封存与回查 ---------------------------------------------
    def seal(self, cutoff: int | None = None) -> SealedRound:
        """封存一个正式轮次，记录承诺成员集合与结算快照。"""

        cutoff = self.exchange.seal if cutoff is None else int(cutoff)
        self._check_cutoff(cutoff, EXPOSURE_FORMAL)
        if cutoff < self.exchange.seal:
            raise LedgerError(
                f"未到 seal 轮次 {self.exchange.seal}（请求 {cutoff}），不能封存为正式结果"
            )
        settlements = {
            grid: self.settle(grid, cutoff, exposure=EXPOSURE_FORMAL)
            for grid in self.exchange.grids()
        }
        committed = {
            grid: tuple((m.center, m.member, m.value) for m in settlements[grid].members)
            for grid in self.exchange.grids()
        }
        token = _token("sealed", cutoff, settlements, committed)
        sealed = SealedRound(cutoff, settlements, committed, token)
        self._sealed = sealed
        return sealed

    def recall_sealed(self, grid: str, *, verify: bool = True) -> Settlement:
        """按封存时的承诺成员集合回查一个格点。

        ``verify`` 时用承诺集合重建结算并与快照比较；不一致说明封存资料被
        破坏，直接报错而不是悄悄采用新状态。
        """

        if self._sealed is None:
            raise LedgerError("尚未封存正式轮次")
        sealed = self._sealed
        if grid not in sealed.settlements:
            raise KeyError(f"封存轮次中没有格点 {grid!r}")
        settlement = sealed.settlements[grid]
        if not verify:
            return settlement

        by_center: dict[str, list[tuple[str, float]]] = {}
        for center, member, value in sealed.committed[grid]:
            by_center.setdefault(center, []).append((member, value))
        slices = tuple(
            CenterSlice(
                center=center,
                grid=grid,
                cutoff=sealed.cutoff,
                members=tuple(sorted(members)),
                revisions=tuple(),
                epoch=settlement.epoch,
            )
            for center, members in sorted(by_center.items())
        )
        rebuilt = self._compose(
            grid,
            sealed.cutoff,
            EXPOSURE_FORMAL,
            slices,
            epoch=settlement.epoch,
        )
        if rebuilt.token != settlement.token:
            raise LedgerError(
                f"{grid}@{sealed.cutoff} 的承诺成员集合已不能重建原结算；"
                "封存资料与快照不一致"
            )
        return Settlement(**{**settlement.__dict__, "availability": SEALED})

    def settled_members(
        self, grid: str, cutoff: int | None = None
    ) -> tuple[tuple[str, str], ...]:
        """结算实际使用的成员编号，供核对或回传。"""

        cutoff = self.exchange.seal if cutoff is None else int(cutoff)
        return self._compose(
            grid, cutoff, EXPOSURE_FORMAL, self.slices(grid, cutoff)
        ).member_ids

    # -- 只读查询 -----------------------------------------------------
    def history(self, grid: str) -> tuple[Settlement, ...]:
        return tuple(self._history.get(grid, ()))

    @property
    def sealed_round(self) -> SealedRound | None:
        return self._sealed


def _token(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(_canonical(part)).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _canonical(value: object) -> object:
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, MemberShare):
        return (
            value.center,
            value.member,
            format(value.value, ".17g"),
            format(value.weight, ".17g"),
            value.rank,
            value.group,
        )
    if isinstance(value, tuple):
        return tuple(_canonical(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonical(v) for v in value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _canonical(v)) for k, v in value.items()))
    return value


def read_delivery_rows(
    exchange: Exchange, arrival: int
) -> list[tuple[str, str, str, int, bool]]:
    """一个到达的摘要行，便于调试或按摘要页核对。"""

    return [summarize_revision(r) for r in exchange.deliveries(arrival)]
