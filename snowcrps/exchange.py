"""交换资料的结构化读取与增量核对。

``entries`` 每行是 ``[arrival, center, member, revision, grid, value, quality]``。
读取时只做结构校验、去重和分组，不解释中心权重、截止或读取预算；这些语义
在 :mod:`snowcrps.core` 的账本里按格点和轮次落实。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

#: 交换资料中必须为文本的列（稳定成员、中心、格点）。
TEXT_FIELDS = ("center", "member", "grid")


class ExchangeValidationError(ValueError):
    """交换资料的结构或取值不满足约定。"""


class BudgetError(RuntimeError):
    """读取越过了本次请求允许的中心或格点范围，或超出预算。"""


@dataclass(frozen=True)
class Revision:
    """一条到达记录。``revision`` 在 ``(center, member, grid)`` 内递增。"""

    arrival: int
    center: str
    member: str
    revision: int
    grid: str
    value: float
    quality: bool

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.center, self.member, self.grid)

    @property
    def member_id(self) -> tuple[str, str]:
        return (self.center, self.member)


@dataclass(frozen=True)
class ReadRecord:
    """一次读取：``kind`` 为 ``window`` / ``member`` / ``center_grid`` / ``summary`` / ``segment``。"""

    kind: str
    center: str
    grid: str
    member: str
    count: int


@dataclass(frozen=True)
class ReadWindow:
    """一次预览后允许读取的原始输入组：一个中心的若干受影响格点。"""

    arrival: int
    center: str
    grids: tuple[str, ...]

    def contains(self, center: str, grid: str) -> bool:
        return center == self.center and grid in self.grids


def _as_str(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExchangeValidationError(f"{what} 必须是非空字符串: {value!r}")
    return value


def _as_int(value: Any, what: str) -> int:
    if isinstance(value, bool):
        raise ExchangeValidationError(f"{what} 必须是整数: {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ExchangeValidationError(f"{what} 必须是整数: {value!r}")


def _as_float(value: Any, what: str) -> float:
    if isinstance(value, bool):
        raise ExchangeValidationError(f"{what} 必须是有限数值: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExchangeValidationError(f"{what} 必须是有限数值: {value!r}") from exc
    if not np.isfinite(number):
        raise ExchangeValidationError(f"{what} 必须是有限数值: {value!r}")
    return number


def _as_bool(value: Any, what: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ExchangeValidationError(f"{what} 必须是布尔量: {value!r}")
    return bool(value)


def _parse_revision(row: Sequence[Any], index: int) -> Revision:
    if not isinstance(row, (list, tuple)):
        raise ExchangeValidationError(f"entries[{index}] 必须是序列: {row!r}")
    if len(row) != 7:
        raise ExchangeValidationError(
            f"entries[{index}] 需要 7 列 "
            f"(arrival, center, member, revision, grid, value, quality)，实际 {len(row)}: {row!r}"
        )
    arrival, center, member, revision, grid, value, quality = row
    return Revision(
        arrival=_as_int(arrival, f"entries[{index}].arrival"),
        center=_as_str(center, f"entries[{index}].center"),
        member=_as_str(member, f"entries[{index}].member"),
        revision=_as_int(revision, f"entries[{index}].revision"),
        grid=_as_str(grid, f"entries[{index}].grid"),
        value=_as_float(value, f"entries[{index}].value"),
        quality=_as_bool(quality, f"entries[{index}].quality"),
    )


class Exchange:
    """一份已校验的交换资料。

    ``entries`` 保持到达顺序，并已去掉完全相同的重复投递。``seal`` 之后的
    到达仍保留在资料里，但正式结算不会把它们计入。
    """

    def __init__(
        self,
        *,
        center_weight: Mapping[str, Any],
        grid_area: Mapping[str, Any],
        observations: Mapping[str, Any],
        entries: Iterable[Sequence[Any]],
        seal: Any,
        query: Iterable[Any],
        max_update_work: Mapping[str, Any] | None = None,
        read_windows: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        self.center_weight = {
            _as_str(k, "center_weight 中心"): _as_float(v, f"center_weight[{k}]")
            for k, v in center_weight.items()
        }
        if not self.center_weight:
            raise ExchangeValidationError("center_weight 不能为空")
        for center, weight in self.center_weight.items():
            if weight <= 0.0:
                raise ExchangeValidationError(f"center_weight[{center}] 必须为正: {weight!r}")

        self.grid_area = {
            _as_str(k, "grid_area 格点"): _as_float(v, f"grid_area[{k}]")
            for k, v in grid_area.items()
        }
        if not self.grid_area:
            raise ExchangeValidationError("grid_area 不能为空")
        for grid, area in self.grid_area.items():
            if area < 0.0:
                raise ExchangeValidationError(f"grid_area[{grid}] 不能为负: {area!r}")

        observations = dict(observations)
        self.observation_value: dict[str, float] = {}
        self.observation_valid: dict[str, bool] = {}
        for grid, pair in observations.items():
            grid = _as_str(grid, "observations 格点")
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ExchangeValidationError(
                    f"observations[{grid}] 需要 [value, valid] 两列: {pair!r}"
                )
            self.observation_value[grid] = _as_float(pair[0], f"observations[{grid}].value")
            self.observation_valid[grid] = _as_bool(pair[1], f"observations[{grid}].valid")
        missing = set(self.grid_area) - set(self.observation_value)
        if missing:
            raise ExchangeValidationError(f"格点缺少观测记录: {sorted(missing)}")

        self.seal = _as_int(seal, "seal")
        self.query = tuple(_as_int(q, "query[]") for q in query)
        self.max_update_work = {
            _as_str(k, "max_update_work 项"): _as_int(v, f"max_update_work[{k}]")
            for k, v in (max_update_work or {}).items()
        }

        # 重复投递不是新样本：同一 (中心, 成员, 格点) 的同一修订再次到达时只保留
        # 最早一次；修订号相同但内容不同则是资料错误。
        parsed = [_parse_revision(row, index) for index, row in enumerate(entries)]
        seen: dict[tuple[tuple[str, str, str], int], Revision] = {}
        deduped: list[Revision] = []
        self.duplicate_deliveries = 0
        for revision in parsed:
            slot = (revision.key, revision.revision)
            previous = seen.get(slot)
            if previous is not None:
                if (previous.value, previous.quality) == (revision.value, revision.quality):
                    self.duplicate_deliveries += 1
                    continue
                raise ExchangeValidationError(
                    f"{revision.key} 的修订 {revision.revision} 出现不同内容: "
                    f"{previous!r} 与 {revision!r}"
                )
            seen[slot] = revision
            deduped.append(revision)
        self.entries: list[Revision] = deduped

        unknown = {r.grid for r in self.entries} - set(self.grid_area)
        if unknown:
            raise ExchangeValidationError(f"entries 出现未登记格点: {sorted(unknown)}")
        unknown_centers = {r.center for r in self.entries} - set(self.center_weight)
        if unknown_centers:
            raise ExchangeValidationError(f"entries 出现未登记中心: {sorted(unknown_centers)}")

        # 每个 (格点, 中心, 成员) 的修订序列，按修订递增（到达也随之递增）。
        self._series: dict[str, dict[tuple[str, str], tuple[list[int], list[Revision]]]] = {}
        for revision in self.entries:
            grid_series = self._series.setdefault(revision.grid, {})
            arrivals, revisions = grid_series.setdefault(
                revision.member_id, ([], [])
            )
            arrivals.append(revision.arrival)
            revisions.append(revision)
        for grid_series in self._series.values():
            for arrivals, revisions in grid_series.values():
                order = sorted(range(len(revisions)), key=lambda i: revisions[i].revision)
                arrivals[:] = [arrivals[i] for i in order]
                revisions[:] = [revisions[i] for i in order]
                if any(
                    arrivals[i] > arrivals[i + 1] for i in range(len(arrivals) - 1)
                ):
                    raise ExchangeValidationError(
                        "同一成员的修订号必须随到达序号递增"
                    )

        self.read_windows: tuple[ReadWindow, ...] = tuple(
            self._parse_window(window, index) for index, window in enumerate(read_windows)
        )
        by_arrival: dict[int, list[ReadWindow]] = {}
        for window in self.read_windows:
            by_arrival.setdefault(window.arrival, []).append(window)
        self._windows_by_arrival = by_arrival

        for revision in self.query:
            if revision < 1:
                raise ExchangeValidationError(f"query 轮次必须为正: {revision!r}")

    # -- 结构查询 -----------------------------------------------------
    @staticmethod
    def _parse_window(window: Mapping[str, Any], index: int) -> ReadWindow:
        if not isinstance(window, Mapping):
            raise ExchangeValidationError(f"read_windows[{index}] 必须是对象: {window!r}")
        if "arrival" not in window or "center" not in window or "grid" not in window:
            raise ExchangeValidationError(
                f"read_windows[{index}] 需要 arrival、center、grid: {window!r}"
            )
        grids = window["grid"]
        if isinstance(grids, str):
            grids = [grids]
        if not grids:
            raise ExchangeValidationError(f"read_windows[{index}].grid 不能为空")
        return ReadWindow(
            arrival=_as_int(window["arrival"], f"read_windows[{index}].arrival"),
            center=_as_str(window["center"], f"read_windows[{index}].center"),
            grids=tuple(_as_str(g, f"read_windows[{index}].grid") for g in grids),
        )

    def centers(self) -> tuple[str, ...]:
        return tuple(sorted(self.center_weight))

    def grids(self) -> tuple[str, ...]:
        return tuple(sorted(self.grid_area))

    def revisions_for_grid(self, grid: str) -> list[Revision]:
        return [r for r in self.entries if r.grid == grid]

    def revisions_for_center_grid(self, center: str, grid: str) -> list[Revision]:
        return [r for r in self.entries if r.grid == grid and r.center == center]

    def deliveries(self, arrival: int) -> list[Revision]:
        return [r for r in self.entries if r.arrival == arrival]

    def windows_at(self, arrival: int) -> tuple[ReadWindow, ...]:
        return tuple(self._windows_by_arrival.get(arrival, ()))

    @property
    def formal_entries(self) -> list[Revision]:
        """``seal`` 之内到达、可进入正式结果的记录。"""
        return [r for r in self.entries if r.arrival <= self.seal]

    # -- 按轮次选修订 -------------------------------------------------
    def selection(
        self, grid: str, cutoff: int, *, before_arrival: int | None = None
    ) -> list[Revision]:
        """截至 ``cutoff`` 每个 ``(中心, 成员)`` 的最新修订。

        ``before_arrival`` 给出时只取到达序号严格小于它的修订，用于还原某次
        到达之前的状态。返回的条目可能带 ``quality=False``：那表示该成员在
        此轮次已被撤回或剔除，调用方需据此把它排除出有效成员集合。
        """

        grid_series = self._series.get(grid, {})
        chosen: list[Revision] = []
        for member_id in sorted(grid_series):
            arrivals, revisions = grid_series[member_id]
            if before_arrival is None:
                position = bisect_right(arrivals, cutoff) - 1
            else:
                position = bisect_left(arrivals, before_arrival) - 1
            if position >= 0:
                chosen.append(revisions[position])
        return chosen

    def effective_selection(self, grid: str, cutoff: int) -> list[Revision]:
        """截至 ``cutoff`` 通过质量标志的有效成员。"""

        return [r for r in self.selection(grid, cutoff) if r.quality]

    def same_grid(self, other: "Exchange", grid: str) -> bool:
        return self.observation_value[grid] == other.observation_value[grid]


def load_exchange(source: Mapping[str, Any]) -> Exchange:
    """从已解析的 JSON 映射构造 :class:`Exchange`。"""

    required = ("center_weight", "grid_area", "observations", "entries", "seal", "query")
    missing = [name for name in required if name not in source]
    if missing:
        raise ExchangeValidationError(f"交换资料缺少字段: {missing}")
    return Exchange(
        center_weight=source["center_weight"],
        grid_area=source["grid_area"],
        observations=source["observations"],
        entries=source["entries"],
        seal=source["seal"],
        query=source["query"],
        max_update_work=source.get("max_update_work"),
        read_windows=source.get("read_windows", ()),
    )


def summarize_revision(revision: Revision) -> tuple[str, str, str, int, bool]:
    """摘要页中的一个成员条目：``(center, member, grid, revision, quality)``。

    读取其他中心的资料只允许以这种摘要粒度进行；摘要不含数值，避免把摘要
    读取混同为原始成员读取。
    """

    return (
        revision.center,
        revision.member,
        revision.grid,
        revision.revision,
        revision.quality,
    )


@dataclass(frozen=True)
class ReadMark:
    """读取器状态的快照，用来核对一次请求没有碰到窗口外的资料。"""

    log_length: int
    window_member_reads: int
    other_center_reads: int
    summary_page_reads: int
    gradient_member_reads: int


class RevisionReader:
    """按中心、格点、成员范围读取交换资料，并记录各类读取量。

    读取量的四个口径分开记账，因为它们的预算分别计算：

    * ``window_member_reads``：按 ``read_windows`` 读取受影响中心的原始成员片；
    * ``other_center_reads``：正式准备输入时读取其他中心的原始成员；
    * ``summary_page_reads``：其他中心的摘要页读取（每页 ``page_size`` 条）；
    * ``gradient_member_reads``：逐段取回梯度时按成员和格点范围读取的成员数。

    预览请求应放在 :meth:`restricted` 里，任何越出该中心受影响格点的读取都会
    立即抛 :class:`BudgetError`，不会静默扩大读取范围。
    """

    def __init__(self, exchange: Exchange, *, own_center: str, page_size: int = 64) -> None:
        self.exchange = exchange
        if own_center not in exchange.center_weight:
            raise ExchangeValidationError(f"own_center 未登记: {own_center!r}")
        self.own_center = own_center
        self.page_size = max(1, int(page_size))
        self.window_member_reads = 0
        self.other_center_reads = 0
        self.summary_page_reads = 0
        self.gradient_member_reads = 0
        self.log: list[ReadRecord] = []
        self._scope: tuple[str, frozenset[str]] | None = None

    # -- 作用域 -------------------------------------------------------
    @contextmanager
    def restricted(self, center: str, grids: Iterable[str]) -> Iterator["RevisionReader"]:
        """把读取限定在一个中心的一个或多个格点上。"""

        if center not in self.exchange.center_weight:
            raise ExchangeValidationError(f"限制作用域的中心未登记: {center!r}")
        previous = self._scope
        self._scope = (center, frozenset(grids))
        try:
            yield self
        finally:
            self._scope = previous

    def _check_scope(self, center: str, grid: str) -> None:
        if self._scope is None:
            return
        scope_center, scope_grids = self._scope
        if center != scope_center or grid not in scope_grids:
            raise BudgetError(
                f"读取 ({center}, {grid}) 越出本次请求允许的 "
                f"({scope_center}, {sorted(scope_grids)})"
            )

    def mark(self) -> ReadMark:
        return ReadMark(
            log_length=len(self.log),
            window_member_reads=self.window_member_reads,
            other_center_reads=self.other_center_reads,
            summary_page_reads=self.summary_page_reads,
            gradient_member_reads=self.gradient_member_reads,
        )

    # -- 按读取窗口取原始成员片 ---------------------------------------
    def read_window(self, window: ReadWindow) -> list[Revision]:
        """读取一个读取窗口列出的成员；这是预览后可用的原始资料范围。"""

        readings: list[Revision] = []
        for grid in window.grids:
            self._check_scope(window.center, grid)
            segment = self.exchange.revisions_for_center_grid(window.center, grid)
            members = {r.member for r in segment}
            self.window_member_reads += len(members)
            self.log.append(ReadRecord("window", window.center, grid, "", len(members)))
            readings.extend(segment)
        return readings

    def after(self, mark: ReadMark) -> list[ReadRecord]:
        return self.log[mark.log_length :]

    # -- 原始成员读取 -------------------------------------------------
    def read_member(self, center: str, member: str, grid: str) -> list[Revision]:
        self._check_scope(center, grid)
        readings = [
            r
            for r in self.exchange.revisions_for_center_grid(center, grid)
            if r.member == member
        ]
        if center == self.own_center:
            self.window_member_reads += 1
        else:
            self.other_center_reads += 1
        self.log.append(ReadRecord("member", center, grid, member, len(readings)))
        return readings

    def read_center_grid(self, center: str, grid: str) -> list[Revision]:
        self._check_scope(center, grid)
        readings = self.exchange.revisions_for_center_grid(center, grid)
        members = {r.member for r in readings}
        if center == self.own_center:
            self.window_member_reads += len(members)
        else:
            self.other_center_reads += len(members)
        self.log.append(ReadRecord("center_grid", center, grid, "", len(members)))
        return readings

    # -- 摘要页读取 ---------------------------------------------------
    def read_summary_page(self, center: str, page: int = 0) -> list[tuple[str, str, str, int, bool]]:
        if center == self.own_center:
            raise ExchangeValidationError("摘要页读取只用于其他中心")
        if self._scope is not None:
            scope_center, scope_grids = self._scope
            if center != scope_center:
                raise BudgetError(
                    f"摘要页读取 {center!r} 越出本次请求允许的 {scope_center!r}"
                )
        rows = [summarize_revision(r) for r in self.exchange.formal_entries if r.center == center]
        start = page * self.page_size
        chunk = rows[start : start + self.page_size]
        self.summary_page_reads += 1
        self.log.append(ReadRecord("summary", center, "", "", len(chunk)))
        return chunk

    # -- 分段读取 -----------------------------------------------------
    def read_segment(
        self, center: str, grid: str, members: Iterable[str], *, for_gradient: bool = True
    ) -> list[Revision]:
        wanted = set(members)
        self._check_scope(center, grid)
        readings = [
            r
            for r in self.exchange.revisions_for_center_grid(center, grid)
            if r.member in wanted
        ]
        if for_gradient:
            self.gradient_member_reads += len(wanted)
        self.log.append(ReadRecord("segment", center, grid, "", len(wanted)))
        return readings

    # -- 预算核对 -----------------------------------------------------
    def check_budget(
        self,
        *,
        other_member_scan: int | None = None,
        summary_pages: int | None = None,
        gradient_member_reads: int | None = None,
    ) -> None:
        if other_member_scan is not None and self.other_center_reads > other_member_scan:
            raise BudgetError(
                f"其他中心原始成员读取 {self.other_center_reads} 超过预算 {other_member_scan}"
            )
        if summary_pages is not None and self.summary_page_reads > summary_pages:
            raise BudgetError(
                f"摘要页读取 {self.summary_page_reads} 超过预算 {summary_pages}"
            )
        if (
            gradient_member_reads is not None
            and self.gradient_member_reads > gradient_member_reads
        ):
            raise BudgetError(
                f"梯度分段读取 {self.gradient_member_reads} 超过预算 {gradient_member_reads}"
            )
