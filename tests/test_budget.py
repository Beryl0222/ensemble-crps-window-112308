"""读取预算与大量重复值：预算不随其他中心成员数增长。"""

import time
import unittest

from snowcrps import (
    DELIVERED,
    FORMAL,
    INSUFFICIENT_SUPPORT,
    PENDING,
    Ledger,
    RevisionReader,
    load_exchange,
)


def synthetic(other_members: int, *, own_members: int = 3):
    """构造一个中心成员很多、另一个中心很少的格点。"""

    entries = []
    arrival = 1
    for index in range(other_members):
        entries.append([arrival, "bulk", f"b{index}", 1, "g1", 0.5, True])
    for index in range(own_members):
        entries.append([arrival, "own", f"o{index}", 1, "g1", 0.4 + 0.01 * index, True])
    # 到达 3：own 中心 o0 的值被更正；到达 4：own 中心 o1 退出。
    entries.append([3, "own", "o0", 2, "g1", 0.7, True])
    entries.append([4, "own", "o1", 2, "g1", 0.0, False])
    return load_exchange(
        {
            "center_weight": {"own": 0.5, "bulk": 0.5},
            "grid_area": {"g1": 1},
            "observations": {"g1": [0.4, True]},
            "entries": entries,
            "seal": 5,
            "query": [5],
            "max_update_work": {"other_member_scan": 0, "other_summary_page_reads": 128},
            "read_windows": [
                {"arrival": 3, "center": "own", "grid": ["g1"]},
                {"arrival": 4, "center": "own", "grid": ["g1"]},
            ],
        }
    )


class BudgetScalingTests(unittest.TestCase):
    def _reads(self, other_members: int) -> tuple[int, int, int]:
        exchange = synthetic(other_members)
        ledger = Ledger(exchange)
        ledger.prepare(5)
        reader = RevisionReader(exchange, own_center="own")
        change = ledger.apply_change("g1", "own", cutoff=3, arrival=3, reader=reader)
        exit_ = ledger.apply_change("g1", "own", cutoff=4, arrival=4, reader=reader)
        self.assertEqual(change.own_raw_members, 1)
        self.assertEqual(exit_.own_raw_members, 1)
        return (
            reader.window_member_reads,
            reader.other_center_reads,
            reader.summary_page_reads,
        )

    def test_reads_do_not_grow_with_other_center(self):
        small = self._reads(4)
        large = self._reads(4000)
        self.assertEqual(small, large)
        # 两次更新各只读一个受影响成员；不读其他中心原始成员。
        self.assertEqual(large[0], 2)
        self.assertEqual(large[1], 0)
        self.assertEqual(large[2], 0)

    def test_duplicate_heavy_grid_settles_without_pair_matrix(self):
        count = 20000
        exchange = synthetic(count, own_members=2)
        ledger = Ledger(exchange)
        started = time.perf_counter()
        settlement = ledger.settle("g1", 5)
        elapsed = time.perf_counter() - started
        self.assertEqual(settlement.status, DELIVERED)
        self.assertLess(elapsed, 2.0)
        # o1 在到达 4 退出，只剩 o0。
        self.assertEqual(settlement.effective_count, count + 1)
        # 大量同值成员只有一个值组；导数仍逐成员给出。
        self.assertEqual(len(settlement.gradients), count + 1)

    def test_withdrawal_of_center_last_member_reweights_other_center(self):
        exchange = load_exchange(
            {
                "center_weight": {"own": 0.5, "bulk": 0.5},
                "grid_area": {"g1": 1},
                "observations": {"g1": [0.4, True]},
                "entries": [
                    [1, "bulk", "b0", 1, "g1", 0.5, True],
                    [1, "own", "o0", 1, "g1", 0.4, True],
                    [4, "own", "o0", 2, "g1", 0.0, False],
                ],
                "seal": 5,
                "query": [5],
                "max_update_work": {"other_member_scan": 0, "other_summary_page_reads": 128},
                "read_windows": [{"arrival": 4, "center": "own", "grid": ["g1"]}],
            }
        )
        ledger = Ledger(exchange)
        ledger.prepare(5)
        before = ledger.settle("g1", 3)
        self.assertEqual(set(before.center_coverage), {"own", "bulk"})
        # own 中心唯一成员 o0 在到达 4 退出，own 中心在该格点消失。
        reader = RevisionReader(exchange, own_center="own")
        report = ledger.apply_change("g1", "own", cutoff=4, arrival=4, reader=reader)
        after = report.settlement
        self.assertEqual(after.center_coverage, ("bulk",))
        self.assertAlmostEqual(after.centers[0].weight, 1.0)
        # 只剩一个有效成员，成对项不可定义，但权重归属与覆盖仍已完成调整。
        self.assertEqual(after.status, INSUFFICIENT_SUPPORT)
        # cutoff=4 早于 seal=5，因此属于尚可补齐的 pending。
        self.assertEqual(after.availability, PENDING)


if __name__ == "__main__":
    unittest.main()
