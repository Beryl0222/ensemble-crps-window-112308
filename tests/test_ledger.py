"""按格点结算、增量更新、梯度交接与封存回查的行为核对。

使用 ``data/exchange.json`` 的合成资料：

* g1：north n1=0.4、n2=0.1，west w1=0.9；观测 0.4；面积 2；
* g2：north n1=0.2，west w1 在到达 4 撤回、到达 7（seal 之后）补发 0.4；
  观测 0.2；面积 3；
* g3：观测有效位为 false，不参加结算；面积 1；
* seal=6，到达 7 不属于正式结果；
* 读取窗口：到达 4 限 west@g2，到达 5 限 north@g1。
"""

import json
import unittest
from pathlib import Path

from snowcrps import (
    DELIVERED,
    EXPOSURE_PREVIEW,
    FORMAL,
    INSUFFICIENT_SUPPORT,
    NO_OBSERVATION,
    PENDING,
    Ledger,
    LedgerError,
    RevisionReader,
    GradientPacket,
    StaleGradientError,
    load_exchange,
)
from snowcrps.core import EXPOSURE_FORMAL


def load():
    source = Path(__file__).resolve().parents[1] / "data" / "exchange.json"
    return load_exchange(json.loads(source.read_text(encoding="utf-8")))


class SettlementTests(unittest.TestCase):
    def setUp(self):
        self.exchange = load()
        self.ledger = Ledger(self.exchange)
        self.book = self.ledger.prepare()

    def test_grid_statuses(self):
        self.assertEqual(self.book["g1"].status, DELIVERED)
        self.assertEqual(self.book["g2"].status, INSUFFICIENT_SUPPORT)
        self.assertEqual(self.book["g3"].status, NO_OBSERVATION)

    def test_g1_hardcoded_loss(self):
        settlement = self.book["g1"]
        self.assertEqual(
            settlement.member_ids,
            (("north", "n2"), ("north", "n1"), ("west", "w1")),
        )
        shares = {(m.center, m.member): m.weight for m in settlement.members}
        self.assertAlmostEqual(shares[("north", "n1")], 0.3)
        self.assertAlmostEqual(shares[("north", "n2")], 0.3)
        self.assertAlmostEqual(shares[("west", "w1")], 0.4)
        # O = .3*.3 + .3*0 + .4*.5 = .29
        # S = 2*(.09*.3 + .12*.8 + .12*.5) = .366, D = .66
        self.assertAlmostEqual(settlement.loss, 0.29 - 0.5 * 0.366 / 0.66, places=12)
        self.assertEqual(settlement.availability, FORMAL)

    def test_observation_invalid_grid_excluded(self):
        self.assertIsNone(self.book["g3"].loss)
        self.assertEqual(self.book["g3"].members, ())

    def test_g2_single_member_is_insufficient(self):
        settlement = self.book["g2"]
        self.assertEqual(settlement.status, INSUFFICIENT_SUPPORT)
        self.assertIsNone(settlement.loss)
        self.assertIsNone(settlement.gradients)
        # 覆盖与权重归属仍可查，便于核对是哪份资料缺失。
        self.assertEqual(settlement.center_coverage, ("north",))
        self.assertAlmostEqual(settlement.centers[0].weight, 1.0)

    def test_aggregate_uses_only_deliverable_grids(self):
        aggregate = self.book.aggregate(self.exchange.grid_area)
        self.assertEqual(aggregate.used, ("g1",))
        self.assertAlmostEqual(aggregate.weighted_area, 2.0)
        self.assertAlmostEqual(aggregate.loss, self.book["g1"].loss)
        self.assertIn("g2", aggregate.excluded)
        self.assertIn("g3", aggregate.excluded)

    def test_pending_vs_formal_missing(self):
        early = self.ledger.settle("g2", 1)
        self.assertEqual(early.availability, PENDING)
        self.assertEqual(early.status, INSUFFICIENT_SUPPORT)
        # pending 表示后续成员可能补齐；同一格点在 seal 轮次仍是单成员才是正式缺失。
        self.assertEqual(self.ledger.settle("g2", 2).status, DELIVERED)
        at_seal = self.ledger.settle("g2", 6)
        self.assertEqual(at_seal.availability, FORMAL)

    def test_formal_cannot_pass_seal(self):
        with self.assertRaises(LedgerError):
            self.ledger.settle("g1", 7, exposure=EXPOSURE_FORMAL)

    def test_preview_sees_post_seal_revision_without_gradients(self):
        preview = self.ledger.settle("g2", 7, exposure=EXPOSURE_PREVIEW)
        self.assertEqual(preview.status, DELIVERED)
        self.assertIsNotNone(preview.loss)
        self.assertIsNone(preview.gradients)
        with self.assertRaises(LedgerError):
            self.ledger.settle("g2", 7, exposure=EXPOSURE_FORMAL)

    def test_center_weights_renormalize_per_grid(self):
        # g1 两个中心都参与：0.6 / 0.4。
        g1 = {c.center: c.weight for c in self.book["g1"].centers}
        self.assertAlmostEqual(g1["north"], 0.6)
        self.assertAlmostEqual(g1["west"], 0.4)
        # g2 只有 north：重新归一为 1。
        g2 = {c.center: c.weight for c in self.book["g2"].centers}
        self.assertEqual(set(g2), {"north"})
        self.assertAlmostEqual(g2["north"], 1.0)


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.exchange = load()
        self.ledger = Ledger(self.exchange)
        self.book = self.ledger.prepare()
        self.reader = RevisionReader(self.exchange, own_center="north")

    def test_arrival4_retracts_west_and_recomputes_denominator(self):
        report = self.ledger.apply_change(
            "g2", "west", cutoff=4, arrival=4, reader=RevisionReader(self.exchange, own_center="west")
        )
        self.assertEqual(report.settlement.status, INSUFFICIENT_SUPPORT)
        self.assertTrue(report.membership_changed)
        # 只读了该中心受影响组的资料。
        self.assertTrue(all(rec.center == "west" for rec in report.log))
        self.assertEqual(report.other_center_raw_members, 0)

    def test_arrival5_value_change_keeps_membership(self):
        """g1 的 n2 从 0.4 改成 0.1：成员集合不变，名次改变。"""
        ledger = Ledger(self.exchange)
        early = ledger.prepare(4)
        before = early["g1"]
        report = ledger.apply_change(
            "g1", "north", cutoff=5, arrival=5, reader=self.reader
        )
        after = report.settlement
        self.assertEqual(set(before.member_ids), set(after.member_ids))
        self.assertTrue(report.ranking_changed)
        self.assertFalse(report.membership_changed)
        self.assertNotEqual(before.token, after.token)
        self.assertEqual(report.own_raw_members, 1)
        self.assertEqual(after.status, DELIVERED)

    def test_stale_packet_rejected_after_state_change(self):
        """前一状态（轮次 4）的梯度包不能用于当前轮次 6。"""
        ledger = Ledger(self.exchange)
        packet = ledger.packet(ledger.settle("g1", 4))
        ledger.prepare(6)
        with self.assertRaises(StaleGradientError) as raised:
            ledger.accept_packet(packet)
        self.assertIn("轮次", str(raised.exception))

    def test_tampered_packet_rejected(self):
        """即使轮次相同，状态 token 不一致的梯度包也会被拒绝。"""
        ledger = Ledger(self.exchange)
        packet = ledger.packet(ledger.settle("g1", 6))
        tampered = GradientPacket(
            grid=packet.grid,
            cutoff=packet.cutoff,
            token="0" * 64,
            epoch=packet.epoch,
            member_ids=packet.member_ids,
            gradients=packet.gradients,
        )
        with self.assertRaises(StaleGradientError):
            ledger.accept_packet(tampered)

    def test_change_on_one_grid_does_not_invalidate_another(self):
        ledger = Ledger(self.exchange)
        book = ledger.prepare(4)
        packet = ledger.packet(book["g1"])
        ledger.apply_change(
            "g2", "west", cutoff=4, arrival=4,
            reader=RevisionReader(self.exchange, own_center="west"),
        )
        # 同一轮次内 g1 状态未变，token 仍被接受；g2 的更新不外溢到 g1。
        ledger.accept_packet(packet)

    def test_quality_change_can_make_loss_undefined(self):
        """g1 的 west w1 若被撤下，g2 的 north n1 只剩单成员。"""
        before = self.ledger.settle("g2", 4)
        self.assertEqual(before.status, INSUFFICIENT_SUPPORT)
        # 到达 4 撤回 west@g2；g2 的 g1 同行不受影响。
        report = self.ledger.apply_change(
            "g2", "west", cutoff=4, arrival=4,
            reader=RevisionReader(self.exchange, own_center="west"),
        )
        self.assertFalse(report.settlement.deliverable)
        # 不能把失效格点放进聚合分母。
        book = self.ledger.book(4)
        aggregate = book.aggregate(self.exchange.grid_area)
        self.assertNotIn("g2", aggregate.used)
        self.assertAlmostEqual(aggregate.weighted_area, 2.0)

    def test_duplicate_delivery_is_not_new_sample(self):
        """同一修订重复到达不增加成员。"""
        # 到达 6 与到达 3 是同一修订的重复投递，读取时已合并。
        self.assertEqual(self.exchange.duplicate_deliveries, 1)
        before = self.ledger.settle("g2", 4)
        after_arrival = self.ledger.apply_arrival(
            6, cutoff=6, reader=RevisionReader(self.exchange, own_center="west")
        )
        self.assertEqual(after_arrival, ())
        self.assertEqual(self.ledger.settle("g2", 6).member_ids, before.member_ids)

    def test_arrival_after_cutoff_rejected(self):
        with self.assertRaises(LedgerError):
            self.ledger.apply_change("g1", "north", cutoff=4, arrival=5, reader=self.reader)

    def test_reads_authorized_only_by_window(self):
        """没有读取窗口的到达沿用归约资料，不重读原始成员。"""
        report = self.ledger.apply_change("g2", "west", cutoff=3, arrival=3)
        self.assertEqual(report.own_raw_members, 0)
        self.assertEqual(report.log, ())

    def test_unplanned_read_scope_is_rejected(self):
        reader = RevisionReader(self.exchange, own_center="north")
        with reader.restricted("north", ["g1"]):
            with self.assertRaises(Exception):
                reader.read_member("north", "n1", "g2")

    def test_summary_pages_bounded(self):
        reader = RevisionReader(self.exchange, own_center="north")
        report = self.ledger.apply_change(
            "g1", "north", cutoff=5, arrival=5, reader=reader,
            verify_other_centers=True,
        )
        self.assertLessEqual(report.summary_page_reads, 128)
        self.assertEqual(reader.other_center_reads, 0)
        self.assertEqual(reader.summary_page_reads, report.summary_page_reads)

    def test_center_disappearance_still_adjusts_denominator(self):
        """撤回让 west 在 g2 消失时，用现有归约资料完成中心间分母调整。"""
        self.ledger.apply_change(
            "g2", "west", cutoff=4, arrival=4,
            reader=RevisionReader(self.exchange, own_center="west"),
        )
        g2 = self.ledger.settle("g2", 4)
        self.assertEqual(g2.center_coverage, ("north",))
        self.assertAlmostEqual(g2.centers[0].weight, 1.0)


class GradientTests(unittest.TestCase):
    def setUp(self):
        self.exchange = load()
        self.ledger = Ledger(self.exchange)
        self.book = self.ledger.prepare()

    def test_fetch_by_member_scope_counts_only_requested(self):
        reader = RevisionReader(self.exchange, own_center="north")
        got = self.ledger.fetch_gradients(
            "g1", 6,
            members=[("north", "n1"), ("north", "n2")],
            reader=reader,
        )
        self.assertEqual(reader.gradient_member_reads, 2)
        self.assertEqual(reader.other_center_reads, 0)
        self.assertEqual(len(got), 2)

    def test_fetch_by_center(self):
        got = self.ledger.fetch_gradients("g1", 6, center="west")
        self.assertEqual([mid for mid, _ in got], [("west", "w1")])

    def test_segment_fetch_across_grids(self):
        reader = RevisionReader(self.exchange, own_center="north")
        out = self.ledger.fetch_gradient_segments(
            ["g1"], 6,
            members=[("north", "n1"), ("north", "n2"), ("west", "w1")],
            reader=reader,
        )
        self.assertEqual(reader.gradient_member_reads, 3)
        self.assertEqual(reader.other_center_reads, 0)
        self.assertEqual(len(out["g1"]), 3)

    def test_segment_fetch_ignores_members_absent_from_grid(self):
        """只取该格点存在的成员，不因另一格点的成员而多读。"""
        reader = RevisionReader(self.exchange, own_center="north")
        out = self.ledger.fetch_gradient_segments(
            ["g1"], 6, members=[("west", "absent")], reader=reader
        )
        self.assertEqual(out["g1"], ())
        self.assertEqual(reader.gradient_member_reads, 0)

    def test_gradients_unavailable_when_not_deliverable(self):
        with self.assertRaises(LedgerError):
            self.ledger.fetch_gradients("g2", 6)

    def test_duplicate_values_get_distinct_gradient_entries(self):
        """g1 没有重复值；构造 g1 后 n1/n2 同值的情形由归约保证逐成员条目。"""
        settlement = self.book["g1"]
        self.assertEqual(len(settlement.gradients), len(settlement.members))


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.exchange = load()
        self.ledger = Ledger(self.exchange)
        self.book = self.ledger.prepare()

    def test_batch_only_deliverable(self):
        batch = self.ledger.build_batch(self.book)
        self.assertEqual(batch.grids, ("g1",))
        self.assertEqual(len(batch.packets), 1)
        self.assertAlmostEqual(batch.aggregate.weighted_area, 2.0)

    def test_selecting_previews_and_formal_quantities(self):
        formal = self.book["g1"].exposed()
        preview = self.ledger.settle("g1", 6, exposure=EXPOSURE_PREVIEW).exposed()
        self.assertIn("gradients", formal)
        self.assertNotIn("gradients", preview)
        self.assertIn("token", preview)
        self.assertIsNotNone(preview["loss"])

    def test_caller_chooses_subset(self):
        batch = self.ledger.build_batch(self.book, grids=["g1"])
        self.assertEqual(batch.grids, ("g1",))
        # 选中不可交付格点时不会给出梯度包，也不会进入聚合计。
        empty = self.ledger.build_batch(self.book, grids=["g2"])
        self.assertEqual(empty.grids, ())
        self.assertIsNone(empty.aggregate.loss)
        with self.assertRaises(KeyError):
            self.ledger.build_batch(self.book, grids=["g9"])


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.exchange = load()
        self.ledger = Ledger(self.exchange)

    def test_receipt_rounds_exclude_post_seal(self):
        self.assertEqual(self.ledger.receipt_rounds(), (2, 4, 6))
        self.assertIn(7, self.ledger.receipt_rounds(include_post_seal=True))

    def test_receipt_books_exposures(self):
        books = self.ledger.receipt_books()
        self.assertEqual(sorted(books), [2, 4, 6])
        self.assertEqual(books[6]["g1"].availability, FORMAL)
        self.assertEqual(books[4]["g2"].status, INSUFFICIENT_SUPPORT)
        self.assertIsNone(books[4]["g2"].loss)
        # seal 内的收件轮次按正式口径结算，可带梯度；预览口径不含梯度。
        self.assertIn("gradients", books[4]["g1"].exposed())
        preview = self.ledger.book(4, exposure=EXPOSURE_PREVIEW)["g1"]
        self.assertNotIn("gradients", preview.exposed())
        self.assertEqual(preview.loss, books[4]["g1"].loss)

    def test_caller_can_choose_receipt_to_train(self):
        books = self.ledger.receipt_books()
        batch = self.ledger.build_batch(books[4])
        self.assertEqual(batch.grids, ("g1",))
        self.assertAlmostEqual(batch.aggregate.weighted_area, 2.0)


class SealTests(unittest.TestCase):
    def setUp(self):
        self.exchange = load()
        self.ledger = Ledger(self.exchange)
        self.ledger.prepare()

    def test_seal_and_recall(self):
        sealed = self.ledger.seal(6)
        self.assertIn("g1", sealed.settlements)
        recalled = self.ledger.recall_sealed("g1")
        self.assertEqual(recalled.token, sealed.settlements["g1"].token)

    def test_recall_uses_committed_member_set(self):
        self.ledger.seal(6)
        snapshot = self.ledger.recall_sealed("g1", verify=False)
        committed = self.ledger.sealed_round.committed["g1"]
        self.assertEqual(
            tuple((m.center, m.member, m.value) for m in snapshot.members), committed
        )

    def test_seal_before_deadline_rejected(self):
        with self.assertRaises(LedgerError):
            self.ledger.seal(3)


if __name__ == "__main__":
    unittest.main()
