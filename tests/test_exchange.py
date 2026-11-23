"""核对交换资料的字段和数值关系。"""
import json
from pathlib import Path
import unittest

import numpy as np


class ExchangeTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "data" / "exchange.json"
        self.data = json.loads(source.read_text(encoding="utf-8"))

    def test_exchange_data(self):
        rows = self.data['entries']
        self.assertEqual(rows[5][1:], rows[6][1:])
        self.assertNotEqual(rows[6][-1], rows[7][-1])
        self.assertGreater(rows[-1][0], self.data['seal'])
