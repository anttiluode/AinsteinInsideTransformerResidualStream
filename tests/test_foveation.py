import unittest

from foveation import (
    adaptive_depth_allocation,
    allocation_entropy,
    total_utility,
    uniform_depth_allocation,
)
from gate1_allocator_receipt import payoff


class FoveationTests(unittest.TestCase):
    def test_uniform_spends_exact_budget(self):
        d = uniform_depth_allocation(["a", "b", "c"], 8)
        self.assertEqual(sum(d.values()), 8)
        self.assertLessEqual(max(d.values()) - min(d.values()), 1)

    def test_adaptive_spends_exact_budget(self):
        d, trace = adaptive_depth_allocation(
            ["A", "B", "C", "D"],
            16,
            payoff,
            diversity_reserve=0.20,
        )
        self.assertEqual(sum(d.values()), 16)
        self.assertEqual(len(trace), 16)

    def test_diversity_reserve_recovers_late_bloomer(self):
        greedy, _ = adaptive_depth_allocation(
            ["A", "B", "C", "D"],
            16,
            payoff,
            diversity_reserve=0.0,
        )
        adaptive, _ = adaptive_depth_allocation(
            ["A", "B", "C", "D"],
            16,
            payoff,
            diversity_reserve=0.20,
        )
        self.assertGreater(adaptive["B"], greedy["B"])
        self.assertGreater(
            total_utility(adaptive, payoff),
            total_utility(greedy, payoff),
        )

    def test_adaptive_beats_uniform_on_declared_receipt_world(self):
        uniform = uniform_depth_allocation(["A", "B", "C", "D"], 16)
        adaptive, _ = adaptive_depth_allocation(
            ["A", "B", "C", "D"],
            16,
            payoff,
            diversity_reserve=0.20,
        )
        self.assertGreater(
            total_utility(adaptive, payoff),
            total_utility(uniform, payoff),
        )

    def test_entropy_bounds(self):
        self.assertAlmostEqual(allocation_entropy({"a": 2, "b": 2}), 1.0)
        self.assertGreaterEqual(allocation_entropy({"a": 4, "b": 1}), 0.0)
        self.assertLessEqual(allocation_entropy({"a": 4, "b": 1}), 1.0)


if __name__ == "__main__":
    unittest.main()
