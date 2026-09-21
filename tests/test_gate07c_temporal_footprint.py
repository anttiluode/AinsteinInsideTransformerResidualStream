import unittest

from temporal_footprint_candidates import (
    HORIZONS,
    ROUTES,
    temporal_candidates,
    trajectory_steps,
)


class Gate07cCandidateTests(unittest.TestCase):
    def test_three_routes_have_three_steps(self):
        rows = trajectory_steps(17, 24)
        self.assertEqual(tuple(rows), ROUTES)
        for route in ROUTES:
            self.assertEqual(len(rows[route]), len(HORIZONS))

    def test_candidates_are_numeric_only(self):
        candidates = temporal_candidates(17, 24)
        for text in candidates.values():
            self.assertFalse(any(ch.isalpha() for ch in text))

    def test_same_first_edge_decoy_really_matches_first_edge(self):
        candidates = temporal_candidates(17, 24)
        true_first = candidates["B@1"].strip()
        drift_first = candidates["B_decoy_same_first_drift"].strip().split(";", 1)[0]
        self.assertEqual(true_first, drift_first)

    def test_matched_decoys_are_not_the_true_route(self):
        candidates = temporal_candidates(17, 24)
        target = candidates["B@3"]
        self.assertNotEqual(target, candidates["B_decoy_same_first_drift"])
        self.assertNotEqual(target, candidates["B_decoy_unit_recurrence"])
        self.assertNotEqual(target, candidates["B_decoy_other_constant"])

    def test_route_b_prefixes_are_nested_as_text(self):
        candidates = temporal_candidates(13, 26)
        self.assertTrue(candidates["B@2"].startswith(candidates["B@1"]))
        self.assertTrue(candidates["B@3"].startswith(candidates["B@2"]))


if __name__ == "__main__":
    unittest.main()
