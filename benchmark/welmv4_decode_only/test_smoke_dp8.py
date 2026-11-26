import unittest

from smoke_dp8 import BUCKETS, CASES, audit_backend, audit_steps


def step(rank, bucket, rows):
    return (f"[DP{rank} TP{rank} EP{rank}] Decode graph replay: worker=target "
            f"key_size={bucket} (bs) mode={'DECODE' if rows else 'IDLE'} raw_bs={rows}")


def steps(rows, bucket):
    return [step(rank, bucket, count) for _ in range(3) for rank, count in enumerate(rows)]


class TestDP8Smoke(unittest.TestCase):
    def test_balanced_skew_and_idle_are_proven_by_rows(self):
        for case, routes in CASES.items():
            with self.subTest(case=case):
                rows = [routes.count(rank) for rank in range(8)]
                bucket = next(size for size in BUCKETS if size >= max(rows))
                result = audit_steps(steps(rows, bucket), routes, 3)
                self.assertTrue(result["qualified"])
                self.assertEqual(result["observed_peak_rows"], len(routes))
                self.assertEqual(result["expected_bucket"], bucket)

    def test_skew_routes_require_real_padding(self):
        for case, expected_max, expected_bucket in (("skew", 3, 4), ("skew8", 5, 8)):
            with self.subTest(case=case):
                routes = CASES[case]
                rows = [routes.count(rank) for rank in range(8)]
                self.assertEqual(max(rows), expected_max)
                self.assertIn(0, rows)
                self.assertTrue(audit_steps(steps(rows, expected_bucket), routes, 3)["qualified"])
        rows = [CASES["skew"].count(rank) for rank in range(8)]
        self.assertFalse(audit_steps(steps(rows, 8), CASES["skew"], 3)["qualified"])

    def test_offered_64_with_observed_56_does_not_qualify(self):
        result = audit_steps(steps([7] * 8, 8), CASES["64"], 3)
        self.assertFalse(result["qualified"])
        self.assertEqual(result["observed_peak_rows"], 56)

    def test_rejects_missing_rank_and_disagreeing_bucket(self):
        lines = steps([1] + [0] * 7, 1)
        with self.assertRaisesRegex(ValueError, "unbalanced"):
            audit_steps(lines[:-1], CASES["1"], 3)
        lines[-1] = step(7, 2, 0)
        with self.assertRaisesRegex(ValueError, "inconsistent buckets"):
            audit_steps(lines, CASES["1"], 3)

    def test_direct_replays_require_capture_per_rank_and_bucket(self):
        lines = []
        layers = dict.fromkeys(range(48), (3, 1))
        for rank in range(8):
            shape = "ShapeKey(size=4, stream_idx=None, variant_label=None)"
            prefix = f"[DP{rank} TP{rank} EP{rank}] welm-mk-target"
            lines.append(f"{prefix} recorded shape={shape} layer_launches(total,fused)={layers} "
                         "coverage={'torch_boundaries': []}")
            lines.append(f"{prefix} replay shape={shape} count=1 backend=mk-direct")
        audit_backend(lines, "megakernel-native", {4})
        with self.assertRaisesRegex(ValueError, "Unproven"):
            audit_backend(lines[:-1], "megakernel-native", {4})
        with self.assertRaisesRegex(ValueError, "Unexpected graph executor"):
            audit_backend([*lines[:-1], lines[-1].replace("mk-direct", "acl")],
                          "megakernel-native", {4})
        with self.assertRaisesRegex(ValueError, "fallback/wrapper"):
            audit_backend([*lines, "welm-mk-target eager fallback"], "megakernel-native", {4})


if __name__ == "__main__":
    unittest.main()
