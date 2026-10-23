import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "report"))

import verify_bundles
import whitebox_ablation

ABLATION = ROOT / "runs" / "kaggle_ablate" / "deepseek-math-3072-full-n52.jsonl"


class WhiteBoxAblationTests(unittest.TestCase):
    row = {
        "samples": [
            {"kind": "orig", "answer": "7", "closed": True, "confidence": 0.9},
            {"kind": "orig", "answer": "7", "closed": False, "confidence": 0.8},
            {"kind": "typos", "answer": "5", "closed": False, "confidence": 0.4},
        ]
    }

    def test_dropping_the_probe_blanks_answers_and_confidence_of_unclosed_traces(self):
        records = whitebox_ablation.records(self.row, probe=False)
        self.assertEqual([r["answer"] for r in records], ["7", "", ""])
        self.assertEqual([r["confidence"] for r in records], [0.0, 0.0, 0.0])

    def test_keeping_the_probe_leaves_forced_answers_in_place(self):
        records = whitebox_ablation.records(self.row, probe=True)
        self.assertEqual([r["answer"] for r in records], ["7", "7", "5"])

    def test_full_variant_replays_the_logged_ablation_predictions(self):
        rows = whitebox_ablation.load(ABLATION)
        replayed = whitebox_ablation.rule_predictions(rows, True, (0.5, 0.3, 0.2))
        self.assertEqual(replayed, [bool(r["prediction"]) for r in rows])

    def test_exact_mcnemar_is_one_without_discordant_pairs(self):
        self.assertEqual(whitebox_ablation.mcnemar_exact(0, 0), 1.0)

    def test_exact_mcnemar_needs_six_one_sided_discordant_pairs_for_five_percent(self):
        self.assertGreater(whitebox_ablation.mcnemar_exact(5, 0), 0.05)
        self.assertLess(whitebox_ablation.mcnemar_exact(6, 0), 0.05)


class BundleVerificationTests(unittest.TestCase):
    def test_every_committed_bundle_matches_its_manifest_hash_and_rebuilds_identically(self):
        for name, expected in verify_bundles.manifest().items():
            with self.subTest(bundle=name):
                result = verify_bundles.verify(ROOT / "submissions" / name, expected, ROOT / "solution")
                self.assertTrue(result["sha256_matches_manifest"])
                self.assertTrue(result["rebuild_is_byte_identical"])

    def test_bundled_code_differs_from_the_released_source_only_in_profiles(self):
        for name, expected in verify_bundles.manifest().items():
            with self.subTest(bundle=name):
                result = verify_bundles.verify(ROOT / "submissions" / name, expected, ROOT / "solution")
                for member in result["members_differing_from_solution"]:
                    self.assertTrue(member.startswith("aimo_bas/profiles.py"), member)


if __name__ == "__main__":
    unittest.main()
