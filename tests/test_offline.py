import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution"))

from aimo_bas import answers, perturb, runner, profiles


class PerturbTests(unittest.TestCase):
    problem = "Alice picks $n \\le 1000$ with \\(\\frac{n}{7}\\) an integer. Find the number of such integers $n$ and compute \\mathrm{lcm} quickly."

    def test_math_spans_untouched(self):
        for kind in ("typos", "distract", "rephrase", "rename"):
            text = perturb.variant(self.problem, kind)
            for span in ("$n \\le 1000$", "\\(\\frac{n}{7}\\)", "$n$", "\\mathrm{lcm}"):
                self.assertIn(span, text)

    def test_deterministic(self):
        for kind in ("typos", "distract", "rephrase", "rename"):
            self.assertEqual(perturb.variant(self.problem, kind), perturb.variant(self.problem, kind))

    def test_variants_differ(self):
        for kind in ("typos", "distract", "rephrase", "rename"):
            self.assertNotEqual(perturb.variant(self.problem, kind), self.problem)

    def test_numbers_preserved(self):
        import re

        digits = re.findall(r"\d+", self.problem)
        for kind in ("typos", "distract", "rephrase", "rename"):
            self.assertEqual(re.findall(r"\d+", perturb.variant(self.problem, kind)), digits)


class AnswerTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(answers.explicit_answer("work\nANSWER: $\\boxed{1,024}$"), "1024")
        self.assertEqual(answers.explicit_answer("ANSWER: NaN"), "NaN")
        self.assertEqual(answers.explicit_answer("thus \\boxed{\\frac{1}{2}}"), "\\frac{1}{2}")
        self.assertEqual(answers.normalize(" 0339\nmore"), "339")
        self.assertEqual(answers.explicit_answer("no final line"), "")


class DecisionTests(unittest.TestCase):
    settings = profiles.active(False)

    def test_features(self):
        records = [
            {"answer": "12", "closed": True, "confidence": 0.9},
            {"answer": "12", "closed": True, "confidence": 0.8},
            {"answer": "13", "closed": False, "confidence": 0.4},
            {"answer": "", "closed": False, "confidence": 0.1},
        ]
        features = runner.problem_features(records, ["orig", "orig", "typos", "distract"], [0.5, 0.3, 0.2])
        self.assertAlmostEqual(features["agree"], 0.5)
        self.assertAlmostEqual(features["finish"], 0.5)
        self.assertAlmostEqual(features["stable"], 0.0)

    def test_decide_is_bool_and_balanced(self):
        feats = [{"score": s} for s in (0.95, 0.6, 0.5, 0.45, 0.1)]
        problems = ["a" * (i + 1) for i in range(5)]
        predictions = runner.decide(feats, problems, self.settings)
        self.assertEqual(predictions, [True, True, True, False, False])
        self.assertTrue(all(type(p) is bool for p in predictions))

    def test_length_prior(self):
        self.assertEqual(runner.length_prior(["aaa", "a", "aa"]), [False, True, True])

    def test_allot(self):
        settings = dict(self.settings, total_budget_s=3600.0, safety_s=600.0)
        runner.SEEN.clear()
        share = runner.allot_seconds(settings, "openai/gpt-oss-120b", "main", runner.STARTED)
        self.assertGreater(share, 500)
        self.assertLess(share, 3000)


if __name__ == "__main__":
    unittest.main()
