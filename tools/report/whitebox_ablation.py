import csv
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, str(HERE))

from ablation_tables import auc, boot_ci, median_split_acc
from aimo_bas.profiles import BASE
from aimo_bas.runner import decide, problem_features

DEFAULT_INPUT = ROOT / "runs" / "kaggle_ablate" / "deepseek-math-3072-full-n52.jsonl"
DEFAULT_OUTPUT = ROOT / "report" / "tables" / "whitebox_ablation.csv"
BOOT = 4000
SEED = 20261101

VARIANTS = [
    ("full", "BAS as submitted: forced answers, agree/finish/confidence 0.5/0.3/0.2", True, (0.5, 0.3, 0.2)),
    ("no-confidence", "forced answers kept, confidence dropped: agree/finish 0.625/0.375", True, (0.625, 0.375, 0.0)),
    ("no-probe", "no forced-answer pass: unclosed traces have no answer, no confidence: 0.625/0.375", False, (0.625, 0.375, 0.0)),
    ("confidence-only", "confidence of the forced answer alone", True, (0.0, 0.0, 1.0)),
]


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("label") is not None and row.get("samples"):
                rows.append(row)
    return rows


def records(row, probe):
    out = []
    for sample in row["samples"]:
        closed = bool(sample["closed"])
        answer = sample.get("answer") or ""
        confidence = float(sample.get("confidence") or 0.0)
        if not probe:
            confidence = 0.0
            if not closed:
                answer = ""
        out.append({"answer": answer, "closed": closed, "confidence": confidence})
    return out


def features(row, probe, weights):
    kinds = [sample["kind"] for sample in row["samples"]]
    return problem_features(records(row, probe), kinds, list(weights))


def batches(rows):
    out = []
    start = 0
    while start < len(rows):
        size = int(rows[start].get("batch_size") or len(rows))
        out.append(rows[start : start + size])
        start += size
    return out


def rule_predictions(rows, probe, weights):
    settings = {"hi": BASE["hi"], "lo": BASE["lo"]}
    predictions = []
    for batch in batches(rows):
        feats = [features(row, probe, weights) for row in batch]
        problems = ["x" * int(row["problem_chars"]) for row in batch]
        predictions.extend(decide(feats, problems, settings))
    return predictions


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_auc_ci(scores, reference, labels, n=BOOT, seed=SEED):
    rng = random.Random(seed)
    idx = list(range(len(labels)))
    diffs = []
    for _ in range(n):
        pick = [rng.choice(idx) for _ in idx]
        y = [labels[i] for i in pick]
        a = auc([scores[i] for i in pick], y)
        r = auc([reference[i] for i in pick], y)
        if not (math.isnan(a) or math.isnan(r)):
            diffs.append(a - r)
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]


def evaluate(rows):
    labels = [bool(row["label"]) for row in rows]
    results = []
    reference_scores = None
    reference_predictions = None
    for name, description, probe, weights in VARIANTS:
        scores = [features(row, probe, weights)["score"] for row in rows]
        predictions = rule_predictions(rows, probe, weights)
        if reference_scores is None:
            reference_scores, reference_predictions = scores, predictions
        lo, hi = boot_ci(scores, labels)
        rule_right = [p == y for p, y in zip(predictions, labels)]
        ref_right = [p == y for p, y in zip(reference_predictions, labels)]
        b = sum(1 for r, a in zip(ref_right, rule_right) if r and not a)
        c = sum(1 for r, a in zip(ref_right, rule_right) if a and not r)
        dlo, dhi = paired_auc_ci(scores, reference_scores, labels)
        results.append(
            {
                "variant": name,
                "description": description,
                "n": len(rows),
                "robust": sum(labels),
                "auc": round(auc(scores, labels), 3),
                "auc_ci_low": round(lo, 3),
                "auc_ci_high": round(hi, 3),
                "delta_auc_vs_full": round(auc(scores, labels) - auc(reference_scores, labels), 3),
                "delta_auc_ci_low": round(dlo, 3),
                "delta_auc_ci_high": round(dhi, 3),
                "rule_accuracy": round(sum(rule_right) / len(rows), 3),
                "rule_positives": sum(predictions),
                "topk_accuracy": round(median_split_acc(scores, labels), 3),
                "full_right_variant_wrong": b,
                "variant_right_full_wrong": c,
                "mcnemar_p_vs_full": round(mcnemar_exact(b, c), 4),
                "changed_predictions_vs_full": sum(1 for p, q in zip(predictions, reference_predictions) if p != q),
            }
        )
    return results


def main(argv):
    source = Path(argv[0]) if argv else DEFAULT_INPUT
    target = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUTPUT
    rows = load(source)
    logged = [bool(row["prediction"]) for row in rows]
    replayed = rule_predictions(rows, True, (0.5, 0.3, 0.2))
    if replayed != logged:
        raise SystemExit("replayed full-BAS predictions differ from the logged ones")
    always_true = sum(bool(row["label"]) for row in rows) / len(rows)
    results = evaluate(rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    print(f"source {source.name}: {len(rows)} cases, always-true accuracy {always_true:.3f}; replayed predictions match the log")
    print("| variant | AUC | 95% CI | delta AUC vs full (95% CI) | rule accuracy | changed | McNemar p |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['variant']} | {r['auc']:.3f} | {r['auc_ci_low']:.2f} to {r['auc_ci_high']:.2f} | "
            f"{r['delta_auc_vs_full']:+.3f} ({r['delta_auc_ci_low']:+.2f} to {r['delta_auc_ci_high']:+.2f}) | "
            f"{r['rule_accuracy']:.3f} | {r['changed_predictions_vs_full']} | {r['mcnemar_p_vs_full']} |"
        )
    print("wrote", target)


if __name__ == "__main__":
    main(sys.argv[1:])
