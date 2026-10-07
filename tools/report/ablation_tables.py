import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

FEATURES = ["agree", "finish", "confidence", "stable", "orig_agree", "score"]


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("label") is None or "features" not in r:
                continue
            f = dict(r["features"])
            f["score"] = 0.5 * f.get("agree", 0) + 0.3 * f.get("finish", 0) + 0.2 * f.get("confidence", 0)
            r["f"] = f
            rows.append(r)
    return rows


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def boot_ci(scores, labels, n=4000, seed=20261101):
    rng = random.Random(seed)
    idx = list(range(len(scores)))
    vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        a = auc([scores[i] for i in s], [labels[i] for i in s])
        if not math.isnan(a):
            vals.append(a)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def median_split_acc(scores, labels):
    k = sum(labels)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    pred = [False] * len(scores)
    for i in order[:k]:
        pred[i] = True
    return sum(p == y for p, y in zip(pred, labels)) / len(labels)


def sample_agreement(rows, kinds):
    out = []
    for r in rows:
        ans = [s.get("answer") for s in r.get("samples", []) if s.get("kind") in kinds]
        ans = [a for a in ans if a]
        out.append(Counter(ans).most_common(1)[0][1] / len(ans) if ans else 0.0)
    return out


def main(paths):
    print("| plan | file | n | robust | feature | AUC | 95% CI | top-k accuracy |")
    print("|---|---|---|---|---|---|---|---|")
    for path in paths:
        rows = load(path)
        if not rows:
            print(f"| - | {Path(path).name} | 0 | - | - | - | - | - |")
            continue
        labels = [bool(r["label"]) for r in rows]
        plan = rows[0].get("variants") or "mixed"
        feats = {name: [r["f"].get(name, 0.0) or 0.0 for r in rows] for name in FEATURES}
        feats["orig-only agreement"] = sample_agreement(rows, {"orig"})
        feats["perturbed-only agreement"] = sample_agreement(rows, {"typos", "distract", "rephrase", "rename"})
        feats["shorter problem"] = [-r["problem_chars"] for r in rows]
        for name, vals in feats.items():
            if len(set(vals)) <= 1:
                continue
            a = auc(vals, labels)
            lo, hi = boot_ci(vals, labels)
            print(f"| {plan[:24]} | {Path(path).name} | {len(rows)} | {sum(labels)} | {name} | {a:.3f} | {lo:.2f}-{hi:.2f} | {median_split_acc(vals, labels):.3f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
