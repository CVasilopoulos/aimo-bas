# /// script
# dependencies = ["pandas>=2.2", "pyarrow>=17", "matplotlib>=3.9", "numpy>=2"]
# ///
from math import comb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, DARKGRAY, GRAY, INK2, ORANGE, SEED, SUBMISSIONS, save_fig, save_table, style

BOOT = 100_000
Z = 1.959963984540054
CONSTANT = {"small": [9, 10], "main": [10, 11]}
PREDICTED_POSITIVES = {"small": 10, "main": 12}


def bootstrap_ci(k, n, rng):
    outcomes = np.zeros(n)
    outcomes[:k] = 1
    draws = rng.choice(outcomes, size=(BOOT, n), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), draws


def wilson(k, n):
    p = k / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return centre - half, centre + half


def binom_cdf(k, n, p):
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, alpha=0.05):
    def solve(target, lo, hi):
        for _ in range(200):
            mid = (lo + hi) / 2
            if target(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) < alpha / 2, 0.0, 1.0)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) > alpha / 2, 0.0, 1.0)
    return lower, upper


def binom_two_sided(k, n):
    probs = [comb(n, i) * 0.5**n for i in range(n + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] + 1e-12))


def hypergeom_sf(tp, N, K, n):
    total = comb(N, n)
    return sum(comb(K, i) * comb(N - K, n - i) for i in range(tp, min(K, n) + 1)) / total


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) * 0.5**n for i in range(min(b, c) + 1)))


def score_rows(rng):
    rows = []
    draws = {}
    for track, meta in SUBMISSIONS.items():
        k, n = meta["correct"], meta["n"]
        lo, hi, d = bootstrap_ci(k, n, rng)
        draws[track] = d
        wl, wh = wilson(k, n)
        cl, ch = clopper_pearson(k, n)
        rows.append(
            {
                "track": track,
                "submission": meta["id"],
                "correct": k,
                "n": n,
                "accuracy": round(k / n, 3),
                "case_resolution": round(1 / n, 3),
                "bootstrap_lo": round(lo, 3),
                "bootstrap_hi": round(hi, 3),
                "wilson_lo": round(wl, 3),
                "wilson_hi": round(wh, 3),
                "clopper_pearson_lo": round(cl, 3),
                "clopper_pearson_hi": round(ch, 3),
                "binomial_p_vs_0.5": round(binom_two_sided(k, n), 3),
                "constant_low": round(CONSTANT[track][0] / n, 3),
                "constant_high": round(CONSTANT[track][1] / n, 3),
            }
        )
    return pd.DataFrame(rows), draws


def permutation_rows():
    rows = []
    for track, meta in SUBMISSIONS.items():
        k, n = meta["correct"], meta["n"]
        P = PREDICTED_POSITIVES[track]
        for r in range(n + 1):
            twice = k - (n - P - r)
            if twice % 2 or twice < 0:
                continue
            tp = twice // 2
            if tp > min(P, r) or tp < max(0, P + r - n):
                continue
            rows.append(
                {
                    "track": track,
                    "n": n,
                    "predicted_positive": P,
                    "true_robust": r,
                    "true_positive": tp,
                    "correct": k,
                    "p_random_ranking": round(hypergeom_sf(tp, n, r, P), 3),
                }
            )
    return pd.DataFrame(rows)


def mcnemar_rows():
    rows = []
    for track, meta in SUBMISSIONS.items():
        k, n = meta["correct"], meta["n"]
        for m in CONSTANT[track]:
            d = k - m
            for c in range(0, n + 1):
                b = c + d
                if b < 0 or b + c > n or b > k or c > m or c > n - k or b > n - m:
                    continue
                rows.append(
                    {
                        "track": track,
                        "bas_correct": k,
                        "constant_correct": m,
                        "n": n,
                        "b_bas_only": b,
                        "c_constant_only": c,
                        "discordant": b + c,
                        "mcnemar_exact_p": round(mcnemar_exact(b, c), 4),
                    }
                )
    return pd.DataFrame(rows)


def ceiling_rows():
    rows = []
    for label, n, m in (("v7 small (our run)", 19, 10), ("v7 main (our run)", 21, 11), ("v8 small (inferred)", 14, 9), ("v8 main (inferred)", 21, 13)):
        best = n - m
        rows.append({"hidden_set": label, "n": n, "constant_baseline_correct": m, "max_net_lead": best, "best_achievable_mcnemar_p": round(mcnemar_exact(best, 0), 4), "significant_at_0.05_possible": mcnemar_exact(best, 0) < 0.05})
    return pd.DataFrame(rows)


def lead_rows():
    rows = []
    for lead in range(1, 11):
        rows.append({"net_lead_cases": lead, "min_mcnemar_p": round(mcnemar_exact(lead, 0), 4), "significant_at_0.05": mcnemar_exact(lead, 0) < 0.05})
    return pd.DataFrame(rows)


def figure_scores(scores, draws):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.35})
    for ax, (_, row) in zip(axes, scores.iterrows()):
        track = row["track"]
        n = int(row["n"])
        d = draws[track]
        values, counts = np.unique(d, return_counts=True)
        ax.bar(values, counts / len(d), width=0.8 / n, color=GRAY)
        ax.axvline(row["accuracy"], color=BLUE, linewidth=1.5)
        ax.axvspan(row["bootstrap_lo"], row["bootstrap_hi"], color=BLUE, alpha=0.08, linewidth=0)
        ax.axvspan(row["constant_low"], row["constant_high"], color=ORANGE, alpha=0.18, linewidth=0)
        ax.text(row["accuracy"], ax.get_ylim()[1] * 0.98, f" {row['correct']}/{n} = {row['accuracy']:.3f}", color=BLUE, fontsize=8, va="top", ha="left")
        ax.text((row["constant_low"] + row["constant_high"]) / 2, ax.get_ylim()[1] * 0.02, "constant\nbaselines", color=ORANGE, fontsize=7, ha="center", va="bottom")
        ax.set_xlim(0.2, 1.0)
        ax.set_xlabel("accuracy")
        ax.set_ylabel("bootstrap mass")
        ax.grid(axis="y")
        ax.grid(False, axis="x")
        ax.set_title(f"{track.capitalize()}: 95% CI [{row['bootstrap_lo']:.2f}, {row['bootstrap_hi']:.2f}]", loc="left")
    save_fig(fig, "fig3_leaderboard_bootstrap")


def figure_lead(leads):
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.plot(leads["net_lead_cases"], leads["min_mcnemar_p"], color=BLUE, marker="o", markersize=4, linewidth=1.5)
    ax.axhline(0.05, color=INK2, linewidth=0.8, linestyle="--")
    ax.text(10, 0.055, "p = 0.05", ha="right", va="bottom", fontsize=8, color=INK2)
    for lead, label in ((1, "Small vs 10/19"), (3, "Main vs 11/21")):
        p = float(leads.loc[leads["net_lead_cases"] == lead, "min_mcnemar_p"].iloc[0])
        ax.annotate(label, (lead, p), textcoords="offset points", xytext=(8, 4), fontsize=7, color=DARKGRAY)
    ax.set_yscale("log")
    ax.set_xlabel("net cases gained over a constant baseline (none lost)")
    ax.set_ylabel("smallest exact McNemar p")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y")
    ax.grid(False, axis="x")
    save_fig(fig, "fig4_mcnemar_minimum_p")


def main():
    style()
    rng = np.random.default_rng(SEED)
    scores, draws = score_rows(rng)
    save_table(scores, "leaderboard_stats")
    save_table(permutation_rows(), "permutation_null")
    save_table(mcnemar_rows(), "mcnemar_feasibility")
    leads = lead_rows()
    save_table(leads, "mcnemar_minimum_lead")
    save_table(ceiling_rows(), "mcnemar_ceiling")
    figure_scores(scores, draws)
    figure_lead(leads)


if __name__ == "__main__":
    main()
