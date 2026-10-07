# /// script
# dependencies = ["pandas>=2.2", "pyarrow>=17", "matplotlib>=3.9", "numpy>=2"]
# ///
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ALIASES, BLUE, DARKGRAY, GRAY, INK2, ORANGE, SEED, V1_FAMILIES, families, family_class, jitter, read_parquet, save_fig, save_table, solution_path, style, val_cases

solution_path()
from aimo_bas.runner import length_prior

THRESHOLDS = (0.3, 0.5)


def add_rules(frame):
    out = frame.copy()
    out["rule_always_true"] = True
    out["rule_always_false"] = False
    out["rule_base_eq_1"] = out["base_accuracy"] >= 1.0
    prior = pd.Series(index=out.index, dtype=bool)
    for _, idx in out.groupby("model_id").groups.items():
        lengths = [int(out.loc[i, "problem_chars"]) for i in idx]
        prior.loc[list(idx)] = length_prior(["x" * n for n in lengths])
    out["rule_length_prior"] = prior.astype(bool)
    for rule in ("rule_always_true", "rule_always_false", "rule_base_eq_1", "rule_length_prior"):
        out[rule + "_correct"] = out[rule] == out["label"]
    return out


def rule_summary(frame, name):
    rows = []
    for rule in ("rule_always_true", "rule_always_false", "rule_base_eq_1", "rule_length_prior"):
        correct = int(frame[rule + "_correct"].sum())
        rows.append({"dataset": name, "cases": len(frame), "robust": int(frame["label"].sum()), "rule": rule.replace("rule_", ""), "correct": correct, "accuracy": round(correct / len(frame), 3)})
    return rows


def math_cases(name, model_map):
    rows = read_parquet(name)
    rows["model_id"] = rows["model_id"].map(lambda m: model_map.get(m, m))
    rows["fams"] = rows["permutations_causing_decay"].apply(families)
    grouped = rows.groupby(["model_id", "original_problem"], sort=False)
    out = grouped.agg(
        problem_id=("problem_id", "first"),
        dataset_id=("dataset_id", "first"),
        base_accuracy=("base_accuracy", "first"),
        n_base_predictions=("n_base_predictions", "first"),
        label=("model_is_robust", "first"),
        hf_rows=("problem_id", "size"),
        n_detrimental=("n_detrimental_permutations", "first"),
    ).reset_index()
    out["decay_families"] = grouped["fams"].agg(lambda s: " ".join(sorted({f for l in s for f in l}))).values
    out["family_class"] = grouped["fams"].agg(lambda s: family_class(sorted({f for l in s for f in l}))).values
    out["problem_chars"] = out["original_problem"].str.len()
    out["case_id"] = [f"{name}-{i:03d}" for i in range(len(out))]
    out["label"] = out["label"].astype(bool)
    return out.drop(columns=["original_problem"])


def alias_check():
    agg = read_parquet("augmented-sample-math-agg")
    filtered = read_parquet("augmented-sample-math-agg-filtered")
    test = agg[agg["dataset_id"] == "hendrycks-math-test-reference-100"]
    merged = test.merge(filtered, on=["problem_id", "original_problem"], suffixes=("_agg", "_filtered"))
    return pd.DataFrame(
        [
            {
                "agg_model_id": ",".join(sorted(agg["model_id"].unique())),
                "filtered_model_id": ",".join(sorted(filtered["model_id"].unique())),
                "agg_test_rows": len(test),
                "filtered_rows": len(filtered),
                "merged_rows": len(merged),
                "identical_base_accuracy": bool((merged["base_accuracy_agg"] == merged["base_accuracy_filtered"]).all()),
                "identical_label": bool((merged["model_is_robust_agg"] == merged["model_is_robust_filtered"]).all()),
                "identical_permuted_accuracy": bool((merged["permuted_accuracy_agg"] == merged["permuted_accuracy_filtered"]).all()),
            }
        ]
    )


def family_counts(frame, name):
    rows = []
    negatives = frame[~frame["label"]]
    for cls in ("v1-families-only", "mixed", "outside-v1-families", "none"):
        rows.append({"dataset": name, "non_robust_cases": len(negatives), "family_class": cls, "cases": int((negatives["family_class"] == cls).sum())})
    counts = {}
    for text in negatives["decay_families"]:
        for f in text.split():
            counts[f] = counts.get(f, 0) + 1
    for f, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        rows.append({"dataset": name, "non_robust_cases": len(negatives), "family_class": f"family:{f}", "cases": c})
    return rows


def decay_by_type(frame, name):
    rows = []
    for ptype, part in frame.groupby("permutation_type"):
        row = {
            "dataset": name,
            "permutation_type": ptype,
            "emulated_by_bas_v1": ptype in V1_FAMILIES,
            "pairs": len(part),
            "models": part["model_id"].nunique(),
            "mean_base_accuracy": round(float(part["base_accuracy"].mean()), 3),
            "mean_permuted_accuracy": round(float(part["permuted_accuracy"].mean()), 3),
            "mean_absolute_decay": round(float(part["absolute_accuracy_decay"].mean()), 3),
        }
        for t in THRESHOLDS:
            row[f"share_decay_ge_{t}"] = round(float((part["absolute_accuracy_decay"] >= t).mean()), 3)
        rows.append(row)
    return rows


def figure_base_accuracy(val, math):
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.4})
    for ax, frame, title in zip(axes, (val, math), (f"val-sample (AIMO reference, {len(val)} cases)", f"MATH test-100, DeepSeek-R1-0528-Qwen3-8B ({len(math)} cases)")):
        for label, color, x0 in ((True, BLUE, 1.0), (False, ORANGE, 0.0)):
            part = frame[frame["label"] == label]
            ax.scatter(part["base_accuracy"] + jitter(rng, len(part), 0.02), x0 + jitter(rng, len(part), 0.18), s=14, color=color, alpha=0.85, linewidths=0)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["non-robust", "robust"])
        ax.set_xlabel("base accuracy on the original problem")
        ax.set_xlim(0.5, 1.05)
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(title, loc="left")
        ax.grid(False, axis="x")
    save_fig(fig, "fig5_base_accuracy_vs_label")


def figure_decay(decay):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.6})
    panels = (("aimo-sample-full", "AIMO reference problems, 7 models"), ("math-full-filtered-deepseek", "MATH, DeepSeek-R1-0528-Qwen3-8B"))
    for ax, (name, title) in zip(axes, panels):
        part = decay[decay["dataset"] == name].sort_values("share_decay_ge_0.3")
        colors = [BLUE if e else GRAY for e in part["emulated_by_bas_v1"]]
        y = np.arange(len(part))
        bars = ax.barh(y, part["share_decay_ge_0.3"], color=colors, height=0.65)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{t} (n={n})" for t, n in zip(part["permutation_type"], part["pairs"])])
        ax.set_xlim(0, 0.8)
        ax.set_title(title, loc="left")
        for bar, v in zip(bars, part["share_decay_ge_0.3"]):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=7, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=GRAY)]
    axes[0].legend(handles, ["family emulated by BAS v1", "not emulated"], loc="lower right")
    fig.supxlabel("share of (model, problem) pairs whose accuracy drops by 0.3 or more under the perturbation", fontsize=9, color=INK2, y=-0.04)
    save_fig(fig, "fig6_perturbation_family_decay")


def main():
    style()
    val = add_rules(val_cases())
    save_table(val, "val_sample_cases")
    math_test = add_rules(math_cases("augmented-sample-math-agg-filtered", ALIASES))
    save_table(math_test, "math_deepseek_test_cases")
    math_all = add_rules(math_cases("augmented-sample-math-agg", ALIASES))
    save_table(math_all, "math_deepseek_all_cases")
    rules = rule_summary(val, "val-sample") + rule_summary(math_test, "math-deepseek-test-100") + rule_summary(math_all, "math-deepseek-all-141")
    save_table(pd.DataFrame(rules), "proxy_rules")
    save_table(alias_check(), "dataset_alias_check")
    fams = family_counts(val, "val-sample") + family_counts(math_test, "math-deepseek-test-100") + family_counts(math_all, "math-deepseek-all-141")
    save_table(pd.DataFrame(fams), "label_families")
    full = read_parquet("aimo-interp-challenge-sample-full")
    full_ds = full[full["model_id"] == "qwen3-8b:low"]
    math_full = read_parquet("augmented-sample-math-full-filtered")
    decay = pd.DataFrame(decay_by_type(full, "aimo-sample-full") + decay_by_type(full_ds, "aimo-sample-full-deepseek") + decay_by_type(math_full, "math-full-filtered-deepseek"))
    save_table(decay, "perturbation_decay")
    crosstab = pd.crosstab(val["base_accuracy"] >= 1.0, val["label"]).rename(index={True: "base_eq_1", False: "base_lt_1"}, columns={True: "robust", False: "non_robust"})
    save_table(crosstab.reset_index().rename(columns={"base_accuracy": "group"}), "val_sample_crosstab")
    lengths = pd.DataFrame(
        [
            {"dataset": "val-sample", "label": "robust", "cases": int(val["label"].sum()), "mean_chars": round(float(val[val["label"]]["problem_chars"].mean()), 1), "median_chars": float(val[val["label"]]["problem_chars"].median())},
            {"dataset": "val-sample", "label": "non-robust", "cases": int((~val["label"]).sum()), "mean_chars": round(float(val[~val["label"]]["problem_chars"].mean()), 1), "median_chars": float(val[~val["label"]]["problem_chars"].median())},
            {"dataset": "math-deepseek-all-141", "label": "robust", "cases": int(math_all["label"].sum()), "mean_chars": round(float(math_all[math_all["label"]]["problem_chars"].mean()), 1), "median_chars": float(math_all[math_all["label"]]["problem_chars"].median())},
            {"dataset": "math-deepseek-all-141", "label": "non-robust", "cases": int((~math_all["label"]).sum()), "mean_chars": round(float(math_all[~math_all["label"]]["problem_chars"].mean()), 1), "median_chars": float(math_all[~math_all["label"]]["problem_chars"].median())},
        ]
    )
    save_table(lengths, "problem_length_by_label")
    figure_base_accuracy(val, math_test)
    figure_decay(decay)


if __name__ == "__main__":
    main()
