# /// script
# dependencies = ["pandas>=2.2", "pyarrow>=17", "matplotlib>=3.9", "numpy>=2"]
# ///
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, DARKGRAY, GRAY, INK2, MODEL_ORDER, SHORT, SUBMISSIONS, bar_labels, read_logs, save_fig, save_table, solution_path, style

solution_path()
from aimo_bas.profiles import active

SETTINGS = active(True)
THINK = SETTINGS["think_tokens"]


def enrich(logs):
    out = logs.copy()
    out["chunks"] = np.ceil(out["total_sequences"] / out["chunk"]).astype(int)
    out["tokens_per_trace"] = out["tokens"] / out["total_sequences"]
    out["tokens_per_s"] = out["tokens"] / out["gen_s"]
    out["overhead_s"] = out["elapsed_s"] - out["gen_s"] - out["force_s"]
    out["reserve_s"] = np.maximum(SETTINGS["force_reserve_min_s"], out["allotted_s"] * SETTINGS["force_reserve_frac"])
    out["generation_deadline_s"] = out["allotted_s"] - out["reserve_s"]
    out["hit_wall_clock"] = out["gen_s"] >= out["generation_deadline_s"] - 5.0
    out["all_traces_capped"] = out["tokens"] == out["total_sequences"] * THINK
    out["positives"] = (out["positive_rate"] * out["n"]).round().astype(int)
    out["budget_share"] = out["elapsed_s"] / out["allotted_s"]
    out["seconds_per_case"] = out["elapsed_s"] / out["n"]
    out["model_rank"] = out["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return out.sort_values(["track", "model_rank"]).reset_index(drop=True)


def totals(frame):
    rows = []
    for track, part in frame.groupby("track", sort=False):
        mid = part[part["model"].isin(["Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"])]
        unused = SETTINGS["total_budget_s"] - SETTINGS["safety_s"] - float(part["elapsed_s"].sum())
        rate = float(mid["gen_s"].sum()) / float(mid["tokens_per_trace"].mean())
        rows.append(
            {
                "track": track,
                "submission": SUBMISSIONS[track]["id"],
                "models": len(part),
                "cases": int(part["n"].sum()),
                "sequences": int(part["total_sequences"].sum()),
                "tokens": int(part["tokens"].sum()),
                "elapsed_s": round(float(part["elapsed_s"].sum()), 1),
                "gen_s": round(float(part["gen_s"].sum()), 1),
                "force_s": round(float(part["force_s"].sum()), 1),
                "share_of_3600": round(float(part["elapsed_s"].sum()) / SETTINGS["total_budget_s"], 3),
                "unused_beyond_safety_s": round(unused, 1),
                "extra_tokens_per_trace_affordable_7b_8b": int(unused / rate),
                "seconds_per_case_min": round(float(part["seconds_per_case"].min()), 1),
                "seconds_per_case_median": round(float(part["seconds_per_case"].median()), 1),
                "seconds_per_case_max": round(float(part["seconds_per_case"].max()), 1),
                "positives": int(part["positives"].sum()),
                "positive_rate": round(float(part["positives"].sum()) / float(part["n"].sum()), 3),
                "models_hit_wall_clock": int(part["hit_wall_clock"].sum()),
                "models_all_traces_capped": int(part["all_traces_capped"].sum()),
                "correct": SUBMISSIONS[track]["correct"],
                "accuracy": round(SUBMISSIONS[track]["correct"] / SUBMISSIONS[track]["n"], 3),
            }
        )
    return pd.DataFrame(rows)


def figure_budget(frame):
    tracks = ["small", "main"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharex=True, gridspec_kw={"wspace": 0.85})
    for ax, track in zip(axes, tracks):
        part = frame[frame["track"] == track].iloc[::-1]
        y = np.arange(len(part))
        ax.barh(y + 0.2, part["allotted_s"], height=0.36, color=GRAY, label="allotted")
        bars = ax.barh(y - 0.2, part["elapsed_s"], height=0.36, color=BLUE, label="used")
        ax.set_yticks(y)
        ax.set_yticklabels(part["model_short"])
        ax.set_xlim(0, 1450)
        bar_labels(ax, bars, lambda w: f"{w:.0f} s")
        ax.set_title(f"{track.capitalize()} track, submission {SUBMISSIONS[track]['id']}", loc="left")
        ax.set_xlabel("seconds")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    save_fig(fig, "fig1_runtime_budget")


def figure_traces(frame):
    tracks = ["small", "main"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.55})
    labels = []
    tokens = []
    finish = []
    colors = []
    for track in tracks:
        part = frame[frame["track"] == track]
        for _, row in part.iterrows():
            labels.append(f"{row['model_short']} ({track})")
            tokens.append(row["tokens_per_trace"])
            finish.append(row["mean_finish"])
            colors.append(BLUE if track == "small" else DARKGRAY)
    y = np.arange(len(labels))[::-1]
    bars = axes[0].barh(y, tokens, height=0.6, color=colors)
    axes[0].axvline(THINK, color=INK2, linewidth=0.8, linestyle="--")
    axes[0].text(THINK, len(labels) - 0.4, f"cap {THINK}", ha="right", va="bottom", fontsize=8, color=INK2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlim(0, 7200)
    axes[0].set_xlabel("mean generated tokens per trace")
    axes[0].set_title("Trace length", loc="left")
    bars = axes[1].barh(y, finish, height=0.6, color=colors)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlim(0, 1.15)
    axes[1].set_xlabel("share of traces that closed reasoning")
    axes[1].set_title("Finish rate", loc="left")
    bar_labels(axes[1], bars, lambda w: f"{w:.2f}")
    save_fig(fig, "fig2_trace_length_finish")


def main():
    style()
    frame = enrich(read_logs())
    columns = [
        "track", "submission", "model", "n", "variants", "total_sequences", "generated_sequences", "chunk", "chunks",
        "allotted_s", "elapsed_s", "gen_s", "force_s", "overhead_s", "generation_deadline_s", "hit_wall_clock",
        "tokens", "tokens_per_trace", "tokens_per_s", "all_traces_capped", "mean_finish", "mean_agree", "positives", "positive_rate", "budget_share", "seconds_per_case",
    ]
    table = frame[columns].copy()
    for col in ("overhead_s", "generation_deadline_s", "tokens_per_trace", "tokens_per_s", "budget_share", "seconds_per_case"):
        table[col] = table[col].round(3 if col == "budget_share" else 1)
    save_table(table, "runtime_per_model")
    save_table(totals(frame), "runtime_totals")
    figure_budget(frame)
    figure_traces(frame)


if __name__ == "__main__":
    main()
