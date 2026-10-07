import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"
TABLES = REPORT / "tables"
LOGS = ROOT / "submissions" / "logs"
HF = ROOT / "research" / "hf"
SOLUTION = ROOT / "solution"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#c3c1ba"
DARKGRAY = "#8a8880"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e4df"

SHORT = {
    "Qwen/Qwen3.5-4B": "Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B": "Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think": "Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-0528-Qwen3-8B",
    "openai/gpt-oss-120b": "gpt-oss-120b",
}
MODEL_ORDER = list(SHORT)
V1_FAMILIES = ("typos", "distract", "rephrase")
ALIASES = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
SUBMISSIONS = {
    "small": {"id": 927463, "log": "927463_small_aimo-bas.jsonl", "correct": 11, "n": 19, "bundle": "aimo-bas-v1-small.zip"},
    "main": {"id": 927466, "log": "927466_main_aimo-bas.jsonl", "correct": 14, "n": 21, "bundle": "aimo-bas-v1-main.zip"},
}
SEED = 20261101


def style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.facecolor": "white",
        }
    )


def solution_path():
    if str(SOLUTION) not in sys.path:
        sys.path.insert(0, str(SOLUTION))


def read_logs():
    frames = []
    for track, meta in SUBMISSIONS.items():
        rows = [json.loads(line) for line in (LOGS / meta["log"]).read_text().splitlines() if line.strip()]
        frame = pd.DataFrame(rows)
        frame["submission"] = meta["id"]
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["model_short"] = out["model"].map(SHORT)
    return out


def read_parquet(name):
    return pd.read_parquet(HF / f"{name}.parquet")


def families(text):
    try:
        return sorted({item["permutation_type"] for item in json.loads(text)})
    except (TypeError, ValueError, KeyError):
        return []


def family_class(names):
    if not names:
        return "none"
    inside = set(names) & set(V1_FAMILIES)
    if inside == set(names):
        return "v1-families-only"
    if inside:
        return "mixed"
    return "outside-v1-families"


def val_cases():
    cases = [json.loads(line) for line in (ROOT / "data/val-sample/input/cases.jsonl").read_text().splitlines() if line.strip()]
    labels = {}
    for line in (ROOT / "data/val-sample/reference/labels.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["id"]] = row["is_robust"]
    rows = read_parquet("val-sample")
    rows["fams"] = rows["permutations_causing_decay"].apply(families)
    grouped = rows.groupby(["model_id", "original_problem"])
    stats = grouped.agg(
        problem_id=("problem_id", "first"),
        base_accuracy=("base_accuracy", "first"),
        n_base_predictions=("n_base_predictions", "first"),
        rows=("problem_id", "size"),
    )
    union = grouped["fams"].agg(lambda s: sorted({f for l in s for f in l}))
    stats["decay_families"] = union
    out = []
    for case in cases:
        key = (case["model_id"], case["problem"])
        s = stats.loc[key]
        out.append(
            {
                "case_id": case["id"],
                "model_id": case["model_id"],
                "problem_id": s["problem_id"],
                "problem_chars": len(case["problem"]),
                "base_accuracy": float(s["base_accuracy"]),
                "n_base_predictions": int(s["n_base_predictions"]),
                "hf_rows": int(s["rows"]),
                "decay_families": " ".join(s["decay_families"]),
                "family_class": family_class(s["decay_families"]),
                "label": bool(labels[case["id"]]),
            }
        )
    return pd.DataFrame(out)


def save_fig(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def save_table(frame, name, index=False):
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    frame.to_csv(path, index=index)
    print("wrote", path.relative_to(ROOT), f"({len(frame)} rows)")


def save_text(text, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    path.write_text(text, encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


def bar_labels(ax, bars, fmt, offset=0.01, color=INK2):
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    for bar in bars:
        width = bar.get_width()
        ax.text(width + span * offset, bar.get_y() + bar.get_height() / 2, fmt(width), va="center", ha="left", fontsize=8, color=color)


def jitter(rng, n, width=0.12):
    return rng.uniform(-width, width, size=n)


__all__ = [name for name in dir() if not name.startswith("_")]
