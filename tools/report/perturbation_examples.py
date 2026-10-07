# /// script
# dependencies = ["pandas>=2.2", "pyarrow>=17", "matplotlib>=3.9", "numpy>=2"]
# ///
import difflib
import json
import re

import pandas as pd

from common import read_parquet, save_table, save_text, solution_path

solution_path()
from aimo_bas import perturb

EXAMPLE_PROBLEM_ID = "a1d40b"
EXAMPLE_MODEL = "gpt-oss-120b:low"
WORD = re.compile(r"[A-Za-z]+")


def word_diff(a, b):
    wa = WORD.findall(a)
    wb = WORD.findall(b)
    changed = 0
    added = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).get_opcodes():
        if tag == "replace":
            changed += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            changed += i2 - i1
        elif tag == "insert":
            added += j2 - j1
    return changed, added


def footprint(problems):
    rows = []
    for text in problems:
        for kind in ("typos", "distract", "rephrase", "rename"):
            variant = perturb.variant(text, kind)
            changed, added = word_diff(text, variant)
            rows.append(
                {
                    "kind": kind,
                    "problem_chars": len(text),
                    "variant_chars": len(variant),
                    "added_chars": len(variant) - len(text),
                    "changed_words": changed,
                    "added_words": added,
                    "math_spans_preserved": all(span in variant for span in perturb.MATH_SPAN.findall(text)),
                    "digits_preserved": re.findall(r"\d+", text) == re.findall(r"\d+", variant),
                    "rename_applied": kind == "rename" and perturb.rename(text) is not None,
                }
            )
    frame = pd.DataFrame(rows)
    summary = frame.groupby("kind").agg(
        problems=("kind", "size"),
        mean_added_chars=("added_chars", "mean"),
        mean_changed_words=("changed_words", "mean"),
        mean_added_words=("added_words", "mean"),
        math_spans_preserved=("math_spans_preserved", "all"),
        digits_preserved=("digits_preserved", "all"),
        rename_applied=("rename_applied", "sum"),
    ).reset_index()
    summary["mean_added_chars"] = summary["mean_added_chars"].round(1)
    summary["mean_changed_words"] = summary["mean_changed_words"].round(2)
    summary["mean_added_words"] = summary["mean_added_words"].round(2)
    return summary


def main():
    rows = read_parquet("val-sample")
    problems = sorted(rows["original_problem"].unique())
    save_table(footprint(problems), "perturbation_footprint")
    example = rows[rows["problem_id"] == EXAMPLE_PROBLEM_ID].iloc[0]["original_problem"]
    organiser = rows[(rows["model_id"] == EXAMPLE_MODEL) & (rows["problem_id"] == EXAMPLE_PROBLEM_ID) & rows["permutations_causing_decay"].str.contains("distract")]
    organiser_variant = ""
    if len(organiser):
        for item in json.loads(organiser.iloc[0]["permutations_causing_decay"]):
            if item["permutation_type"] == "distract":
                organiser_variant = item["question"]
                break
    lines = [f"Public AIMO 2025 reference problem `{EXAMPLE_PROBLEM_ID}` from `aimo-interp/val-sample`.", "", "**Original**", "", example, ""]
    for kind in ("typos", "distract", "rephrase", "rename"):
        lines += [f"**BAS v1 `{kind}`**" + (" (unused in v1; falls back to rephrase(typos) when no name is present)" if kind == "rename" else ""), "", perturb.variant(example, kind), ""]
    if organiser_variant:
        lines += [f"**Organisers' `distract` variant for the same problem** (from the `permutations_causing_decay` field of the `{EXAMPLE_MODEL}` row)", "", organiser_variant, ""]
    save_text("\n".join(lines), "perturbation_examples.md")


if __name__ == "__main__":
    main()
