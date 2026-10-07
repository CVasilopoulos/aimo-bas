# /// script
# dependencies = ["pandas>=2.2", "pyarrow>=17"]
# ///
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--model-map", default="qwen3-8b:low=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument("--only-models", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    mapping = dict(item.split("=", 1) for item in args.model_map.split(",") if item)
    keep = set(args.only_models.split(",")) if args.only_models else None
    frame = pd.read_parquet(args.parquet)
    frame["model_id"] = frame["model_id"].map(lambda m: mapping.get(m, m))
    if keep:
        frame = frame[frame["model_id"].isin(keep)]
    cases = []
    labels = []
    seen = {}
    for row in frame.itertuples(index=False):
        key = (row.model_id, row.original_problem)
        label = bool(getattr(row, "model_is_robust", False)) if hasattr(row, "model_is_robust") else None
        if key in seen:
            if seen[key] != label:
                raise SystemExit(f"conflicting labels for {row.model_id} {row.problem_id}")
            continue
        seen[key] = label
        case_id = "case-" + hashlib.sha256((row.model_id + "\x00" + row.original_problem).encode()).hexdigest()[:20]
        cases.append({"id": case_id, "model_id": row.model_id, "problem": row.original_problem})
        if label is not None:
            labels.append({"id": case_id, "is_robust": label})
    if args.limit:
        cases = cases[: args.limit]
        ids = {c["id"] for c in cases}
        labels = [l for l in labels if l["id"] in ids]
    (args.target / "input").mkdir(parents=True, exist_ok=True)
    (args.target / "reference").mkdir(parents=True, exist_ok=True)
    (args.target / "input" / "cases.jsonl").write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cases))
    (args.target / "reference" / "labels.jsonl").write_text("".join(json.dumps(l, sort_keys=True) + "\n" for l in labels))
    (args.target / "metadata.json").write_text(json.dumps({"source": str(args.parquet), "cases": len(cases), "labels": len(labels), "robust": sum(l["is_robust"] for l in labels), "model_map": mapping}, indent=2, sort_keys=True) + "\n")
    print(len(cases), "cases", len(labels), "labels", sum(l["is_robust"] for l in labels), "robust")


if __name__ == "__main__":
    main()
