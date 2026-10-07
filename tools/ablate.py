import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "solution")
    parser.add_argument("--models", default="")
    parser.add_argument("--track", default="small")
    parser.add_argument("--profile", default="")
    parser.add_argument("--think-tokens", type=int)
    parser.add_argument("--variants", default="")
    parser.add_argument("--budget-s", type=float)
    parser.add_argument("--score-weights", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import os

    if args.profile:
        os.environ["AIMO_BAS_PROFILE"] = args.profile
    from aimo_bas import profiles, runner

    original_active = profiles.active

    def patched_active(cuda):
        settings = original_active(cuda)
        if args.think_tokens:
            settings["think_tokens"] = args.think_tokens
        if args.variants:
            settings["variant_plans"] = [[10**6, args.variants.split(",")]]
        if args.score_weights:
            settings["score_weights"] = [float(x) for x in args.score_weights.split(",")]
        return settings

    profiles.active = patched_active
    runner._track = lambda: args.track
    if args.budget_s:
        runner.allot_seconds = lambda settings, model_id, track, now: args.budget_s
    captured = []
    original_features = runner.problem_features

    def spy(records, kinds, weights):
        features = original_features(records, kinds, weights)
        captured.append({"features": features, "samples": [dict(r, kind=k) for r, k in zip(records, kinds)]})
        return features

    runner.problem_features = spy
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if args.limit:
        cases = cases[: args.limit]
    labels = {}
    if args.labels and args.labels.is_file():
        for line in args.labels.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                labels[row["id"]] = row["is_robust"]
    wanted = set(args.models.split(",")) if args.models else None
    batches = {}
    for case in cases:
        if wanted and case["model_id"] not in wanted:
            continue
        batches.setdefault(case["model_id"], []).append(case)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as stream:
        for model_id, batch in batches.items():
            captured.clear()
            started = time.monotonic()
            predictions = runner.are_robust(model_id, [c["problem"] for c in batch])
            elapsed = time.monotonic() - started
            for index, (case, prediction) in enumerate(zip(batch, predictions)):
                record = {
                    "case_id": case["id"],
                    "model_id": model_id,
                    "problem_sha256": hashlib.sha256(case["problem"].encode()).hexdigest()[:16],
                    "problem_chars": len(case["problem"]),
                    "label": labels.get(case["id"]),
                    "prediction": bool(prediction),
                    "think_tokens": args.think_tokens,
                    "variants": args.variants,
                    "profile": args.profile,
                    "batch_elapsed_s": round(elapsed, 1),
                    "batch_size": len(batch),
                }
                if index < len(captured):
                    record["features"] = captured[index]["features"]
                    record["samples"] = [{k: s[k] for k in ("kind", "answer", "closed", "confidence", "tokens", "ran") if k in s} for s in captured[index]["samples"]]
                else:
                    record["mode"] = "fallback"
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
