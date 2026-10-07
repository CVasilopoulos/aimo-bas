import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
models = sys.argv[3].split(",")
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 8

cases = [json.loads(line) for line in (source / "input" / "cases.jsonl").read_text().splitlines() if line.strip()]
labels = {row["id"]: row["is_robust"] for row in map(json.loads, (source / "reference" / "labels.jsonl").read_text().splitlines()) if row}
votes = defaultdict(list)
for case in cases:
    votes[case["problem"]].append(labels[case["id"]])
problems = sorted(votes, key=lambda p: hashlib.sha256(p.encode()).hexdigest())[:limit]
out_cases = []
out_labels = []
for model in models:
    for problem in problems:
        case_id = "smoke-" + hashlib.sha256((model + "\x00" + problem).encode()).hexdigest()[:20]
        out_cases.append({"id": case_id, "model_id": model, "problem": problem})
        out_labels.append({"id": case_id, "is_robust": sum(votes[problem]) * 2 > len(votes[problem])})
(target / "input").mkdir(parents=True, exist_ok=True)
(target / "reference").mkdir(parents=True, exist_ok=True)
(target / "input" / "cases.jsonl").write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in out_cases))
(target / "reference" / "labels.jsonl").write_text("".join(json.dumps(l, sort_keys=True) + "\n" for l in out_labels))
print(len(out_cases), "cases", sum(l["is_robust"] for l in out_labels), "robust")
