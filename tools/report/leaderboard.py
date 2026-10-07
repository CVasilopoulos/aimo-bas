# /// script
# dependencies = ["pandas>=2.2", "pyarrow>=17", "matplotlib>=3.9", "numpy>=2"]
# ///
import json
from collections import Counter
from fractions import Fraction

import pandas as pd

from common import ROOT, save_table

SNAPSHOTS = [("2026-09-16", "comp.json", "lb_30038.json"), ("2026-09-23", "comp_2026-09-23.json", "lb_30038_2026-09-23.json")]
OWN = {927463: "small", 927466: "main"}


def task_names(comp):
    names = {}
    for phase in comp["phases"]:
        for task in phase["tasks"]:
            names[task["id"]] = (phase["name"], task["name"], task["created_when"][:16])
    return names


def infer_n(scores, candidates=range(10, 40)):
    fits = []
    for n in candidates:
        ok = all(any(round(k / n, 2) == round(float(s), 2) for k in range(n + 1)) for s in scores)
        if ok:
            fits.append(n)
    return fits


def main():
    rows = []
    own = []
    for date, comp_file, lb_file in SNAPSHOTS:
        comp = json.loads((ROOT / "research" / comp_file).read_text())
        lb = json.loads((ROOT / "research" / lb_file).read_text())
        names = task_names(comp)
        by_task = {}
        for sub in lb["submissions"]:
            acc = [s for s in sub["scores"] if s["column_key"] == "accuracy"]
            if not acc:
                continue
            task = acc[0]["task_id"]
            by_task.setdefault(task, []).append((sub["owner"], acc[0]["score"], sub["created_when"][:16], sub["id"]))
            if sub["id"] in OWN:
                own.append({"snapshot": date, "submission": sub["id"], "task": task, "score": acc[0]["score"]})
        for task, entries in sorted(by_task.items()):
            scores = [e[1] for e in entries]
            counts = Counter(scores)
            phase, name, created = names.get(task, ("?", "?", "?"))
            fits = infer_n(set(scores))
            rows.append(
                {
                    "snapshot": date,
                    "phase": phase,
                    "task_id": task,
                    "task_name": name,
                    "task_created": created,
                    "entries": len(entries),
                    "participants": len({e[0] for e in entries}),
                    "distinct_scores": " ".join(sorted(counts, key=float, reverse=True)),
                    "most_common_score": counts.most_common(1)[0][0],
                    "most_common_count": counts.most_common(1)[0][1],
                    "best_score": max(scores, key=float),
                    "first_entry": min(e[2] for e in entries),
                    "last_entry": max(e[2] for e in entries),
                    "consistent_case_counts_10_39": " ".join(str(n) for n in fits),
                    "listed_in_api_page": len(lb["submissions"]),
                }
            )
    save_table(pd.DataFrame(rows), "leaderboard_snapshots")
    save_table(pd.DataFrame(own or [{"snapshot": "none", "submission": None, "task": None, "score": None}]), "leaderboard_own_entries")
    meta = []
    for date, comp_file, _ in SNAPSHOTS:
        comp = json.loads((ROOT / "research" / comp_file).read_text())
        for phase in comp["phases"]:
            for task in phase["tasks"]:
                meta.append({"snapshot": date, "phase_id": phase["id"], "phase": phase["name"], "phase_end": phase["end"][:16], "task_id": task["id"], "task_name": task["name"], "task_created": task["created_when"][:16], "participants_count": comp.get("participants_count"), "submissions_count": comp.get("submissions_count")})
    save_table(pd.DataFrame(meta), "competition_tasks")


if __name__ == "__main__":
    main()
