# Budgeted Answer Stability (BAS) for the AIMO Interpretability Challenge

Code and data behind our technical report for the AIMO Interpretability Challenge (Codabench 16180): a label-free, compute-bounded probe that predicts whether a model's correct olympiad answer survives answer-preserving perturbations. The full report is in `report/report.md`.

BAS samples each problem in its original form and in cheap prose-only perturbations under a fixed thinking budget, forces the model to commit to an answer at the cut (the answer-commitment probe), and ranks problems by answer agreement, reasoning completion and commitment confidence. It runs unchanged on every competition model, inside one GPU and 3600 s.

Results in brief (details and confidence intervals in the report): 11/19 on the Small track and 14/21 on Main for the scored v1 bundles; on the replacement hidden sets, four of six budget profiles tie a constant predictor on Main; on 52 labelled public MATH cases no BAS feature separates robust from non-robust cases (every AUC interval contains 0.5). The report explains why and what 14-21 hidden cases can and cannot resolve.

## Layout

| Path | Content |
|---|---|
| `solution/` | the submission code: `solution.py` and `aimo_bas/` (runner, engine, perturbations, answer extraction, profiles) |
| `submissions/aimo-bas-v1-{small,main}.zip` | the two bundles as scored (SHA-256 in `report/tables/submissions.csv`) |
| `submissions/logs/` | runtime logs of the two scored runs (verbatim stderr lines) |
| `tools/build.py` | rebuilds a bundle: `python3 tools/build.py --name aimo-bas-v1 [--profile gpu-fast]` (deterministic zip timestamps) |
| `tools/run_local.sh` | local contract test inside the organisers' runtime image with `--network none` |
| `tools/make_cases.py`, `tools/ablate.py`, `tools/commitment_curve.py` | public-data cases, the ablation driver and the commitment-curve experiment |
| `tools/report/` | scripts that regenerate every table and figure of the report (`run_all.sh`; Table 3 from `ablation_tables.py`) |
| `report/` | the report, its figures and tables |
| `research/` | leaderboard and task snapshots used in the report |
| `runs/kaggle_ablate/` | per-case output of the GPU ablation (Table 3) |
| `data/smoke-qwen/` | synthetic smoke cases for the local contract test |
| `tests/test_offline.py` | offline unit tests (`python3 -m unittest tests/test_offline.py`) |

## Reproducing

1. **Bundles.** `python3 tools/build.py --name aimo-bas-v1` rebuilds the scored bundles from `solution/` and prints their hashes. To score one, upload it to the matching Codabench task; the hidden sets were replaced after our scored runs, so a new score is on the replacement sets.
2. **Local run.** `tools/run_local.sh submissions/aimo-bas-v1-small.zip data/smoke-qwen small` (needs Docker and the organisers' runtime image named in the report).
3. **Report tables and figures.** `tools/report/run_all.sh` (uses `uv run`; each script declares its dependencies inline). The public datasets are not redistributed here: download them from Hugging Face at the revisions listed in Section 7 of the report into `research/hf/`, and regenerate the case files with `tools/make_cases.py`.
4. **GPU ablation (Table 3).** `tools/ablate.py` produced `runs/kaggle_ablate/*.jsonl` on two T4 GPUs; `python3 tools/report/ablation_tables.py` recomputes the table from them.

## License

MIT, see `LICENSE`.

AI-assisted, human-directed.
