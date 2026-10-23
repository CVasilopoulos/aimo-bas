# Budgeted Answer Stability (BAS) for the AIMO Interpretability Challenge

Code, scored submission bundles and data behind our technical report for the AIMO Interpretability Challenge (Codabench 16180). The report is `report/report.md`.

BAS is a label-free, compute-bounded probe that predicts whether a model's correct olympiad answer survives answer-preserving perturbations. It samples each problem in its original form and in cheap prose-only perturbations under a fixed thinking budget, forces the model to commit to an answer at the cut (the answer-commitment probe), and ranks problems by answer agreement, reasoning completion and commitment confidence. It runs unchanged on every competition model, inside one GPU and 3600 s. It has no trained parameters: there is no training data and no training codebase.

Results in brief (details and intervals in the report): 11/19 (Small) and 14/21 (Main) on the retired v7 hidden sets; 6/14 and 13/21 for the same bundles on the v8 sets, matched exactly by a rebuild with the same settings; on v8 Main, four of six budget profiles tie a constant `True` predictor. On 52 labelled public MATH cases no BAS feature separates robust from non-robust cases (every AUC interval contains 0.5), and a dedicated ablation of the only white-box signal, the forced answer and its token probability, moves no ranking measurably.

## Bundles and scores

All 14 bundles we uploaded are in `submissions/`; `report/tables/submissions.csv` (v7) and `report/tables/submissions_v8.csv` (v8) list their SHA-256, Codabench ids and scores, and `submissions/logs/codabench_own_submissions_2026-10-02.txt` is the Codabench listing they come from. The primary submission is the v1 pair:

| Track | Bundle | SHA-256 | v7 score | v8 score |
|---|---|---|---|---|
| Small | `submissions/aimo-bas-v1-small.zip` | `685584fed46bd6ec3279169f41eb55899432b8c427f099ae400c9fe0f396bf20` | 11/19 (927463) | 6/14 (941825; same-settings rebuild 949033: 6/14) |
| Main | `submissions/aimo-bas-v1-main.zip` | `463729c8a5367f16791057b036a467b02a17b865fd1300dd75d7bbab207b588c` | 14/21 (927466) | 13/21 (941828; same-settings rebuild 948991: 13/21) |

To score a bundle, upload it unchanged to the matching Codabench task (a Small bundle contains `small.txt`; the profile is read from `profile.txt`, default `gpu`). The bundle contains code only: it loads the five competition models by id from the organisers' offline cache (`local_files_only`), uses only the Python packages of the organisers' runtime image (`torch`, `transformers`, `accelerate`), and reads no other file.

## Layout

| Path | Content |
|---|---|
| `solution/` | submission code: `solution.py` and `aimo_bas/` (runner, engine, perturbations, answer extraction, profiles) |
| `submissions/*.zip` | the 14 bundles as uploaded: v1 (`gpu`), `gpu` rebuild, `fast`, `faster`, `fast-wide`, `fast-tight`, `long`, each for Main and Small |
| `submissions/logs/` | runtime logs of the two v7 runs, the Codabench listing of our submissions, the v8 sweep runner's log and results |
| `tools/build.py` | builds a bundle pair: `python3 tools/build.py --name aimo-bas-v1 [--profile gpu-fast]` (fixed zip timestamps) |
| `tools/verify_bundles.py` | checks every bundle against its manifest hash, rebuilds it byte-identically and diffs it against `solution/` |
| `tools/fetch_inputs.py` | fetches the organisers' public datasets, model configs and repositories at pinned revisions, checking SHA-256 |
| `tools/fetch_probe_models.sh` | downloads the two small probe models of the local contract test at pinned revisions into `hf-cache/` |
| `tools/run_local.sh` | local contract test inside the organisers' runtime image with `--network none` |
| `tools/make_cases.py`, `tools/ablate.py`, `tools/kaggle_ablation.py`, `tools/commitment_curve.py` | case files, the feature-dump driver, the GPU ablation driver behind Table 3, and the commitment-curve experiment |
| `tools/report/` | scripts that regenerate every table and figure (`run_all.sh`; Table 3 from `ablation_tables.py`; the white-box ablation from `whitebox_ablation.py`) |
| `report/` | the report, its figures and tables |
| `research/` | leaderboard and task snapshots used in the report; fetched inputs land here too (not committed) |
| `runs/kaggle_ablate/` | per-case output of the GPU ablation |
| `data/smoke-qwen/` | 16 smoke cases for the local contract test |
| `tests/` | offline unit tests |

## Reproducing

Verified from a fresh clone on 2026-10-24 on a Linux CPU host (Python 3.12, Docker, a container capped at 6 GB and 6 CPUs); the report's Section 7 lists what each step reproduced.

```bash
git clone --branch report-submission https://github.com/CVasilopoulos/aimo-bas.git && cd aimo-bas
python3 tools/verify_bundles.py
python3 -m unittest tests/test_offline.py tests/test_report_tools.py
```

Report tables and figures (needs network once, then [uv](https://docs.astral.sh/uv/); each script declares its dependencies inline):

```bash
python3 tools/fetch_inputs.py
python3 research/getting-started/scripts/import_hf_dataset.py --dataset aimo-interp/val-sample --revision 1ae454ec1fad9727084eda8f9f3c9ae2239b21de --output-dir data/val-sample
uv run tools/make_cases.py research/hf/augmented-sample-math-agg.parquet data/math-deepseek
tools/report/run_all.sh
python3 tools/report/ablation_tables.py runs/kaggle_ablate/deepseek-math-3072-full-n52.jsonl
python3 tools/report/whitebox_ablation.py
git status --short report/
```

Every table under `report/tables/` regenerates byte-identically. The figures are redrawn pixel-identically (one figure has 91 pixels one intensity level apart); their PNG bytes differ, so `git status` lists them as modified.

Local contract test in the organisers' runtime image (CPU is enough; the `cpu-smoke` profile applies automatically without CUDA):

```bash
docker pull ghcr.io/aimo-interp/aimo-codabench-runtime:torch2.12.1-cuda13.2-cudnn9-transformers5.13.0-sklearn1.8.0-kernels0.15.2
tools/fetch_probe_models.sh
MEMORY=6g CPUS=6 tools/run_local.sh submissions/aimo-bas-v1-small.zip data/smoke-qwen small
MEMORY=6g CPUS=6 tools/run_local.sh submissions/aimo-bas-v1-main.zip data/smoke-qwen main
```

Both runs end with coverage 1.0 and 0 invalid predictions in about two minutes; the smoke accuracy is meaningless (tiny proxy models, 24-token budget, labels copied from other models).

GPU ablation behind Table 3 (two T4 GPUs on Kaggle, about 7,000 s per chunk of 13 cases; not re-run for this release):

```bash
python3 tools/kaggle_ablation.py --plans full
python3 tools/report/ablation_tables.py runs/kaggle_ablate/deepseek-math-3072-full-n52.jsonl
```

## Not included

The organisers' Hugging Face datasets and repositories (fetched by `tools/fetch_inputs.py` at pinned revisions), model weights, and the hidden Codabench sets.

## License

MIT, see `LICENSE`.

AI-assisted, human-directed.
