import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MODEL_REVISION = "6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
PLANS = {
    "full": "",
    "orig": "orig,orig,orig,orig,orig,orig",
    "pert": "typos,distract,rephrase,typos,distract,rephrase",
}


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def download_model():
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL, revision=MODEL_REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
    print("model at", path, flush=True)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "data/math-deepseek/input/cases.jsonl")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/math-deepseek/reference/labels.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/kaggle_ablate")
    parser.add_argument("--limit", type=int, default=52)
    parser.add_argument("--chunk", type=int, default=13)
    parser.add_argument("--budget-s", type=float, default=7000)
    parser.add_argument("--think-tokens", type=int, default=3072)
    parser.add_argument("--plans", default="full,orig,pert")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    if not args.skip_download:
        download_model()
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()][: args.limit]
    parts = args.out_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for tag in args.plans.split(","):
        variants = PLANS[tag]
        out = args.out_dir / f"deepseek-math-{args.think_tokens}-{tag}-n{args.limit}.jsonl"
        for start in range(0, len(cases), args.chunk):
            part = parts / f"part_{tag}_{start}.jsonl"
            part.write_text("".join(json.dumps(case) + "\n" for case in cases[start : start + args.chunk]))
            cmd = [
                sys.executable,
                str(ROOT / "tools/ablate.py"),
                "--cases", str(part),
                "--labels", str(args.labels),
                "--out", str(out),
                "--budget-s", str(args.budget_s),
                "--think-tokens", str(args.think_tokens),
            ]
            if variants:
                cmd += ["--variants", variants]
            began = time.time()
            code = run(cmd)
            print(f"[{tag}] chunk {start}: rc={code} {time.time() - began:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
