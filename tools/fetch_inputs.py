import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co"
PARQUET = "data/validation-00000-of-00001.parquet"

DATASETS = {
    "val-sample": ("1ae454ec1fad9727084eda8f9f3c9ae2239b21de", "3919cb3d951bb360a120be3397e926f48b7819bcd97e12cfee4c01e76618560f"),
    "augmented-sample-math-agg": ("f972ced0705096f8d7ca7fac30825900b8b7fb6a", "b785750259a3f4524cbaf988f51660197cbdfca19d8f45cc35347d4ef29aa4db"),
    "augmented-sample-math-agg-filtered": ("0607cd06cbd789dd33655ff718bbfae0d3edba7c", "b8a485868ce5a14d0c961e009e30bc61f850ad2d2224a01ae1ada401fb73ed3d"),
    "augmented-sample-math-full-filtered": ("f09e41ed0648b0ff7cac220c6ce9150a36e2e8b0", "3993d9d9b6035c23a1ad217eb2a5dec033e8506748e3a97236b6fa6b33bd2a02"),
    "aimo-interp-challenge-sample-full": ("c0ffb7e678294cc8819d50d979ba28ba49bd02d0", "2989ab6a018f2779a821209f414bd0049c9ef1c2488f589ff2d3bf4ed16942ff"),
}

MODEL_CONFIGS = {
    "Qwen/Qwen3.5-4B": ("851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"),
    "Skywork/Skywork-OR1-Math-7B": ("fcef970f1d10682f13c182b280146a194ed57d1e", "6673df32586bddb5814977e6dff3cf301a48d10a48f485c90e7709e33ab79a95"),
    "allenai/Olmo-3-7B-Think": ("d97e442d7cc678210054dbcc9b440894d62c89a4", "c395e580e9e181b271048bbef57d0316ea7f2aea17761340c7cc6ccec6241119"),
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": ("6e8885a6ff5c1dc5201574c8fd700323f23c25fa", "1b1cbbba841c180e94192f7a25d2d97e21498fc1a0a0b4785942158af5eee183"),
    "openai/gpt-oss-120b": ("b5c939de8f754692c1647ca79fbf85e8c1e70f8a", "933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566"),
}

REPOSITORIES = {
    "getting-started": ("https://github.com/aimo-interp/getting-started", "995f9923a680a157c9239e08e4ef0fa0fa53bb58"),
    "baselines": ("https://github.com/aimo-interp/baselines", "7e8839966750059b6b1d247a12ab56552c79b342"),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "aimo-bas-fetch"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def place(target, data, expected):
    actual = sha256(data)
    if actual != expected:
        raise SystemExit(f"{target.name}: sha256 {actual} does not match the pinned {expected}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print("ok", target.relative_to(ROOT), actual[:12])


def fetch_datasets(force):
    for name, (revision, expected) in DATASETS.items():
        target = ROOT / "research" / "hf" / f"{name}.parquet"
        if target.is_file() and not force and sha256(target.read_bytes()) == expected:
            print("present", target.relative_to(ROOT))
        else:
            place(target, download(f"{HF}/datasets/aimo-interp/{name}/resolve/{revision}/{PARQUET}"), expected)
        info = {"id": f"aimo-interp/{name}", "sha": revision, "file": PARQUET, "sha256": expected}
        (target.parent / f"{name}.info.json").write_text(json.dumps(info, indent=1) + "\n")


def fetch_model_configs(force):
    for model_id, (revision, expected) in MODEL_CONFIGS.items():
        target = ROOT / "research" / "modelcfg" / (model_id.replace("/", "_") + ".config.json")
        if target.is_file() and not force and sha256(target.read_bytes()) == expected:
            print("present", target.relative_to(ROOT))
            continue
        place(target, download(f"{HF}/{model_id}/resolve/{revision}/config.json"), expected)


def checkout_repositories():
    if shutil.which("git") is None:
        raise SystemExit("git is required to check out the organisers' repositories")
    for name, (url, commit) in REPOSITORIES.items():
        target = ROOT / "research" / name
        if not (target / ".git").is_dir():
            if target.exists():
                raise SystemExit(f"{target} exists but is not a git checkout; move it away first")
            subprocess.run(["git", "clone", "--quiet", url, str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "-c", "advice.detachedHead=false", "checkout", "--quiet", commit], check=True)
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        if head != commit:
            raise SystemExit(f"{name}: HEAD {head} is not the pinned {commit}")
        print("ok", target.relative_to(ROOT), commit[:12])


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-repositories", action="store_true")
    args = parser.parse_args(argv)
    fetch_datasets(args.force)
    fetch_model_configs(args.force)
    if not args.skip_repositories:
        checkout_repositories()


if __name__ == "__main__":
    main(sys.argv[1:])
