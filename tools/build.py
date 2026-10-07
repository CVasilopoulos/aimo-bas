import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = (2026, 9, 16, 0, 0, 0)


def members(source):
    for path in sorted(source.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts or path.suffix == ".pyc" or path.name.startswith("."):
            continue
        yield path, path.relative_to(source).as_posix()


def build(source, destination, small, profile):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path, name in members(source):
            if name.casefold() in {"small.txt", "profile.txt"}:
                continue
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, path.read_bytes())
        if small:
            info = zipfile.ZipInfo("small.txt", date_time=FIXED_TIME)
            info.external_attr = 0o644 << 16
            bundle.writestr(info, "small model track\n")
        if profile:
            info = zipfile.ZipInfo("profile.txt", date_time=FIXED_TIME)
            info.external_attr = 0o644 << 16
            bundle.writestr(info, profile + "\n")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(destination.name, destination.stat().st_size, "bytes", "sha256", digest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "solution")
    parser.add_argument("--out", type=Path, default=ROOT / "submissions")
    parser.add_argument("--name", default="aimo-bas-v1")
    parser.add_argument("--profile", default="")
    args = parser.parse_args()
    build(args.source, args.out / f"{args.name}-main.zip", False, args.profile)
    build(args.source, args.out / f"{args.name}-small.zip", True, args.profile)


if __name__ == "__main__":
    main()
