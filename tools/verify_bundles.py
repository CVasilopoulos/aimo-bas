import contextlib
import csv
import hashlib
import io
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build import build

MANIFESTS = [ROOT / "report" / "tables" / "submissions.csv", ROOT / "report" / "tables" / "submissions_v8.csv"]
MARKERS = {"small.txt", "profile.txt"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(paths=MANIFESTS):
    expected = {}
    for path in paths:
        with Path(path).open(newline="") as stream:
            for row in csv.DictReader(stream):
                previous = expected.setdefault(row["bundle"], row["sha256"])
                if previous != row["sha256"]:
                    raise SystemExit(f"{row['bundle']} has two different hashes in the manifests")
    return expected


def rebuild_from_contents(bundle, workdir):
    source = workdir / "contents"
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(source)
    small = (source / "small.txt").is_file()
    profile = (source / "profile.txt").read_text(encoding="utf-8").strip() if (source / "profile.txt").is_file() else ""
    target = workdir / bundle.name
    with contextlib.redirect_stdout(io.StringIO()):
        build(source, target, small, profile)
    return target, small, profile


def compare_with_source(bundle, solution):
    report = {}
    with zipfile.ZipFile(bundle) as archive:
        for name in archive.namelist():
            if name.casefold() in MARKERS:
                continue
            candidate = solution / name
            if not candidate.is_file():
                report[name] = "not in solution/"
            elif candidate.read_bytes() == archive.read(name):
                report[name] = "identical"
            else:
                report[name] = "differs"
    return report


def verify(bundle, expected_sha, solution):
    actual = sha256(bundle)
    with tempfile.TemporaryDirectory() as tmp:
        rebuilt, small, profile = rebuild_from_contents(bundle, Path(tmp))
        rebuilt_sha = sha256(rebuilt)
    members = compare_with_source(bundle, solution)
    return {
        "bundle": bundle.name,
        "track": "small" if small else "main",
        "profile": profile or "gpu (default)",
        "sha256_matches_manifest": actual == expected_sha,
        "rebuild_is_byte_identical": rebuilt_sha == actual,
        "members_differing_from_solution": sorted(n for n, s in members.items() if s != "identical"),
        "member_status": members,
    }


def main(argv):
    solution = Path(argv[0]) if argv else ROOT / "solution"
    expected = manifest()
    failures = 0
    print("| bundle | track | profile | SHA-256 matches manifest | rebuild byte-identical | files differing from solution/ |")
    print("|---|---|---|---|---|---|")
    for name in sorted(expected):
        bundle = ROOT / "submissions" / name
        if not bundle.is_file():
            print(f"| {name} | - | - | missing | - | - |")
            failures += 1
            continue
        result = verify(bundle, expected[name], solution)
        detail = ", ".join(f"{n} ({result['member_status'][n]})" for n in result["members_differing_from_solution"]) or "none"
        print(
            f"| {name} | {result['track']} | {result['profile']} | {'yes' if result['sha256_matches_manifest'] else 'NO'} | "
            f"{'yes' if result['rebuild_is_byte_identical'] else 'NO'} | {detail} |"
        )
        if not (result["sha256_matches_manifest"] and result["rebuild_is_byte_identical"]):
            failures += 1
    if failures:
        raise SystemExit(f"{failures} bundle(s) failed verification")


if __name__ == "__main__":
    main(sys.argv[1:])
