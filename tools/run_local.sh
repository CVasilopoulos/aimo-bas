#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ZIP="$1"
DATA="$(cd "$2" && pwd)"
TRACK="$3"
NAME="${4:-aimo-run}"
LIMIT="${5:-3000}"
OFFICIAL=ghcr.io/aimo-interp/aimo-codabench-runtime:torch2.12.1-cuda13.2-cudnn9-transformers5.13.0-sklearn1.8.0-kernels0.15.2
REPLICA=aimo-cpu-runtime:torch2.12.1-tf5.13.0
if [ -z "${IMAGE:-}" ]; then
  if docker image inspect "$OFFICIAL" >/dev/null 2>&1; then IMAGE="$OFFICIAL"; else IMAGE="$REPLICA"; fi
fi
case "$NAME" in aimo-*) ;; *) echo "container name must start with aimo-"; exit 2;; esac
avail=$(free -g | awk '/^Mem:/{print $7}')
if [ "$avail" -lt 20 ]; then echo "only ${avail}G available; not starting"; exit 75; fi
if docker ps --format '{{.Names}}' | grep -q '^aimo-'; then echo "another aimo container is running; not starting"; exit 75; fi
mkdir -p "$HERE/runs" "$HERE/hf-cache"
WORK="$(mktemp -d "$HERE/runs/$(date +%Y%m%d-%H%M%S)-XXXX")"
mkdir -p "$WORK/submission" "$WORK/input/res" "$WORK/scores"
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP" "$WORK/submission"
cp -r "$DATA/reference" "$WORK/input/ref"
start=$(date +%s)
set +e
timeout --foreground "$LIMIT" docker run --rm --name "$NAME" --network none \
  --memory=10g --memory-swap=10g --cpus=8 \
  -e AIMO_BAS_PROFILE="${AIMO_BAS_PROFILE:-}" -e OMP_NUM_THREADS=8 \
  -v "$HERE/hf-cache:/app/data/huggingface:ro" \
  -v "$HERE/research/getting-started/components/ingestion_program:/app/program:ro" \
  -v "$DATA/input:/app/input_data:ro" \
  -v "$WORK/input/res:/app/output" \
  -v "$WORK/submission:/app/ingested_program" \
  "$IMAGE" python3 /app/program/ingestion.py /app/input_data /app/output /app/ingested_program --track "$TRACK" \
  > "$WORK/ingestion.log" 2>&1
code=$?
set -e
end=$(date +%s)
echo "image $IMAGE; ingestion exit $code in $((end-start)) s; log $WORK/ingestion.log"
docker rm -f "$NAME" >/dev/null 2>&1 || true
python3 "$HERE/research/getting-started/components/scoring_program/scoring.py" "$WORK/input" "$WORK/scores" --track "$TRACK"
cat "$WORK/scores/scores.json"
cp "$WORK/input/res/predictions.jsonl" "$WORK/predictions.jsonl" 2>/dev/null || true
echo "$WORK"
