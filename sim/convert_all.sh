#!/bin/bash
# Convert the campaign's ROOT output into ../data/{TIER}_{SCENE}.npz
# Run from sim/ after campaign.sh.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUTD=$(cd "$HERE/.." && pwd)/data
mkdir -p "$OUTD"
for TIER in A B; do
  for D in "$HERE"/work/$TIER/*/; do
    [ -d "$D" ] || continue
    SCENE=$(basename "$D")
    OUT=$OUTD/${TIER}_${SCENE}.npz
    [ -s "$OUT" ] && { echo "skip $OUT (exists)"; continue; }
    echo "=== $TIER $SCENE ==="
    python3 "$HERE/convert.py" --glob "$D/root_file/job_*.root" --out "$OUT" --workers 48
  done
done
echo "convert done -> $OUTD"
