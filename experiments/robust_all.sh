#!/bin/bash
cd "$(dirname "$0")/.." || exit 1   # repo root
rm -f results/robust/*.json
i=0
for sg in 0.05 0.5 1.0 2.0; do
  python experiments/robustness.py --sigma "$sg" --device "cuda:$i" &
  i=$((i + 1))
done
wait
python experiments/robust_table.py
echo ROBUST_DONE
