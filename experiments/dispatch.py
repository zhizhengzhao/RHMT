"""Multi-GPU dispatcher over the experiment matrix.

Usage: python experiments/dispatch.py [gpu,gpu,...]   (default 1..7)
"""
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))
from matrix import cells
from mdmt.scenes import SCENES


def tag_of(job):
    d = dict(zip(job[::2], job[1::2]))
    return (f"{d['--method']}_{d['--scene']}_cap{int(d.get('--cap', 0))}"
            f"_s{int(d.get('--seed', 7))}")


def main(gpus):
    jobs = [j for j in cells()
            if not os.path.exists(f"results/cells/{tag_of(j)}.json")]
    gpuq = queue.Queue()
    for g in gpus:
        gpuq.put(g)
    lock = threading.Lock()
    done = [0]

    def run(job):
        g = gpuq.get()
        try:
            r = subprocess.run(
                ["python3", "experiments/run_cell.py", "--device", f"cuda:{g}"] + job,
                capture_output=True, text=True)
            with lock:
                done[0] += 1
                line = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
                if r.returncode != 0:
                    line = "ERR: " + (r.stderr.strip().splitlines() or [""])[-1][:110]
                print(f"[{done[0]}/{len(jobs)}] {' '.join(job)} -> {line}", flush=True)
        finally:
            gpuq.put(g)

    # one calibration per (method, background) first, so cells reuse the cache
    seen, warm = set(), []
    for j in jobs:
        d = dict(zip(j[::2], j[1::2]))
        key = (d["--method"], SCENES[d["--scene"]][0])
        if key not in seen:
            seen.add(key)
            warm.append(j)
    print(f"=== warming {len(warm)} calibrations ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(run, warm))
    done[0] = 0
    print(f"=== {len(jobs)} cells on GPUs {gpus} ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(run, jobs))
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    gpus = ([int(g) for g in sys.argv[1].split(",")]
            if len(sys.argv) > 1 else [1, 2, 3, 4, 5, 6, 7])
    main(gpus)
