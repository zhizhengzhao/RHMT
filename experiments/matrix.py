"""Experiment matrix: each entry is the argv tail for run_cell.py."""
OBJ = [f"u_{m}_{bg}" for bg in ("sio2", "concrete")
       for m in ("water", "pb", "al")]   # 6 single-U scenes
ALL_METHODS = ["poca", "asr", "mlsem", "asr_p", "rht_3p", "rht_s", "rht_6p"]
CAPS = [0, 50000, 15000]    # 0 = full exposure


def cells(mode="main"):
    out = []
    for sc in OBJ:
        for m in ALL_METHODS:
            for cap in CAPS:
                out.append(["--method", m, "--scene", sc, "--cap", str(cap)])
    return out
