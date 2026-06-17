"""Collect results/robust/*.json into results/tables/robust{,_main}_body.tex + header."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.metrics import GROUPS, LABELS, MAIN, MAIN_GROUPS, hl, _decorate


def _body(rows, sigmas, labels, groups):
    """Mean-AUC vs sigma_pos body for the given row set; each column ranked within the set."""
    col = {sg: hl([rows.get(l, {}).get(sg, float("nan")) for l in labels]) for sg in sigmas}
    body = []
    for gname, glabels in groups:
        body.append(f"\\multicolumn{{{1 + len(sigmas)}}}{{@{{}}l}}{{\\textit{{{gname}}}}} \\\\")
        for l in glabels:
            i = labels.index(l)
            body.append(f"\\quad {_decorate(l)} & " + " & ".join(col[sg][i] for sg in sigmas) + r" \\")
    return "\n".join(body) + "\n"


def main():
    rows = {}
    for f in glob.glob("results/robust/*.json"):
        d = json.load(open(f))
        for label, v in d["auc"].items():
            rows.setdefault(label, {})[d["sigma"]] = v
    sigmas = sorted({s for v in rows.values() for s in v})

    os.makedirs("results/tables", exist_ok=True)
    open("results/tables/robust_body.tex", "w").write(_body(rows, sigmas, LABELS, GROUPS))
    open("results/tables/robust_main_body.tex", "w").write(_body(rows, sigmas, MAIN, MAIN_GROUPS))

    hdr = " & ".join(f"$\\sigma_{{\\mathrm{{pos}}}}{{=}}{sg:g}$" + (r"\,mm" if sg == sigmas[-1] else "")
                     for sg in sigmas)
    open("results/tables/robust_header.tex", "w").write("method & " + hdr + r" \\" + "\n")
    print(_body(rows, sigmas, MAIN, MAIN_GROUPS))


if __name__ == "__main__":
    main()
