#!/usr/bin/env python3
"""Extract the hA2018 -> hA2025 pion-FSI reweight to CSV.

Produces two CSVs in data/ from the binary ROOT files in hA_TGraphs_2D/
(which stay OUT of the git repository):

1. data/ha_pion_fsi_tgraphs.csv -- the raw TGraph2D points (archival
   provenance; read with uproot, no ROOT needed).

2. data/ha_pion_fsi_weights_A40.csv -- the fate fractions and weights
   tabulated on a dense KE grid at A = TARGET_A, evaluated with the
   REFERENCE implementation (hA_TGraphs_2D/hA2025Reweighter.py, i.e.
   ROOT's own TGraph2D::Interpolate Delaunay interpolation). This is the
   table the pi_fsi_ha2025 calculator uses at run time: ROOT's Delaunay
   triangulation is not bit-reproducible outside ROOT (diagonal choices on
   the coarse A grid differ between implementations), so the information
   is extracted as ROOT evaluates it. Between the 0.5-MeV grid points the
   curve is piecewise linear, so linear interpolation reproduces ROOT to
   better than the graph's own granularity.

Weight guard: where the hA2018 fraction is < FRAC_MIN (exactly-zero or
numerically-zero from Delaunay roundoff, e.g. pipro below threshold) the
weight is set to 1 -- the reference implementation's f18 != 0 test lets
1e-18 roundoff fractions through and produces O(1e15) weights.

Requires PyROOT (Homebrew ROOT works):
    cd <repo> && python3 scripts/extract_ha_tgraphs.py
Run whenever the TGraph inputs change; commit the CSVs.
"""

import csv
import os
import sys

import numpy as np

TARGET_A = 40.0
KE_GRID = np.round(np.arange(1.0, 999.0 + 0.25, 0.5), 2)
FRAC_MIN = 1e-6
FATES = ("cex", "abs", "inel", "pipro")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TG_DIR = os.path.join(REPO, "hA_TGraphs_2D")


def write_raw_points():
    import uproot

    files = {
        "2018": ("TGraphs_2018.root",
                 {"TfracPipA_CEx": "cex", "TfracPipA_Abs": "abs",
                  "TfracPipA_Inelas": "inel", "TfracPipA_PiPro": "pipro"}),
        "2025": ("TGraphs_2025.root",
                 {"TPipA_CEx": "cex", "TPipA_Abs": "abs",
                  "TPipA_Inelas": "inel", "TPipA_PiPro": "pipro",
                  "TPipA_Tot": "tot"}),
    }
    out = os.path.join(REPO, "data", "ha_pion_fsi_tgraphs.csv")
    with open(out, "w", newline="") as fout:
        fout.write(
            "# hA2018/hA2025 INTRANUKE pi+ FSI TGraph2D points (archival), extracted\n"
            "# from hA_TGraphs_2D/TGraphs_{2018,2025}.root by scripts/extract_ha_tgraphs.py.\n"
            "# model=2018: value = fate fraction(A, KE[MeV]).\n"
            "# model=2025: value = log(cross section).\n"
            "# The runtime table is ha_pion_fsi_weights_A40.csv.\n")
        w = csv.writer(fout)
        w.writerow(["model", "graph", "fate", "A", "KE", "value"])
        n = 0
        for model, (path, graphs) in files.items():
            f = uproot.open(os.path.join(TG_DIR, path))
            for gname, fate in graphs.items():
                g = f[gname]
                for xi, yi, zi in zip(g.member("fX"), g.member("fY"),
                                      g.member("fZ")):
                    w.writerow([model, gname, fate, repr(float(xi)),
                                repr(float(yi)), repr(float(zi))])
                    n += 1
    print(f"wrote {n} raw points to {out}")


def write_weight_table():
    sys.path.insert(0, TG_DIR)
    from hA2025Reweighter import hA2025Reweighter

    rw = hA2025Reweighter(os.path.join(TG_DIR, "TGraphs_2018.root"),
                          os.path.join(TG_DIR, "TGraphs_2025.root"))
    out = os.path.join(REPO, "data", "ha_pion_fsi_weights_A40.csv")
    with open(out, "w", newline="") as fout:
        fout.write(
            "# hA2018 -> hA2025 pi+ FSI fate-fraction weights at A = %g,\n"
            "# tabulated from hA_TGraphs_2D/*.root with ROOT TGraph2D::Interpolate\n"
            "# via the reference hA2025Reweighter (scripts/extract_ha_tgraphs.py).\n"
            "# frac2018/frac2025: renormalized fate fractions; weight = f25/f18,\n"
            "# set to 1 where f18 < %g (unmeasured / roundoff region).\n"
            "# KE in MeV, clamped by the calculator to [%g, %g].\n"
            % (TARGET_A, FRAC_MIN, KE_GRID[0], KE_GRID[-1]))
        w = csv.writer(fout)
        w.writerow(["KE", "fate", "frac2018", "frac2025", "weight"])
        for ke in KE_GRID:
            f18 = rw.frac2018(ke, TARGET_A)
            f25 = rw.frac2025(ke, TARGET_A)
            for fate in FATES:
                wt = f25[fate] / f18[fate] if f18[fate] >= FRAC_MIN else 1.0
                w.writerow([repr(float(ke)), fate, repr(f18[fate]),
                            repr(f25[fate]), repr(wt)])
    print(f"wrote {len(KE_GRID) * len(FATES)} weight rows to {out}")


if __name__ == "__main__":
    write_raw_points()
    write_weight_table()
