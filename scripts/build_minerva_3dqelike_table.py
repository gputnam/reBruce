#!/usr/bin/env python3
"""Build data/minerva_3dqelike_bnb.csv from the MvA_LE_ME analysis output.

Extracts the runtime columns of the MINERvA LE/ME 3D QE-like ->AR23 weight
table (bin indices/edges and the BNB-extrapolated weight) from

    MvA_LE_ME/results/weights_ptpzsumtp.csv   (+ weights_meta.json)

into the self-contained data/ table the `minerva_3dqelike` calculator reads,
with a provenance header in the style of the other data/*.csv files.

Usage:  ./venv/bin/python scripts/build_minerva_3dqelike_table.py \
            [--src MvA_LE_ME] [--out data/minerva_3dqelike_bnb.csv]
"""

import argparse
import csv
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# columns copied through; w_BNB is renamed to the conventional "weight"
COLUMNS = ["ipz", "ipt", "itp",
           "pz_lo", "pz_hi", "pt_lo", "pt_hi", "tp_lo", "tp_hi",
           "w_LE", "w_ME", "w_BNB_raw", "weight", "w_BNB_err", "excluded"]

HEADER = """\
# MINERvA LE/ME triple-differential QE-like cross section, extrapolated to the
# BNB flux peak.  d^3 sigma / dp_z dp_T dSumT_p on hydrocarbon, LE and ME NuMI
# beams; D. Ruterbories et al., arXiv:2606.00745 (see also PRL 129 021803).
#
# Built by scripts/build_minerva_3dqelike_table.py from the standalone
# analysis in MvA_LE_ME/ (results/weights_ptpzsumtp.csv + weights_meta.json);
# see MvA_LE_ME/README.md for the full derivation.
#
# Binning ({n_pz} x {n_pt} x {n_tp} = {n_bins} bins, dense, ordered by
# (ipz, ipt, itp)):
#   p_z(mu)  [GeV/c] : {pz_edges}
#   p_T(mu)  [GeV/c] : {pt_edges}
#   SumT_p   [GeV]   : {tp_edges}
#
# w_LE = sigma_data_LE / sigma_AR23_LE  and  w_ME likewise, per bin, both
# bin-integrated cm^2 / CH-nucleon.  sigma_AR23 is GENIE v3_06_00 with the
# AR23_20i_00_000 tune (gevgen, numu on C12 + H1, 8M + 0.5M CC events per
# flux), normalized per CH nucleon as
#     sigma_i = [ <sigma_C> f_C,i + <sigma_H> f_H,i ] / 13.
# Unmeasured / no-MC bins are flagged `excluded` and carry weight 1.
#
# Linear per-bin extrapolation in effective neutrino energy (each beam a
# point at its flux peak) to the BNB peak:
#   E_LE = {E_LE} GeV, E_ME = {E_ME} GeV, E_BNB = {E_BNB} GeV
#   w_BNB_raw = {coeff_LE:.6f} * w_LE + {coeff_ME:.6f} * w_ME
#   weight    = clip(w_BNB_raw, {clip_lo}, {clip_hi})
# The extrapolation reaches far outside the [{E_LE}, {E_ME}] GeV lever arm: it
# is a linear trend estimate, not a measurement.  {n_lo} of {n_bins} bins
# extrapolate below {clip_lo} (clipped to {clip_lo}, i.e. the event is killed)
# and {n_hi} above {clip_hi}.  w_LE / w_ME / w_BNB_raw are stored UNCLIPPED.
#
# The measurement is on CH; applying the weights to argon assumes the
# data/model discrepancy transfers.  How the calculator maps BNB events into
# this binning (the E_nu-relative p_z scaling) is documented in the top-level
# README under `minerva_3dqelike`.
#
# w_BNB_err: data total uncertainty (covariance diagonal) (+) MC stat,
# propagated linearly through the extrapolation, LE/ME treated uncorrelated.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=os.path.join(REPO, "MvA_LE_ME"),
                    help="MvA_LE_ME analysis directory")
    ap.add_argument("--out", default=os.path.join(REPO, "data",
                                                  "minerva_3dqelike_bnb.csv"))
    args = ap.parse_args()

    with open(os.path.join(args.src, "results", "weights_ptpzsumtp.csv")) as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(args.src, "results", "weights_meta.json")) as f:
        meta = json.load(f)

    # the grid must be dense and in (ipz, ipt, itp) order -- the calculator
    # indexes the flattened weight array directly
    idx = [(int(r["ipz"]), int(r["ipt"]), int(r["itp"])) for r in rows]
    n_pz, n_pt, n_tp = (max(i[k] for i in idx) + 1 for k in range(3))
    assert idx == sorted(idx), "source CSV is not sorted by (ipz, ipt, itp)"
    assert len(set(idx)) == len(rows) == n_pz * n_pt * n_tp, "grid not dense"

    def edges(lo, hi):
        vals = sorted({float(r[lo]) for r in rows} | {float(r[hi]) for r in rows})
        return "[" + ", ".join(f"{v:g}" for v in vals) + "]"

    header = HEADER.format(
        n_pz=n_pz, n_pt=n_pt, n_tp=n_tp, n_bins=len(rows),
        pz_edges=edges("pz_lo", "pz_hi"),
        pt_edges=edges("pt_lo", "pt_hi"),
        tp_edges=edges("tp_lo", "tp_hi"),
        E_LE=meta["E_LE"], E_ME=meta["E_ME"], E_BNB=meta["E_BNB"],
        coeff_LE=meta["coeff_LE"], coeff_ME=meta["coeff_ME"],
        clip_lo=meta["weight_clip"][0], clip_hi=meta["weight_clip"][1],
        n_lo=meta["n_clipped_low"], n_hi=meta["n_clipped_high"],
    )

    with open(args.out, "w", newline="") as f:
        f.write(header)
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        for r in rows:
            out = {c: r[c] for c in COLUMNS if c != "weight"}
            out["weight"] = r["w_BNB"]
            w.writerow(out)

    print(f"wrote {args.out}: {len(rows)} bins "
          f"({n_pz} x {n_pt} x {n_tp}), {meta['n_excluded']} excluded")


if __name__ == "__main__":
    main()
