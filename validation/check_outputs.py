#!/usr/bin/env python3
"""Post-production checks on the fakedata output files.

For every output/*_fakedata.root:
  - fakedataTree entry count matches SelectedEvents / multisigmaTree
  - all weights finite and >= 0
  - per-calculator coverage (fraction of events with weight != 1)
  - W-mode closure: sum(cvwgt * w_mode) == sum(cvwgt * w_nominal) exactly,
    UNLESS the mode clipped tercile weights to the WEIGHT_CLIP range
    (default [0, 10]), in which case the total differs from nominal by the
    clipped amount -- clipping is reported and the residual must be small.

Usage: ./venv/bin/python validation/check_outputs.py [glob]
"""

import glob
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fakedata.tercile import WEIGHT_CLIP   # noqa: E402

XSEC_CALCS = ["wgt_ub_cc1p0pi_dpt_dat", "wgt_ub_cc2p0pi_delta_pt",
              "wgt_ub_ccpi_pionmomentum", "wgt_t2k_nc1pi_p_costh"]
ALL_CALCS = ["wgt_mec_bdt", "wgt_qe_zexp_mva_to_lqcd"] + XSEC_CALCS


def check_file(fn, tol_clip=1e-3):
    f = uproot.open(fn)
    fd = f["fakedataTree"]
    n = fd.num_entries
    ok = (n == f["SelectedEvents"].num_entries)
    cv = f["SelectedEvents"]["cvwgt"].array(library="np").astype(np.float64)

    fracs = {}
    for b in ALL_CALCS:
        w = fd[b].array(library="np")
        ok &= bool(np.all(np.isfinite(w)) and np.all(w >= 0))
        fracs[b] = 100.0 * np.mean(w != 1.0)

    n_clip = 0
    max_resid = 0.0
    for b in XSEC_CALCS:
        nom = np.sum(cv * fd[b].array(library="np"))
        for s in ["_loW", "_midW", "_hiW"]:
            w = fd[b + s].array(library="np")
            ok &= bool(np.all(np.isfinite(w)) and np.all(w >= 0))
            tot = np.sum(cv * w)
            clipped = int(np.sum((w == WEIGHT_CLIP[0]) | (w == WEIGHT_CLIP[1])))
            n_clip += clipped
            resid = abs(tot - nom) / abs(nom)
            if clipped == 0:
                ok &= bool(np.isclose(tot, nom, rtol=1e-9))
            else:
                # clipping shifts the total by the clipped amount
                ok &= resid < tol_clip
            max_resid = max(max_resid, resid)

    return ok, n, fracs, n_clip, max_resid


def main(pattern="output/*_fakedata.root"):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no files match {pattern}")
        return 1
    print(f"{'file':44s} {'evts':>6} " +
          " ".join(f"{b.split('_')[1][:4]:>5}" for b in ALL_CALCS) +
          f" {'clip':>5} {'resid':>8}  status")
    n_bad = 0
    for fn in files:
        ok, n, fracs, n_clip, resid = check_file(fn)
        n_bad += not ok
        print(f"{os.path.basename(fn):44s} {n:>6} " +
              " ".join(f"{fracs[b]:5.1f}" for b in ALL_CALCS) +
              f" {n_clip:>5} {resid:8.1e}  " + ("OK" if ok else "FAIL"))
    print(f"\n{len(files)} files, {n_bad} failures")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
