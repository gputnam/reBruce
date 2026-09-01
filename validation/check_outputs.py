#!/usr/bin/env python3
"""Post-production checks on the fakedata output files.

For every output/*_fakedata.root:
  - fakedataTree entry count matches SelectedEvents / multisigmaTree
  - every dial is a well-formed two-knot grid: sigma == [0, 1] and the cv
    knot identically 1.0, with the fake-data weight on the ps1 knot
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

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fakedata.output import CV_WEIGHT, SIGMA_KNOTS, dial_name   # noqa: E402
from fakedata.tercile import WEIGHT_CLIP   # noqa: E402

XSEC_CALCS = ["fdwgt_ub_cc1p0pi_dpt_dat", "fdwgt_ub_cc2p0pi_delta_pt",
              "fdwgt_ub_ccpi_pionmomentum", "fdwgt_t2k_nc1pi_p_costh"]
# nominal-only calculators (no _loW/_midW/_hiW branches, so they are not in
# XSEC_CALCS, which drives the W-mode closure check)
ALL_CALCS = ["fdwgt_mec_bdt", "fdwgt_qe_zexp_mva_to_lqcd",
             "fdwgt_pi_fsi_ha2025",
             "fdwgt_jaesung_lowq2_pi_enhancement_postfsi",
             "fdwgt_jaesung_lowq2_pi_enhancement_prefsi",
             ] + XSEC_CALCS + ["fdwgt_minerva_3dqelike_bnb",
                            "fdwgt_minerva_3dqelike_bnb_pzmarg"]


def _short(branch):
    """5-char column label for the coverage table."""
    parts = branch.split("_")
    # branches that differ only in their last part are labelled by it
    return (parts[-1][:5]
            if parts[-1] in ("pzmarg", "bnb", "postfsi", "prefsi")
            else parts[1][:4])


def _ps1(fd, branch):
    """The ps1 (sigma = +1) knot of a dial, checking the grid is well formed.

    Returns (weights, ok). Works for both writers: uproot's leaf arrays and
    PyROOT's std::vector<double> both read back as one row per entry.
    """
    name = dial_name(branch)
    knots = np.asarray(ak.to_numpy(fd[name].array()), dtype=np.float64)
    sigma = np.asarray(ak.to_numpy(fd[name + "_sigma"].array()),
                       dtype=np.float64)
    ok = (knots.shape[1] == len(SIGMA_KNOTS)
          and bool(np.all(sigma == np.asarray(SIGMA_KNOTS)))
          and bool(np.all(knots[:, 0] == CV_WEIGHT)))
    return knots[:, 1], ok


def check_file(fn, tol_clip=1e-3):
    f = uproot.open(fn)
    fd = f["fakedataTree"]
    n = fd.num_entries
    ok = (n == f["SelectedEvents"].num_entries)
    cv = f["SelectedEvents"]["cvwgt"].array(library="np").astype(np.float64)

    fracs = {}
    for b in ALL_CALCS:
        w, grid_ok = _ps1(fd, b)
        ok &= grid_ok
        ok &= bool(np.all(np.isfinite(w)) and np.all(w >= 0))
        fracs[b] = 100.0 * np.mean(w != 1.0)

    n_clip = 0
    max_resid = 0.0
    for b in XSEC_CALCS:
        nom = np.sum(cv * _ps1(fd, b)[0])
        for s in ["_loW", "_midW", "_hiW"]:
            w, grid_ok = _ps1(fd, b + s)
            ok &= grid_ok
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
          " ".join(f"{_short(b):>5}" for b in ALL_CALCS) +
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
