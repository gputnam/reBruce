#!/usr/bin/env python3
"""Validate the Nieves CCQE cross-section port against GENIE.

The sBruce multisigmaTree stores, per event, the GENIE/nusystematics
ZExpPCAWeighter_SBN_v3_MvA b1..b4 dial weights, normalized to the sigma=0
(MINERvA CV) point. Each sigma=+-v entry is a pure ratio of Nieves cross
sections between the PCA-shifted z-expansion coefficients and the MINERvA
CV set -- exactly what fakedata.nieves computes. This script reproduces the
PCA construction (Nature_614_102522 covariance, verbatim from
nusystematics/systproviders/ZExpPCAWeighter_tool.cc) and compares
event-by-event.

NOTE the eigenvector sign convention: numpy's eigh returns the OPPOSITE
sign to Eigen's SelfAdjointEigenSolver for this matrix (both are valid;
the sign of an eigenvector is arbitrary). The comparison therefore matches
our +v against the stored -v.

Result on ICARUSRun2_SpringMCOverlay_rewgt_2 (numu CC QE events), all four
dials at v = +-1, +-2, +-3:
    mean(w_computed / w_stored) = 1.0000, RMS ~ 0.0003-0.002
which validates the tensor contraction, RPA, Coulomb treatment, form
factors, AND the hit-nucleon-radius marginalization (r is not stored in
sBruce; see MISSING_INFO.md). Antineutrino agreement is looser (bulk ~2%,
rare high-Q2 RPA-zero-crossing outliers) -- see MISSING_INFO.md item 1.

Usage: ./venv/bin/python validation/validate_nieves.py [sbruce_file]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fakedata.sbruce import SBruceFile                      # noqa: E402
from fakedata.zexp import ZExpAxialFF                       # noqa: E402
from fakedata.nieves import NievesQEXSec                    # noqa: E402
from fakedata.calculators.qe_zexp import (                  # noqa: E402
    qe_numu_prefsi_mask, qe_kinematics, QE_BRANCHES, GA_CV)

DEFAULT_FILE = ("/Users/gputnam/Work/osc/sbn-rewgted-20-sBruce/sbn-rewgted-20/"
                "ICARUSRun2_SpringMCOverlay_rewgt_2_sbruce.root")

# verbatim from nusystematics ZExpPCAWeighter_tool.cc (Nature_614_102522)
A_CV = np.array([1.50, -1.2, -0.1, 0.2])
COV = np.array([[0.0961, 0.002604, -0.54777, 0.403],
                [0.002604, 0.49, -0.4256, -1.365],
                [-0.54777, -0.4256, 3.61, -1.2825],
                [0.403, -1.365, -1.2825, 6.25]])


def main(path=DEFAULT_FILE):
    evals, evecs = np.linalg.eigh(COV)
    pca = evecs * np.sqrt(evals)

    sb = SBruceFile(path)
    a = sb.arrays(QE_BRANCHES)
    mask = qe_numu_prefsi_mask(a)
    print(f"{os.path.basename(path)}: {mask.sum()} numu CC QE events with pre-FSI kinematics")
    enu, p_lep, p_nf = qe_kinematics(a, mask)

    xs = NievesQEXSec()
    fa_cv = ZExpAxialFF(a=list(A_CV), t0=-0.75, tcut=0.1764, ga=GA_CV)

    print(f"{'dial':>5} {'v':>3} {'n':>6} {'mean':>8} {'rms':>8} {'p1':>8} {'p99':>8}")
    worst = 0.0
    for bi in range(4):
        for v in [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]:
            fa_shift = ZExpAxialFF(a=list(A_CV + v * pca[:, bi]),
                                   t0=-0.75, tcut=0.1764, ga=GA_CV)
            w = xs.weight_ratio(enu, p_lep, p_nf, fa_shift, fa_cv)
            # numpy/Eigen eigenvector sign convention: compare against -v
            s = sb.multisigma_at_sigma(
                f"multisigma_ZExpPCAWeighter_SBN_v3_MvA_b{bi+1}", -v)[mask]
            ok = np.isfinite(s) & (s > 0)
            r = w[ok] / s[ok]
            print(f"  b{bi+1} {v:+3.0f} {ok.sum():>6} {r.mean():8.4f} {r.std():8.4f} "
                  f"{np.percentile(r, 1):8.4f} {np.percentile(r, 99):8.4f}")
            worst = max(worst, abs(r.mean() - 1.0), r.std())
    print(f"\nworst |mean-1| or rms: {worst:.5f}")
    ok = worst < 0.01
    print("VALIDATION " + ("PASSED" if ok else "FAILED") + " (threshold 1%)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
