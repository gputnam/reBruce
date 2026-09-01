"""MINERvA LE/ME 3D QE-like reweight: AR23 -> measurement, extrapolated to BNB.

Applies the per-bin weights of the MINERvA triple-differential QE-like
measurement (arXiv:2606.00745, d^3 sigma / dp_z dp_T dSumT_p on hydrocarbon,
LE and ME NuMI beams) after the per-bin linear extrapolation in effective
neutrino energy down to the BNB flux peak,

    w_BNB = 1.9808 * w_LE - 0.9808 * w_ME,   clipped to [0, 10]

as derived by the standalone MvA_LE_ME analysis (weight table in
data/minerva_3dqelike_bnb.csv, see data/README.md).

The p_z axis is treated as SCALED RELATIVE TO THE NEUTRINO ENERGY rather
than as absolute momentum: a measured bin edge p_z corresponds at BNB to
E_BNB * (p_z / E_beam), so an event's measurement-frame longitudinal
momentum is p_z / pz_scale with pz_scale = E_BNB / E_beam. Taken literally
the measured window 1.5 <= p_z < 4.5 GeV/c is empty at BNB energies (0 of
15203 events in SBNDMCCV_0); with the default LE scaling it maps to
0.323 - 0.969 GeV/c, the bulk of the BNB muon spectrum. p_T and SumT_p are
NOT scaled -- p_T is set by the Q^2 scale and SumT_p is hadronic recoil
energy, neither of which tracks E_nu the way the longitudinal boost does,
and both already populate the measured bins with ~0% overflow.

Two weight branches are produced:

  fdwgt_minerva_3dqelike_bnb        the full 3D (p_z, p_T, SumT_p) lookup;
                                    weight 1 outside the measured p_z window.
  fdwgt_minerva_3dqelike_bnb_pzmarg the same measurement MARGINALIZED over
                                    p_z and applied to every signal event in
                                    a (p_T, SumT_p) cell regardless of its
                                    p_z. For each cell the 5 p_z bins are
                                    averaged with the cvwgt-weighted p_z
                                    spectrum OF THE INPUT FILE itself,

                                      w_marg(j,k) = sum_i N_ijk w_ijk
                                                    / sum_i N_ijk

                                    with N_ijk the cvwgt sum of signal events
                                    of this file in 3D bin (i,j,k). This
                                    branch also DROPS the theta(mu,nu) < 20
                                    deg cut: for the 3D weight that cut is
                                    nearly redundant with the p_z window and
                                    costs ~0.2% of cvwgt, but once the weight
                                    is applied at any p_z it becomes the
                                    binding acceptance constraint. Both the
                                    spectrum N_ijk and the events the weight
                                    is applied to use this no-theta signal,
                                    so the closure below is internal to this
                                    branch. Every
                                    in-p_z-range bin contributes, including
                                    the 11 `excluded` bins at their
                                    conventional 1.0, which makes the
                                    marginalized weight reproduce the 3D
                                    lookup's total reweighted yield in each
                                    cell exactly, over this branch's own
                                    in-p_z-range population (checked in the
                                    tests). Cells with
                                    no in-range events in this file fall back
                                    to 1.0.

    The marginalized branch trades a stronger assumption -- that the
    data/AR23 discrepancy in a (p_T, SumT_p) cell is p_z-independent -- for
    much larger reach, since the p_z window is what limits the 3D branch (it
    alone costs ~12% of cvwgt; see the README). Being spectrum-weighted per
    file, it is NOT a fixed table: two files with different p_z spectra get
    slightly different marginalized weights.

Options:
  pz_ref: "LE" (default) | "ME" | "none"
          reference beam for the p_z scaling; "none" (scale 1.0) reproduces
          the literal published binning, for reference/debugging only.
  w_modes, divide_out_ff, branch, variable: as XSecMeasCalculator (w_modes
          defaults to nominal only for this calculator).
"""

import numpy as np

from ..calculator import register
from ..tki import (E_BNB, E_LE_MINERVA, E_ME_MINERVA, MINERVA_QELIKE_BRANCHES,
                   minerva_pz_pt, sig_minerva_qelike, sum_tp)
from ..xsec_table import load_minerva_3dqelike_table
from .xsec_meas import XSecMeasCalculator

PZ_REF_ENERGY = {"LE": E_LE_MINERVA, "ME": E_ME_MINERVA}

# branch-name suffix of the p_z-marginalized weight
MARG_SUFFIX = "_pzmarg"


@register("minerva_3dqelike")
class MINERvA3DQELike(XSecMeasCalculator):
    calc_name = "minerva_3dqelike"
    default_variable = "pz_pt_sumtp"

    def __init__(self, pz_ref="LE", w_modes=("nominal",), branch=None,
                 **kwargs):
        if pz_ref not in PZ_REF_ENERGY and pz_ref != "none":
            raise ValueError(
                f"unknown pz_ref '{pz_ref}'; expected LE, ME or none")
        self.pz_ref = pz_ref
        self.pz_scale = (1.0 if pz_ref == "none"
                         else E_BNB / PZ_REF_ENERGY[pz_ref])
        super().__init__(w_modes=w_modes,
                         branch=branch or f"fdwgt_{self.calc_name}_bnb",
                         **kwargs)
        if self.variable != self.default_variable:
            raise ValueError(
                "minerva_3dqelike has a single 3D (p_z, p_T, SumT_p) "
                f"measurement; variable must be '{self.default_variable}'")

    def table_branches(self):
        return list(MINERVA_QELIKE_BRANCHES)

    def load_table(self):
        return load_minerva_3dqelike_table()

    def signal_mask(self, a):
        return sig_minerva_qelike(a, self.pz_scale)

    def observable(self, a):
        pz, pt = minerva_pz_pt(a, self.pz_scale)
        return pz, pt, sum_tp(a)

    def compute(self, sbruce):
        out = super().compute(sbruce)
        out[f"{self.branch}{MARG_SUFFIX}"] = self._marginalized(sbruce)
        return out

    def _marginalized(self, sbruce):
        """p_z-marginalized weight, applied at any p_z (see module docstring)."""
        a = sbruce.arrays(self.branches_needed())
        n = sbruce.n_entries
        table = self.load_table()
        # no theta cut here: see sig_minerva_qelike(theta_cut=False)
        sig = sig_minerva_qelike(a, self.pz_scale, theta_cut=False)
        pz, pt, tp = self.observable(a)

        with np.errstate(invalid="ignore"):
            ipz, ipt, itp = table.axis_indices(
                *(np.where(sig, o, -1e9) for o in (pz, pt, tp)))

        # cvwgt-weighted p_z spectrum of THIS file, per 3D bin
        cv = a["cvwgt"].astype(np.float64)
        in3d = (ipz >= 0) & (ipt >= 0) & (itp >= 0)
        pop = np.zeros(table.shape, dtype=np.float64)
        np.add.at(pop, (ipz[in3d], ipt[in3d], itp[in3d]), cv[in3d])

        w3d = table.weights.reshape(table.shape)
        den = pop.sum(axis=0)
        num = (pop * w3d).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            w2d = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)

        # applied to every signal event in a (p_T, SumT_p) cell, ANY p_z
        in2d = (ipt >= 0) & (itp >= 0)
        weights = self.ones(n)
        weights[in2d] = w2d[ipt[in2d], itp[in2d]]

        name = f"{self.branch}{MARG_SUFFIX}"
        self.report_coverage(name, in2d, n)
        n_empty = int(np.count_nonzero(den <= 0))
        if n_empty:
            print(f"    [{name}] {n_empty}/{den.size} (p_T, SumT_p) cells have "
                  f"no in-p_z-range events in this file; weight 1.0 there")
        return weights
