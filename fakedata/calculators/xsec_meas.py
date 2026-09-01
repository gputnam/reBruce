"""Cross-section-measurement reweight calculators: AR23 -> measured xsec.

Measurements (weight tables in data/, see data/README.md):
  ub_cc1p0pi : MicroBooNE CC1mu1p0pi   (PRL 131 101802)
  ub_cc2p0pi : MicroBooNE CC1mu2p0pi   (PLB 872 140052)
  ub_ccpi    : MicroBooNE CC1pi+-      (PRD 113 032007)
  t2k_nc1pi  : T2K ND280 NC1pi         (PRL 135 171803)
and, in its own module (minerva_qelike.py, which subclasses the base here):
  minerva_3dqelike : MINERvA LE/ME 3D QE-like (arXiv:2606.00745)

Per config entry:
  variable:      reweighting observable (default per calculator; the
                 available names are the CSV "obs" fields, plus the alias
                 "dpt_dat" for the CC1p0pi 2D DeltaAlphaT x DeltaPT product
                 and "p_costh" for the T2K 2D (p_pi, cos_theta_pi) product)
  w_modes:       list from [nominal, loW, midW, hiW] (default all four);
                 each produces its own weight branch (suffix _loW etc.)
  divide_out_ff: divide out a deuterium->MINERvA axial-form-factor reweight
                 (computed with the validated Nieves port; the sBruce
                 multisigma trees store only CV-normalized variations, so
                 there is no stored branch to use) -- for input MC where
                 that reweight has been applied to the CV. Default false.

Signal events in a measured bin get the bin's weight; everything else gets
exactly 1.0. W modes follow fakedata/tercile.py.
"""

import numpy as np

from ..calculator import Calculator, dedup, register
from ..sbruce import valid
from ..tercile import W_MODES, wmode_weights
from ..tki import (BE_ECAL, M_MU, M_P, POSTFSI_BRANCHES, POSTFSI_PION_BRANCHES,
                   mom3, opening_cos, sig_cc1p0pi, sig_cc2p0pi,
                   sig_ccpi, sig_t2k_nc1pi, tki_ptx_pty, tki_vars)
from ..xsec_table import Table1D, load_t2k_table, load_ub_table
from .qe_zexp import QE_BRANCHES, deut_to_minerva_weight


def _sanitize(name):
    out = "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()
    while "__" in out:
        out = out.replace("__", "_")
    return out


class XSecMeasCalculator(Calculator):
    """Base class; subclasses define calc_name, default_variable, the
    observable computation, and the signal mask."""

    calc_name = None
    default_variable = None

    def __init__(self, variable=None, w_modes=("nominal", "loW", "midW", "hiW"),
                 divide_out_ff=False, branch=None):
        self.variable = variable or self.default_variable
        self.w_modes = list(w_modes)
        for m in self.w_modes:
            if m != "nominal" and m not in W_MODES:
                raise ValueError(f"unknown W mode: {m}")
        self.divide_out_ff = divide_out_ff
        self.branch = branch or f"wgt_{self.calc_name}_{_sanitize(self.variable)}"

    # subclass API ---------------------------------------------------------
    def load_table(self):
        raise NotImplementedError

    def signal_mask(self, a):
        raise NotImplementedError

    def observable(self, a):
        """Returns (x,) for 1D, (x, y) for 2D, (x, y, z) for 3D."""
        raise NotImplementedError

    def table_branches(self):
        """Branches signal_mask() and observable() read (subclass hook).

        Deliberately incomplete: branches_needed() below adds what the base
        compute() loads on every instance's behalf.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------
    def branches_needed(self):
        """cvwgt / genie_W drive the W-tercile modes and are loaded
        unconditionally by compute(); divide_out_ff additionally pulls in the
        Nieves QE branches read by deut_to_minerva_weight()."""
        return dedup(self.table_branches(),
                     ["cvwgt", "genie_W"],
                     QE_BRANCHES if self.divide_out_ff else [])

    def compute(self, sbruce):
        n = sbruce.n_entries
        a = sbruce.arrays(self.branches_needed())
        table = self.load_table()

        sig = self.signal_mask(a)
        obs = self.observable(a)
        with np.errstate(invalid="ignore"):
            if isinstance(table, Table1D):
                bin_idx = table.bin_index(np.where(sig, obs[0], -1e9))
            else:
                # 2D/3D tables take one masked array per observable
                bin_idx = table.bin_index(*(np.where(sig, o, -1e9) for o in obs))
        bin_idx = np.where(sig, bin_idx, -1)
        in_bin = bin_idx >= 0
        self.report_coverage(self.branch, in_bin, n)

        if self.divide_out_ff:
            w_ff = deut_to_minerva_weight(sbruce)

        out = {}
        for mode in self.w_modes:
            if mode == "nominal":
                w = np.where(in_bin, table.weights[np.maximum(bin_idx, 0)], 1.0)
                name = self.branch
            else:
                w = wmode_weights(
                    bin_idx, table.n_bins, table.weights,
                    a["genie_W"].astype(np.float64), valid(a["genie_W"]),
                    a["cvwgt"].astype(np.float64), mode)
                name = f"{self.branch}_{mode}"
            if self.divide_out_ff:
                w = w / w_ff
            out[name] = w.astype(np.float64)
        return out


@register("ub_cc1p0pi")
class UBCC1p0pi(XSecMeasCalculator):
    calc_name = "ub_cc1p0pi"
    default_variable = "dpt_dat"
    _CSV = "ub_cc1p0pi_xsec.csv"

    def table_branches(self):
        return list(POSTFSI_BRANCHES)

    def load_table(self):
        obs = ("DeltaAlphaT_DeltaPT" if self.variable in ("dpt_dat", "DeltaAlphaT_DeltaPT")
               else self.variable)
        return load_ub_table(self._CSV, obs)

    def signal_mask(self, a):
        return sig_cc1p0pi(a)

    def observable(self, a):
        vmu = mom3(a, "true_mu")
        vp = mom3(a, "true_p")
        dpt, dat, dphit = tki_vars(vmu, vp)
        if self.variable in ("dpt_dat", "DeltaAlphaT_DeltaPT"):
            return dpt, dat
        if self.variable == "DeltaPT":
            return (dpt,)
        if self.variable == "DeltaAlphaT":
            return (dat,)
        if self.variable == "DeltaPhiT":
            return (dphit,)
        if self.variable == "MuonCosTheta":
            return (a["true_mu_dir_z"],)
        if self.variable == "ProtonCosTheta":
            return (a["true_p_dir_z"],)
        if self.variable == "MuonMomentum":
            return (a["true_mu_p"],)
        if self.variable == "ProtonMomentum":
            return (a["true_p_p"],)
        if self.variable in ("DeltaPtx", "DeltaPty"):
            ptx, pty = tki_ptx_pty(vmu, vp)
            return (ptx,) if self.variable == "DeltaPtx" else (pty,)
        if self.variable == "ECal":
            emu = np.sqrt(a["true_mu_p"] ** 2 + M_MU ** 2)
            tp = np.sqrt(a["true_p_p"] ** 2 + M_P ** 2) - M_P
            return (emu + tp + BE_ECAL,)
        raise ValueError(f"unknown variable {self.variable} for {self.calc_name}")


@register("ub_cc2p0pi")
class UBCC2p0pi(XSecMeasCalculator):
    calc_name = "ub_cc2p0pi"
    default_variable = "delta_PT"
    _CSV = "ub_cc2p0pi_xsec.csv"

    def table_branches(self):
        return list(POSTFSI_BRANCHES)

    def load_table(self):
        return load_ub_table(self._CSV, self.variable)

    def signal_mask(self, a):
        return sig_cc2p0pi(a)

    def observable(self, a):
        vmu = mom3(a, "true_mu")
        v1 = mom3(a, "true_p")
        v2 = mom3(a, "true_p2")
        if self.variable in ("delta_PT", "delta_alphaT", "delta_phiT"):
            dpt, dat, dphit = tki_vars(vmu, v1 + v2)
            return {"delta_PT": (dpt,), "delta_alphaT": (dat,),
                    "delta_phiT": (dphit,)}[self.variable]
        if self.variable == "muon_mom":
            return (a["true_mu_p"],)
        if self.variable == "muon_costheta":
            return (a["true_mu_dir_z"],)
        if self.variable == "muon_phi":
            return (np.arctan2(a["true_mu_dir_y"], a["true_mu_dir_x"]),)
        if self.variable == "leading_mom":
            return (a["true_p_p"],)
        if self.variable == "leading_costheta":
            return (a["true_p_dir_z"],)
        if self.variable == "leading_phi":
            return (np.arctan2(a["true_p_dir_y"], a["true_p_dir_x"]),)
        if self.variable == "recoil_mom":
            return (a["true_p2_p"],)
        if self.variable == "recoil_costheta":
            return (a["true_p2_dir_z"],)
        if self.variable == "recoil_phi":
            return (np.arctan2(a["true_p2_dir_y"], a["true_p2_dir_x"]),)
        if self.variable == "opening_angle_protons_lab":
            return (opening_cos(v1, v2),)
        if self.variable == "opening_angle_mu_both":
            return (opening_cos(vmu, v1 + v2),)
        raise ValueError(f"unknown variable {self.variable} for {self.calc_name}")


@register("ub_ccpi")
class UBCCpi(XSecMeasCalculator):
    calc_name = "ub_ccpi"
    default_variable = "PionMomentum"
    _CSV = "ub_ccpi_xsec.csv"

    def table_branches(self):
        return POSTFSI_BRANCHES + POSTFSI_PION_BRANCHES

    def load_table(self):
        return load_ub_table(self._CSV, self.variable)

    def signal_mask(self, a):
        return sig_ccpi(a)

    def observable(self, a):
        with np.errstate(invalid="ignore", divide="ignore"):
            if self.variable == "MuonCosTheta":
                return (a["true_mu_dir_z"],)
            if self.variable == "MuonMomentum":
                return (a["true_mu_p"],)
            if self.variable == "PionCosTheta":
                return (a["true_cpi_dir_z"],)
            if self.variable == "PionMomentum":
                return (a["true_cpi_p"],)
            if self.variable == "ThetaMuPi":
                return (np.arccos(opening_cos(mom3(a, "true_mu"),
                                              mom3(a, "true_cpi"))),)
        raise ValueError(f"unknown variable {self.variable} for {self.calc_name}")


@register("t2k_nc1pi")
class T2KNC1pi(XSecMeasCalculator):
    calc_name = "t2k_nc1pi"
    default_variable = "p_costh"

    def table_branches(self):
        # sig_t2k_nc1pi is a pure NC / one-charged-pion final-state cut: no
        # beam-energy, flavour or proton-count branch is read.
        return ["true_isnc", "true_npi", "true_npi0"] + POSTFSI_PION_BRANCHES

    def load_table(self):
        if self.variable != "p_costh":
            raise ValueError(
                "t2k_nc1pi has a single 2D (p_pi, cos_theta_pi) measurement; "
                "variable must be 'p_costh'")
        return load_t2k_table()

    def signal_mask(self, a):
        return sig_t2k_nc1pi(a)

    def observable(self, a):
        return a["true_cpi_p"], a["true_cpi_dir_z"]
