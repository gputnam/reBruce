"""QE axial-form-factor reweight: MINERvA measurement -> LQCD average.

The weight is a ratio of Nieves/Valencia CCQE differential cross sections
evaluated at the event's true kinematics with only the z-expansion axial
form factor swapped (NOT an FA^2 ratio -- see fakedata/nieves.py):

    w = xsec(FA = LQCD) / xsec(FA = MINERvA)

Intended for input MC whose QE axial form factor has (effectively) been
reweighted to the MINERvA measurement; applying this weight moves it to the
LQCD average. Composing with a deuterium->MINERvA weight gives
deuterium->LQCD.

Coefficient-set conventions (config option ga_convention):
  - "tune" (default): each set uses its own GENIE-tune FA(0)
    (minervaNature gA = -1.2723, lqcd gA = -1.2754), matching what a
    regeneration with tunes AR23_20i_01_001 / AR23_20i_02_000 would give.
  - "nusyst": both sets use the AR23 CV FA(0) = -1.2670, matching the
    nusystematics ZExpPCAWeighter convention (which dials only a1..a4/T0).

Domain: numu CC QE (genie_mode == 0, true_isnc == 0, true_pdg == 14) with
valid genie_prefsi_lep and genie_prefsi_p momenta. Weight = 1 elsewhere.
Antineutrinos are NOT reweighted (the pre-FSI record stores no recoil
neutron; see MISSING_INFO.md).

The Nieves port is validated event-by-event against the GENIE-computed
ZExpPCAWeighter_SBN_v3_MvA b-dial weights in the sBruce multisigmaTree
(agreement: mean 1.0000, RMS < 1e-3); run validation/validate_nieves.py.
"""

import numpy as np

from ..calculator import Calculator, register
from ..nieves import NievesQEXSec
from ..sbruce import MODE_QE, valid
from ..zexp import COEFF_SETS, ZExpAxialFF

GA_CV = -1.2670

QE_BRANCHES = [
    "genie_mode",
    "true_isnc",
    "true_pdg",
    "genie_Enu",
    "genie_prefsi_lep_px", "genie_prefsi_lep_py", "genie_prefsi_lep_pz",
    "genie_prefsi_p_px", "genie_prefsi_p_py", "genie_prefsi_p_pz",
]


def qe_numu_prefsi_mask(a):
    """numu CC QE events with the pre-FSI kinematics needed for Nieves."""
    return (
        (a["genie_mode"] == MODE_QE)
        & (a["true_isnc"] == 0)
        & (a["true_pdg"] == 14)
        & valid(
            a["genie_Enu"],
            a["genie_prefsi_lep_py"], a["genie_prefsi_lep_pz"],
            a["genie_prefsi_p_px"], a["genie_prefsi_p_py"], a["genie_prefsi_p_pz"],
        )
    )


def qe_kinematics(a, mask):
    """(enu, p_lep, p_nf) float64 arrays for masked events."""
    enu = a["genie_Enu"][mask].astype(np.float64)
    p_lep = np.stack(
        [a["genie_prefsi_lep_px"][mask], a["genie_prefsi_lep_py"][mask],
         a["genie_prefsi_lep_pz"][mask]], axis=1).astype(np.float64)
    p_nf = np.stack(
        [a["genie_prefsi_p_px"][mask], a["genie_prefsi_p_py"][mask],
         a["genie_prefsi_p_pz"][mask]], axis=1).astype(np.float64)
    return enu, p_lep, p_nf


def _ff(set_name, ga_convention):
    cfg = dict(COEFF_SETS[set_name])
    if ga_convention == "nusyst":
        cfg["ga"] = GA_CV
    elif ga_convention != "tune":
        raise ValueError(f"unknown ga_convention: {ga_convention}")
    return ZExpAxialFF(**cfg)


def deut_to_minerva_weight(sbruce, xsec=None):
    """Per-event deuterium->MINERvA CV Nieves weight (nusystematics
    convention: gA fixed at the AR23 CV, T0 -> -0.75). Weight = 1 outside
    the numu CC QE domain. Used by the divide-out-form-factor option of the
    cross-section-measurement calculators."""
    a = sbruce.arrays(QE_BRANCHES)
    mask = qe_numu_prefsi_mask(a)
    w = np.ones(sbruce.n_entries, dtype=np.float64)
    if not np.any(mask):
        return w
    xsec = xsec or NievesQEXSec()
    fa_deut = _ff("deuterium", "nusyst")
    fa_mva = _ff("minerva_nature", "nusyst")
    enu, p_lep, p_nf = qe_kinematics(a, mask)
    w[mask] = xsec.weight_ratio(enu, p_lep, p_nf, fa_mva, fa_deut)
    return w


@register("qe_zexp_mva_to_lqcd")
class QEZexpMvaToLQCD(Calculator):
    def __init__(self, branch="wgt_qe_zexp_mva_to_lqcd", ga_convention="tune",
                 n_r=16):
        self.branch = branch
        self.ga_convention = ga_convention
        self.xsec = NievesQEXSec(n_r=n_r)

    def compute(self, sbruce):
        a = sbruce.arrays(QE_BRANCHES)
        n = sbruce.n_entries
        mask = qe_numu_prefsi_mask(a)
        self.report_coverage(self.branch, mask, n)

        weights = self.ones(n)
        if not np.any(mask):
            return {self.branch: weights}

        fa_old = _ff("minerva_nature", self.ga_convention)
        fa_new = _ff("lqcd", self.ga_convention)
        enu, p_lep, p_nf = qe_kinematics(a, mask)
        weights[mask] = self.xsec.weight_ratio(enu, p_lep, p_nf, fa_new, fa_old)
        return {self.branch: weights}
