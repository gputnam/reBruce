"""MEC BDT reweight: AR23/SuSAv2 CCMEC -> exclusive-Valencia 2p2h, evaluated
on GENIE pre-FSI kinematics.

The BDT (hep_ml GBReweighter, see data/README.md) was trained on pre-FSI
vertex-nucleon and muon momenta of numu-CC MEC events with a pp pre-FSI
nucleon pair, in a "reaction frame": neutrino along +z, muon pT along -y.

The sBruce genie_prefsi_* momenta are stored in a nu/lepton frame with the
lepton pT along +y and x-hat = y-hat x z-hat. The BDT frame has muon pT along
-y and x-hat constructed left-handed; the two x-axes coincide, so the frame
conversion is exactly a sign flip of the y components.

Feature order (positional, matches training):
    [p_px, p_py, p_pz, p2_px, p2_py, p2_pz, mu_py, mu_pz]

Mask (weight = 1 outside):
    genie_mode == 10 (MEC), true_isnc == 0, true_pdg == 14,
    genie_prefsi_{lep,p,p2} momenta all filled (> -900).
    A filled p2 implies two pre-FSI protons, matching the training's
    pp-only selection. Momentum ordering (|p|) equals the training's
    kinetic-energy ordering for equal-mass protons.

Normalization (default "per-file"): weights are scaled so the cvwgt-weighted
mean weight over masked events is exactly 1 (shape-only reweight; the MEC
rate in the file is unchanged).
"""

import os

import numpy as np

from ..bdt import GBReweighterJSON
from ..calculator import Calculator, register
from ..sbruce import MODE_MEC, valid

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "mec_bdt_susav2_to_valencia.json"
)

BRANCHES = [
    "genie_mode",
    "true_isnc",
    "true_pdg",
    "cvwgt",
    "genie_prefsi_lep_px", "genie_prefsi_lep_py", "genie_prefsi_lep_pz",
    "genie_prefsi_p_px", "genie_prefsi_p_py", "genie_prefsi_p_pz",
    "genie_prefsi_p2_px", "genie_prefsi_p2_py", "genie_prefsi_p2_pz",
]


@register("mec_bdt")
class MECBDTCalculator(Calculator):
    def __init__(self, branch="wgt_mec_bdt", normalize="per-file", norm_scale=None,
                 model=None):
        """normalize: "per-file" (default), "none", or "fixed" with norm_scale."""
        self.branch = branch
        self.normalize = normalize
        self.norm_scale = norm_scale
        self.model = GBReweighterJSON(model or MODEL_PATH)

    def branches_needed(self):
        return list(BRANCHES)

    def compute(self, sbruce):
        a = sbruce.arrays(self.branches_needed())
        n = sbruce.n_entries

        mask = (
            (a["genie_mode"] == MODE_MEC)
            & (a["true_isnc"] == 0)
            & (a["true_pdg"] == 14)
            & valid(
                a["genie_prefsi_lep_py"], a["genie_prefsi_lep_pz"],
                a["genie_prefsi_p_px"], a["genie_prefsi_p_py"], a["genie_prefsi_p_pz"],
                a["genie_prefsi_p2_px"], a["genie_prefsi_p2_py"], a["genie_prefsi_p2_pz"],
            )
        )
        self.report_coverage(self.branch, mask, n)

        weights = self.ones(n)
        if not np.any(mask):
            return {self.branch: weights}

        # nu/lepton frame -> BDT reaction frame: flip y components
        feats = np.column_stack([
            a["genie_prefsi_p_px"][mask],
            -a["genie_prefsi_p_py"][mask],
            a["genie_prefsi_p_pz"][mask],
            a["genie_prefsi_p2_px"][mask],
            -a["genie_prefsi_p2_py"][mask],
            a["genie_prefsi_p2_pz"][mask],
            -a["genie_prefsi_lep_py"][mask],
            a["genie_prefsi_lep_pz"][mask],
        ]).astype(np.float64)

        w_raw = self.model.predict_weights(feats)

        if self.normalize == "per-file":
            cv = a["cvwgt"][mask].astype(np.float64)
            scale = np.sum(cv) / np.sum(cv * w_raw)
            print(f"    [{self.branch}] per-file norm scale: {scale:.6f}")
        elif self.normalize == "fixed":
            scale = float(self.norm_scale)
        elif self.normalize == "none":
            scale = 1.0
        else:
            raise ValueError(f"unknown normalize mode: {self.normalize}")

        weights[mask] = scale * w_raw
        return {self.branch: weights}
