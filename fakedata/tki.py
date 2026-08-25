"""True-kinematics observables and signal definitions for the
cross-section-measurement calculators.

Post-FSI observables are computed from the momentum-ordered final-state
truth branches (true_mu_*, true_p_*, true_p2_*) in detector coordinates
with the beam along +z. Pion observables are computed from the pre-FSI
GENIE record (genie_prefsi_cpi_*, genie_prefsi_lep_*) in the nu/lepton
frame (z along the true neutrino) -- a PLACEHOLDER for missing post-FSI
pion kinematics; see MISSING_INFO.md.

Signal definitions mirror the NUISANCE / paper definitions, with particle
multiplicities taken from the sBruce truth counts (true_np, true_npi,
true_npi0). Per analysis convention these counts are ASSUMED to use the
same phase-space thresholds as the measurements (see MISSING_INFO.md);
explicit momentum windows are additionally applied to the particles whose
kinematics are stored (muon, leading/subleading proton, leading pion).

TKI conventions (PRL 131 101802; PLB 872 140052):
  deltaPT      = |muT + hadT|                                 [GeV]
  deltaAlphaT  = acos( -muT.dPT / (|muT||dPT|) )              [deg]
  deltaPhiT    = acos( -muT.hadT / (|muT||hadT|) )            [deg]
  deltaPtx     = (-muT_y*sumT_x + muT_x*sumT_y)/|muT|         [GeV]
  deltaPty     = -(muT.sumT)/|muT|                            [GeV]
  ECal         = E_mu + T_p + 0.04                            [GeV]
with hadT the transverse momentum of the signal proton (CC1p) or the vector
sum of both signal protons (CC2p), and sumT = muT + hadT.
"""

import numpy as np

from .sbruce import valid

M_MU = 0.10565837
M_P = 0.938272081
BE_ECAL = 0.04

POSTFSI_BRANCHES = [
    "nu_E", "true_isnc", "true_pdg",
    "true_np", "true_npi", "true_npi0",
    "true_mu_p", "true_mu_dir_x", "true_mu_dir_y", "true_mu_dir_z",
    "true_p_p", "true_p_dir_x", "true_p_dir_y", "true_p_dir_z",
    "true_p2_p", "true_p2_dir_x", "true_p2_dir_y", "true_p2_dir_z",
]

PREFSI_PION_BRANCHES = [
    "genie_prefsi_lep_px", "genie_prefsi_lep_py", "genie_prefsi_lep_pz",
    "genie_prefsi_cpi_px", "genie_prefsi_cpi_py", "genie_prefsi_cpi_pz",
]


def mom3(a, prefix):
    """3-momentum (N, 3) from sBruce p + dir branches."""
    p = a[f"{prefix}_p"]
    return p[:, None] * np.stack(
        [a[f"{prefix}_dir_x"], a[f"{prefix}_dir_y"], a[f"{prefix}_dir_z"]],
        axis=1)


def prefsi3(a, species):
    return np.stack([a[f"genie_prefsi_{species}_px"],
                     a[f"genie_prefsi_{species}_py"],
                     a[f"genie_prefsi_{species}_pz"]], axis=1)


def _acos_deg(c):
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def tki_vars(vmu, vhad):
    """(deltaPT, deltaAlphaT[deg], deltaPhiT[deg]) from 3-momenta (N,3)."""
    muT = vmu[:, :2]
    hadT = vhad[:, :2]
    sumT = muT + hadT
    dpt = np.hypot(sumT[:, 0], sumT[:, 1])
    nmu = np.hypot(muT[:, 0], muT[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        dat = _acos_deg(-np.sum(muT * sumT, axis=1) / (nmu * dpt))
        dphit = _acos_deg(-np.sum(muT * hadT, axis=1)
                          / (nmu * np.hypot(hadT[:, 0], hadT[:, 1])))
    return dpt, dat, dphit


def tki_ptx_pty(vmu, vp):
    muT = vmu[:, :2]
    sumT = muT + vp[:, :2]
    nmu = np.hypot(muT[:, 0], muT[:, 1])
    pty = -np.sum(muT * sumT, axis=1) / nmu
    ptx = (-muT[:, 1] * sumT[:, 0] + muT[:, 0] * sumT[:, 1]) / nmu
    return ptx, pty


def opening_cos(v1, v2):
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(np.sum(v1 * v2, axis=1) / (n1 * n2), -1.0, 1.0)


# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------

def sig_cc1p0pi(a):
    """MicroBooNE CC1mu1p0pi (PRL 131 101802)."""
    return (
        (a["true_isnc"] == 0) & (a["true_pdg"] == 14)
        & (a["nu_E"] > 0) & (a["nu_E"] < 6.8)
        & (a["true_mu_p"] > 0.1) & (a["true_mu_p"] < 1.2)
        & (a["true_np"] == 1)
        & (a["true_p_p"] > 0.3) & (a["true_p_p"] < 1.0)
        & (a["true_npi"] == 0) & (a["true_npi0"] == 0)
        & valid(a["true_mu_dir_z"], a["true_p_dir_z"])
    )


def sig_cc2p0pi(a):
    """MicroBooNE CC1mu2p0pi (PLB 872 140052)."""
    return (
        (a["true_isnc"] == 0) & (a["true_pdg"] == 14)
        & (a["nu_E"] > 0) & (a["nu_E"] < 6.8)
        & (a["true_mu_p"] > 0.1) & (a["true_mu_p"] < 1.2)
        & (a["true_np"] == 2)
        & (a["true_p_p"] > 0.3) & (a["true_p_p"] < 1.0)
        & (a["true_p2_p"] > 0.3) & (a["true_p2_p"] < 1.0)
        & (a["true_npi"] == 0) & (a["true_npi0"] == 0)
        & valid(a["true_mu_dir_z"], a["true_p_dir_z"], a["true_p2_dir_z"])
    )


def sig_ccpi(a):
    """MicroBooNE CC1pi+- (PRD 113 032007). Pion kinematics are pre-FSI
    (placeholder); the theta_mu_pi < 2.65 phase-space cut is applied in the
    pre-FSI nu/lepton frame."""
    p_lep = prefsi3(a, "lep")
    p_cpi = prefsi3(a, "cpi")
    pmu = np.linalg.norm(p_lep, axis=1)
    ppi = np.linalg.norm(p_cpi, axis=1)
    with np.errstate(invalid="ignore"):
        th = np.arccos(opening_cos(p_lep, p_cpi))
    return (
        (a["true_isnc"] == 0) & (a["true_pdg"] == 14)
        & (a["nu_E"] > 0)
        & (a["true_npi"] == 1) & (a["true_npi0"] == 0)
        & valid(a["genie_prefsi_lep_pz"], a["genie_prefsi_cpi_pz"])
        & (pmu > 0.15) & (ppi > 0.10) & (th < 2.65)
    )


def sig_t2k_nc1pi(a):
    """T2K ND280 NC1pi (PRL 135 171803). NC, exactly one charged pion;
    pion kinematics are pre-FSI (placeholder). The measured region cut in
    (p_pi, cos_theta_pi) is applied by the weight table itself."""
    return (
        (a["true_isnc"] == 1)
        & (a["true_npi"] == 1) & (a["true_npi0"] == 0)
        & valid(a["genie_prefsi_cpi_pz"])
    )
