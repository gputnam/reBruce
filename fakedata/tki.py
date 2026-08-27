"""True-kinematics observables and signal definitions for the
cross-section-measurement calculators.

All observables are computed from the momentum-ordered final-state truth
branches (true_mu_*, true_p_*, true_p2_*, and -- since sBruce schema 20 --
the leading charged pion true_cpi_*) in detector coordinates with the beam
along +z. The pion's charge is still not stored (true_npi counts
|pdg| == 211); see MISSING_INFO.md.

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

# MINERvA LE/ME QE-like measurement: effective (flux-peak) neutrino energies
# of the two NuMI beams and of the BNB, as used by the MvA_LE_ME analysis
# that produced data/minerva_3dqelike_bnb.csv.
E_LE_MINERVA = 3.25
E_ME_MINERVA = 5.85
E_BNB = 0.70

POSTFSI_BRANCHES = [
    "nu_E", "true_isnc", "true_pdg",
    "true_np", "true_npi", "true_npi0",
    "true_mu_p", "true_mu_dir_x", "true_mu_dir_y", "true_mu_dir_z",
    "true_p_p", "true_p_dir_x", "true_p_dir_y", "true_p_dir_z",
    "true_p2_p", "true_p2_dir_x", "true_p2_dir_y", "true_p2_dir_z",
]

# leading charged pion, post-FSI (sBruce schema >= 20)
POSTFSI_PION_BRANCHES = [
    "true_cpi_p", "true_cpi_dir_x", "true_cpi_dir_y", "true_cpi_dir_z",
]

# MINERvA QE-like signal: muon, the two stored protons, and the multiplicity
# counts plus the leading photon used for the meson / photon vetoes.
MINERVA_QELIKE_BRANCHES = [
    "true_isnc", "true_pdg",
    "true_mu_p", "true_mu_dir_x", "true_mu_dir_y", "true_mu_dir_z",
    "true_p_p", "true_p2_p", "true_np",
    "true_npi", "true_npi0", "true_g_p",
]


def mom3(a, prefix):
    """3-momentum (N, 3) from sBruce p + dir branches."""
    p = a[f"{prefix}_p"]
    return p[:, None] * np.stack(
        [a[f"{prefix}_dir_x"], a[f"{prefix}_dir_y"], a[f"{prefix}_dir_z"]],
        axis=1)


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
    """MicroBooNE CC1pi+- (PRD 113 032007). Post-FSI muon and leading
    charged pion (detector frame, beam +z)."""
    vmu = mom3(a, "true_mu")
    vpi = mom3(a, "true_cpi")
    with np.errstate(invalid="ignore"):
        th = np.arccos(opening_cos(vmu, vpi))
    return (
        (a["true_isnc"] == 0) & (a["true_pdg"] == 14)
        & (a["nu_E"] > 0)
        & (a["true_npi"] == 1) & (a["true_npi0"] == 0)
        & valid(a["true_mu_dir_z"], a["true_cpi_p"], a["true_cpi_dir_z"])
        & (a["true_mu_p"] > 0.15) & (a["true_cpi_p"] > 0.10) & (th < 2.65)
    )


def sig_t2k_nc1pi(a):
    """T2K ND280 NC1pi (PRL 135 171803). NC, exactly one charged pion,
    post-FSI pion kinematics. The measured region cut in
    (p_pi, cos_theta_pi) is applied by the weight table itself."""
    return (
        (a["true_isnc"] == 1)
        & (a["true_npi"] == 1) & (a["true_npi0"] == 0)
        & valid(a["true_cpi_p"], a["true_cpi_dir_z"])
    )


def sum_tp(a):
    """SumT_p = Sum(E - M_P) over the final-state protons [GeV].

    sBruce stores only the leading two protons, so events with true_np > 2
    are under-counted and migrate to lower SumT_p bins (see MISSING_INFO.md).
    Unfilled proton branches contribute 0, so a 0-proton event gets
    SumT_p = 0 and lands in the first bin -- which is correct: the MINERvA
    QE-like signal admits zero-proton final states.
    """
    out = np.zeros(len(a["true_p_p"]), dtype=np.float64)
    for prefix in ("true_p_p", "true_p2_p"):
        p = np.where(valid(a[prefix]), a[prefix], 0.0)
        out += np.sqrt(p ** 2 + M_P ** 2) - M_P
    return out


def minerva_pz_pt(a, pz_scale):
    """Muon (p_z, p_T) mapped into the MINERvA measurement frame [GeV/c].

    The measurement's p_z axis is treated as scaled relative to the neutrino
    energy rather than as absolute momentum: a measured edge p_z corresponds
    at BNB to E_BNB * (p_z / E_beam), so an event's measurement-frame
    longitudinal momentum is p_z / pz_scale with pz_scale = E_BNB / E_beam.
    p_T is NOT scaled (it is set by the Q^2 scale, not the beam energy).
    """
    pz = a["true_mu_p"] * a["true_mu_dir_z"] / pz_scale
    pt = a["true_mu_p"] * np.hypot(a["true_mu_dir_x"], a["true_mu_dir_y"])
    return pz, pt


def sig_minerva_qelike(a, pz_scale, theta_cut=True):
    """MINERvA LE/ME 3D QE-like (arXiv:2606.00745), NUISANCE
    isCC0pi_MINERvAPTPZ: numu CC, theta(mu, nu) < 20 deg, no mesons and no
    photon above 10 MeV in the final state.

    The theta cut is applied on the SCALED kinematics, i.e. in the same frame
    the (p_z, p_T, SumT_p) bin lookup happens; the measured-region cut itself
    is left to the weight table, as in sig_t2k_nc1pi.

    theta_cut=False drops the 20 deg requirement, for the p_z-marginalized
    weight: once that weight is applied at any p_z, the theta cut is no
    longer nearly-redundant with the p_z window (which is what made it cheap
    for the 3D weight) and instead becomes the binding acceptance constraint.

    The measurement additionally vetoes heavy baryons and charm, which sBruce
    does not record (see MISSING_INFO.md); "no other mesons" reduces to the
    pi+- / pi0 vetoes for the same reason.
    """
    sig = (
        (a["true_isnc"] == 0) & (a["true_pdg"] == 14)
        & valid(a["true_mu_p"], a["true_mu_dir_z"])
        & (a["true_npi"] == 0) & (a["true_npi0"] == 0)
        & ~(a["true_g_p"] > 0.010)
    )
    if not theta_cut:
        return sig
    pz, pt = minerva_pz_pt(a, pz_scale)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_th = pz / np.hypot(pz, pt)
    return sig & (cos_th > np.cos(np.radians(20.0)))
