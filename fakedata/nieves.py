"""Nieves/Valencia CCQE differential cross section, ported from GENIE
NievesQELCCPXSec (SBNSoftware/Generator @ v3_06_02_br) for axial-form-factor
reweighting: the weight is the ratio of cross sections evaluated at the
event's kinematics with only the axial form factor swapped
(exactly what GENIE Reweight's CalcWeightZExp computes for a Nieves event).

Reference: Nieves, Amaro, Valverde, PRC 70 (2004) 055503.

Everything that is independent of the axial form factor cancels in the ratio
(Gfactor, the energy-delta-function phase-space factor, NNucl, xsec scale).
The parts that do NOT cancel and are ported here:
  - the lepton-hadron tensor contraction LmunuAnumu (all form factors),
  - the RPA polarization coefficients CN/CT/CL (identical in numerator and
    denominator but they multiply different form-factor terms),
  - the Coulomb-corrected momentum transfer qTilde (enters the tensor),
  - Pauli blocking and validity guards (enter the radius marginalization).

SBN-fork specifics reproduced: LmunuAnumu boosts all four-vectors to the
struck-nucleon rest frame before evaluating the tensor, so the initial
nucleon four-momentum entering the hadron tensor is exactly (mNi, 0, 0, 0).

The hit-nucleon radius r is NOT stored in sBruce files. GENIE evaluates the
model at the sampled radius; here the cross section is marginalized over r
with the vertex-sampling prior rho(r) r^2 dr (times the r-dependent parts of
the cross section itself, i.e. a posterior average):

    w = sum_r [ rho r^2 dr * xsec_new(r) ] / sum_r [ rho r^2 dr * xsec_old(r) ]

This is an approximation; it is validated against the GENIE-computed
deuterium->MINERvA weights stored in the sBruce multisigmaTree
(ZExpPCAWeighter_SBN_v3_MvA at sigma=0). See MISSING_INFO.md.

All momenta GeV; radii fm. q2 is spacelike (< 0).
"""

import numpy as np

from .bba07 import lwlyn_smith_ccff, M_PROTON, M_NEUTRON

HBARC = 0.1973269804        # GeV fm (GENIE fhbarc)
HBARC2 = HBARC * HBARC
ALPHA = 1.0 / 137.03599976  # GENIE kAem
PI = np.pi
PI2 = PI * PI
K_PION_MASS2 = 0.019479835145  # GENIE kPionMass2 (pi+/-)
K_MUON_MASS = 0.10565837
K_SMALL = 1e-6
R0_FM = 1.4                 # AR23 NUCL-R0; CoulombRmaxMode=VertexGenerator

# ---------------------------------------------------------------------------
# Nuclear density (genie::utils::nuclear::Density, A > 20 Woods-Saxon branch)
# ---------------------------------------------------------------------------

_WS_PARAMS = {  # A: (c, z) special cases in NuclearUtils
    27: (3.07, 0.52), 28: (3.07, 0.54), 40: (3.53, 0.54),
    56: (4.10, 0.56), 208: (6.62, 0.55),
}


def ws_density(r, A=40):
    """Woods-Saxon density normalized to integrate to 1 (fm^-3)."""
    c, z = _WS_PARAMS.get(A, (A ** 0.35, 0.54))
    norm = 3.0 / (4.0 * PI * c ** 3) / (1.0 + (PI * z / c) ** 2)
    return norm / (1.0 + np.exp((np.asarray(r, dtype=np.float64) - c) / z))


# ---------------------------------------------------------------------------
# Coulomb potential (NievesQELCCPXSec::vcr, RmaxMode = VertexGenerator)
# ---------------------------------------------------------------------------

def coulomb_vcr(r, A=40, Z=18, npts=200):
    """V_c(r) in GeV (negative). Vectorized over r (fm)."""
    rmax = 3.0 * R0_FM * A ** (1.0 / 3.0)
    r = np.minimum(np.asarray(r, dtype=np.float64), rmax)

    # integral of Z*rho(rin) * (rin^2/r for rin<r else rin), split at the kink
    x1, w1 = np.polynomial.legendre.leggauss(npts)
    out = np.zeros_like(r)
    for i, rc in np.ndenumerate(r):
        # [0, rc]: Z rho rin^2 / rc
        a, b = 0.0, rc
        t = 0.5 * (b - a) * x1 + 0.5 * (b + a)
        i1 = 0.5 * (b - a) * np.sum(w1 * Z * ws_density(t, A) * t * t) / rc if rc > 0 else 0.0
        # [rc, rmax]: Z rho rin
        a, b = rc, rmax
        t = 0.5 * (b - a) * x1 + 0.5 * (b + a)
        i2 = 0.5 * (b - a) * np.sum(w1 * Z * ws_density(t, A) * t)
        out[i] = i1 + i2
    return -ALPHA * 4.0 * PI * out * HBARC


# ---------------------------------------------------------------------------
# Lindhard functions (verbatim ports)
# ---------------------------------------------------------------------------

def rel_lindhard_im(q0, dq, kf1, kf2, M, is_neutrino):
    """Imaginary part of the relativistic Lindhard function (GeV^2).

    kf1, kf2 are passed exactly as the C++ caller does: (kF of the hit
    species, kF of the other species). The C++ parameter names (kFn, kFp)
    then produce the branch below -- reproduced literally, including the
    effective swap for antineutrinos.
    """
    M2 = M * M
    if is_neutrino:
        EF1 = np.sqrt(M2 + kf1 ** 2)
        EF2 = np.sqrt(M2 + kf2 ** 2)
    else:
        EF1 = np.sqrt(M2 + kf2 ** 2)
        EF2 = np.sqrt(M2 + kf1 ** 2)
    q2 = q0 ** 2 - dq ** 2
    a = (-q0 + dq * np.sqrt(1.0 - 4.0 * M2 / q2)) / 2.0
    epsRP = np.maximum(np.maximum(M, EF2 - q0), a)
    return -M2 / 2.0 / PI / dq * (EF1 - epsRP)  # the imaginary part (real array)


def _ru_lin_rel_x(q0, qm, kf, m):
    """Barbaro et al. EPJ A25 (2005) 299 Eq. 61; natural units, real-valued."""
    q02 = q0 ** 2
    qm2 = qm ** 2
    kf2 = kf ** 2
    m2 = m * m
    m4 = m2 * m2
    ef = np.sqrt(m2 + kf2)
    q2 = q02 - qm2
    q4 = q2 ** 2
    ds = np.sqrt(1.0 - 4.0 * m2 / q2)
    L1 = np.log((kf + ef) / m)
    uL2 = (np.log(np.abs((ef + q0 - np.sqrt(m2 + (kf - qm) ** 2))
                         / (ef + q0 - np.sqrt(m2 + (kf + qm) ** 2))))
           + np.log(np.abs((ef + q0 + np.sqrt(m2 + (kf - qm) ** 2))
                           / (ef + q0 + np.sqrt(m2 + (kf + qm) ** 2)))))
    uL3 = (np.log(np.abs(((2 * kf + q0 * ds) ** 2 - qm2)
                         / ((2 * kf - q0 * ds) ** 2 - qm2)))
           + np.log(np.abs(((kf - ef * ds) ** 2 - 4 * m4 * qm2 / q4)
                           / ((kf + ef * ds) ** 2 - 4 * m4 * qm2 / q4))))
    out = (-L1 / (16.0 * PI2) + uL2 * (2.0 * ef + q0) / (32.0 * PI2 * qm)
           - uL3 * ds / (64.0 * PI2))
    return out * 16.0 * m2


def rel_lindhard(q0gev, dqgev, kfgev, M, is_neutrino):
    """Nucleon relativistic Lindhard function (complex, GeV^2)."""
    q0 = q0gev / HBARC
    qm = dqgev / HBARC
    kf = kfgev / HBARC
    m = M / HBARC
    real_part = _ru_lin_rel_x(q0, qm, kf, m) + _ru_lin_rel_x(-q0, qm, kf, m)
    im = rel_lindhard_im(q0gev, dqgev, kfgev, kfgev, M, is_neutrino)
    out = real_part * HBARC2 + 2.0j * im
    # C++ bails out (returns 0) for timelike qTilde
    return np.where(q0 > qm, 0.0 + 0.0j, out)


def delta_lindhard(q0, dq, rho, kf):
    """Delta-hole Lindhard function (Oset et al. Phys.Rept. 188:79 Eq. A.4).

    q0, dq, kf in GeV; rho in fm^-3. Returns complex, GeV^2.
    """
    q_zero = q0 / HBARC
    q_mod = dq / HBARC
    k_fermi = kf / HBARC

    m = 4.7592
    md = 6.2433
    mpi = 0.7045
    fdel_f = 2.13
    wr = md - m

    q_zero2 = q_zero ** 2
    q_mod2 = q_mod ** 2
    k_fermi2 = k_fermi ** 2
    m2 = m * m
    m4 = m2 * m2
    mpi2 = mpi * mpi
    mpi4 = mpi2 * mpi2
    fdel_f2 = fdel_f ** 2

    def _gamma(sv):
        srot = np.sqrt(np.abs(sv))
        qcm = np.sqrt(np.abs(sv ** 2 + mpi4 + m4
                             - 2.0 * (sv * mpi2 + sv * m2 + mpi2 * m2))) / (2.0 * srot)
        g = (1.0 / 3.0 / (4.0 * PI) * fdel_f2 * qcm ** 3 / srot
             * (m + np.sqrt(m2 + qcm ** 2)) / mpi2)
        return np.where(sv > (m + mpi) ** 2, g, 0.0)

    s = m2 + q_zero2 - q_mod2 + 2.0 * q_zero * np.sqrt(m2 + 0.6 * k_fermi2)
    sp = m2 + q_zero2 - q_mod2 - 2.0 * q_zero * np.sqrt(m2 + 0.6 * k_fermi2)
    gamma = _gamma(s)
    gammap = _gamma(sp)

    z = md / (q_mod * k_fermi) * (q_zero - q_mod2 / (2.0 * md) - wr + 0.5j * gamma)
    zp = md / (q_mod * k_fermi) * (-q_zero - q_mod2 / (2.0 * md) - wr + 0.5j * gammap)

    def _pzeta(zz):
        azz = np.abs(zz)
        # general branch (guard the log argument for the other branches)
        safe = np.where((azz <= 50.0) & (azz >= 1e-2), zz, 2.0)
        general = safe + (1.0 - safe * safe) * np.log((safe + 1.0) / (safe - 1.0)) / 2.0
        large = 2.0 / (3.0 * zz) + 2.0 / (15.0 * zz * zz * zz)
        small = 2.0 * zz - 2.0 / 3.0 * zz ** 3 - 0.5j * PI * (1.0 - zz * zz)
        return np.where(azz > 50.0, large, np.where(azz < 1e-2, small, general))

    return (2.0 / 3.0 * rho * md / (q_mod * k_fermi)
            * (_pzeta(z) + _pzeta(zp)) * fdel_f2 * HBARC2)


# ---------------------------------------------------------------------------
# RPA polarization coefficients (NievesQELCCPXSec::CNCTCLimUcalc)
# ---------------------------------------------------------------------------

def cnctcl(q0t, dqt, r, M, is_neutrino, hit_is_proton, A=40, Z=18, N=22):
    """CN, CT, CL, imU. q0t/dqt: qTilde components in the frame used by
    LmunuAnumu (nucleon rest frame). Broadcastable arrays."""
    q0t, dqt, r = np.broadcast_arrays(
        np.asarray(q0t, np.float64), np.asarray(dqt, np.float64),
        np.asarray(r, np.float64))
    dq2 = dqt ** 2
    q2 = q0t ** 2 - dq2

    c0 = 0.380 / HBARC
    dens = ws_density(r, A)
    rhop = dens * Z
    rhon = dens * N
    rho = rhop + rhon
    rho0 = A * ws_density(0.0, A)
    f_prime = (0.33 * rho / rho0 + 0.45 * (1.0 - rho / rho0)) * c0

    kfp = (3.0 * PI2 * rhop) ** (1.0 / 3.0) * HBARC
    kfn = (3.0 * PI2 * rhon) ** (1.0 / 3.0) * HBARC
    kf1 = np.where(hit_is_proton, kfp, kfn)
    kf2 = np.where(hit_is_proton, kfn, kfp)
    kf = (1.5 * PI2 * rho) ** (1.0 / 3.0) * HBARC

    imU = rel_lindhard_im(q0t, dqt, kf1, kf2, M, is_neutrino)

    rel_lin = np.where(imU < 0.0, rel_lindhard(q0t, dqt, kf, M, is_neutrino), 0.0)
    udel = np.where(imU < 0.0, delta_lindhard(q0t, dqt, rho, kf), 0.0)
    rel_lin_tot = rel_lin + udel

    # CRho=2, mRho^2=0.5929, LambdaRho^2=6.25; f^2/4pi=0.08, g'=0.63,
    # LambdaPi^2=1.44 (all GeV^2)
    vt = 0.08 * 4.0 * PI / K_PION_MASS2 * (
        2.0 * ((6.25 - 0.5929) / (6.25 - q2)) ** 2 * dq2 / (q2 - 0.5929) + 0.63)
    vl = 0.08 * 4.0 * PI / K_PION_MASS2 * (
        ((1.44 - K_PION_MASS2) / (1.44 - q2)) ** 2 * dq2 / (q2 - K_PION_MASS2) + 0.63)

    CN = 1.0 / np.abs(1.0 - f_prime * rel_lin / HBARC2) ** 2
    CT = 1.0 / np.abs(1.0 - rel_lin_tot * vt) ** 2
    CL = 1.0 / np.abs(1.0 - rel_lin_tot * vl) ** 2
    return CN, CT, CL, imU


# ---------------------------------------------------------------------------
# Lepton-hadron tensor contraction (NievesQELCCPXSec::LmunuAnumu, SBN fork)
# ---------------------------------------------------------------------------

def _boost(p4, beta):
    """ROOT TLorentzVector::Boost. p4: (..., 4) as (E, px, py, pz);
    beta: (..., 3)."""
    E = p4[..., 0]
    p = p4[..., 1:]
    b2 = np.sum(beta * beta, axis=-1)
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = np.sum(beta * p, axis=-1)
    gamma2 = np.where(b2 > 0, (gamma - 1.0) / np.where(b2 > 0, b2, 1.0), 0.0)
    p_out = p + (gamma2 * bp + gamma * E)[..., None] * beta
    E_out = gamma * (E + bp)
    return np.concatenate([E_out[..., None], p_out], axis=-1)


def tensor_contraction(k4, kp4, q0t, dqt, kz, kpz,
                       m_ni, M, f1v, xif2v, fa, fp, CN, CT, CL, sign):
    """Re(L^{mu nu} A_{mu nu}) in the nucleon rest frame with qTilde along z.

    k4, kp4: neutrino / lepton four-vectors ALREADY boosted to the nucleon
    rest frame, shape (..., 4). kz, kpz: their momentum components along the
    boosted qTilde direction.

    The initial nucleon in this frame is exactly (m_ni, 0, 0, 0)
    (SBN-fork boost), so rulin = diag-like with only [0][0] = m_ni^2.
    """
    q2 = q0t ** 2 - dqt ** 2
    q02 = q0t ** 2
    dq2 = dqt ** 2
    M2 = M * M
    mNi2 = m_ni * m_ni

    F1V2 = f1v * f1v
    xiF2V2 = xif2v * xif2v
    FA2 = fa * fa
    Fp2 = fp * fp

    k0 = k4[..., 0]
    kp0 = kp4[..., 0]
    # Minkowski k'.k
    kpk = kp0 * k0 - np.sum(kp4[..., 1:] * k4[..., 1:], axis=-1)

    # leptonic tensor components in the rotated frame (z along qTilde):
    # transverse products are rotation-invariant around z:
    # k'1 k1 + k'2 k2 = k'_perp . k_perp
    kperp_dot = np.sum(kp4[..., 1:] * k4[..., 1:], axis=-1) - kpz * kz

    L00 = 2.0 * kp0 * k0 - kpk
    L33 = 2.0 * kpz * kz + kpk
    L03r = -(kp0 * kz + kpz * k0)              # Re L^{03}
    L11pL22 = kperp_dot * 2.0 + 2.0 * kpk      # L11 + L22
    # Im L^{12} = k'3 k0 - k'0 k3 (in rotated frame)
    imL12 = kpz * k0 - kp0 * kz

    pion_pole = 2.0 * CL * Fp2 * q2 + 8.0 * fa * fp * CL * M

    A00 = (16.0 * F1V2 * (2.0 * mNi2 * CN + 2.0 * q0t * m_ni + q2 / 2.0)
           + 2.0 * q2 * xiF2V2 * (4.0 - 4.0 * mNi2 / M2 - 4.0 * q0t * m_ni / M2
                                  - q02 * (4.0 / q2 + 1.0 / M2))
           + 4.0 * FA2 * (2.0 * mNi2 + 2.0 * q0t * m_ni + (q2 / 2.0 - 2.0 * M2))
           - pion_pole * q02
           - 16.0 * f1v * xif2v * (-q2 + q02) * CN)

    A03 = (16.0 * F1V2 * (m_ni * dqt * CN)
           + 2.0 * q2 * xiF2V2 * (-2.0 * dqt * m_ni / M2
                                  - dqt * q0t * (4.0 / q2 + 1.0 / M2))
           + 4.0 * FA2 * (dqt * m_ni * CL)
           - pion_pole * dqt * q0t
           - 16.0 * f1v * xif2v * dqt * q0t)

    A33 = (16.0 * F1V2 * (-q2 / 2.0)
           + 2.0 * q2 * xiF2V2 * (-4.0 - dq2 * (4.0 / q2 + 1.0 / M2))
           + 4.0 * FA2 * (-(q2 / 2.0 - 2.0 * CL * M2))
           - pion_pole * dq2
           - 16.0 * f1v * xif2v * (q2 + dq2))

    ATT = (16.0 * F1V2 * (-q2 / 2.0)
           + 2.0 * q2 * xiF2V2 * (-4.0 * CT)
           + 4.0 * FA2 * (-(q2 / 2.0 - 2.0 * CT * M2))
           - 16.0 * f1v * xif2v * CT * q2)   # A11 = A22

    # Im A^{12} (Amunu = i*a12, Anumu = -i*a12); contribution 2*a12*Im(L12)
    a12 = sign * 16.0 * fa * (xif2v + f1v) * (-dqt * m_ni * CT)

    return (L00 * A00 + 2.0 * L03r * A03 + L33 * A33 + L11pL22 * ATT
            + 2.0 * a12 * imL12)


# ---------------------------------------------------------------------------
# Full per-event evaluation with radius marginalization
# ---------------------------------------------------------------------------

class NievesQEXSec:
    """Evaluates the FA-dependent part of the Nieves CCQE differential cross
    section for numu CC QE events on Ar40, marginalized over the (unstored)
    hit-nucleon radius.

    Inputs per event (nu/lepton frame; z along the neutrino):
      enu       (N,)   neutrino energy
      p_lep     (N,3)  final lepton 3-momentum
      p_nf      (N,3)  final (pre-FSI) nucleon 3-momentum
    All events are assumed numu (hit neutron -> recoil proton); invalid /
    antineutrino events must be masked out by the caller.
    """

    def __init__(self, A=40, Z=18, N=22, n_r=16, rpa=True, coulomb=True,
                 m_lep=K_MUON_MASS):
        self.A, self.Z, self.N = A, Z, N
        self.rpa = rpa
        self.coulomb = coulomb
        self.m_lep = m_lep
        rmax = 3.0 * R0_FM * A ** (1.0 / 3.0)
        # Gauss-Legendre nodes over [0, rmax]; prior weight rho(r) r^2
        x, w = np.polynomial.legendre.leggauss(n_r)
        self.r_nodes = 0.5 * rmax * (x + 1.0)
        self.r_prior = 0.5 * rmax * w * ws_density(self.r_nodes, A) * self.r_nodes ** 2
        self.vc_nodes = coulomb_vcr(self.r_nodes, A, Z) if coulomb else np.zeros(n_r)

    def tensor_parts(self, enu, p_lep, p_nf, fa_models, is_neutrino=True):
        """Returns (tensors, common, valid):
          tensors: dict name -> (N, R) tensor contraction per FA model
          common:  (N, R) FA-independent factor (coulombFactor * Pauli * valid)
          valid:   (N,) event-level validity mask
        """
        n_ev = len(enu)
        m_ni = M_NEUTRON if is_neutrino else M_PROTON
        m_nf = M_PROTON if is_neutrino else M_NEUTRON
        M = 0.5 * (m_ni + m_nf)
        sign = 1.0 if is_neutrino else -1.0
        ml = self.m_lep
        ml2 = ml * ml

        # lab (nu-frame) four-vectors
        k4 = np.zeros((n_ev, 4))
        k4[:, 0] = enu
        k4[:, 3] = enu
        el = np.sqrt(ml2 + np.sum(p_lep ** 2, axis=1))
        kp4 = np.concatenate([el[:, None], p_lep], axis=1)
        e_nf = np.sqrt(m_nf ** 2 + np.sum(p_nf ** 2, axis=1))

        # struck nucleon: 3-momentum conservation with the TRUE q
        q3 = k4[:, 1:] - p_lep
        p_ni = p_nf - q3
        e_ni_on = np.sqrt(m_ni ** 2 + np.sum(p_ni ** 2, axis=1))
        p_ni4_on = np.concatenate([e_ni_on[:, None], p_ni], axis=1)

        # binding: energy transfer to the on-shell nucleon
        q0t = e_nf - e_ni_on                       # (N,) r-independent
        valid = q0t > 0.0

        # Coulomb-corrected lepton momentum per (event, radius)
        pl = np.sqrt(np.sum(p_lep ** 2, axis=1))
        if self.coulomb:
            vc = self.vc_nodes[None, :]                       # (1, R)
            el_local = el[:, None] - sign * vc                # (N, R)
            ok_c = (el_local - ml > 0.0) & ((el - ml)[:, None] > np.abs(vc))
            pl_local = np.sqrt(np.maximum(el_local ** 2 - ml2, 0.0))
            coulomb_factor = (pl_local * el_local) / (pl * el)[:, None]
        else:
            n_r = len(self.r_nodes)
            pl_local = np.broadcast_to(pl[:, None], (n_ev, n_r)).copy()
            ok_c = np.ones((n_ev, n_r), dtype=bool)
            coulomb_factor = np.ones((n_ev, n_r))

        # qTilde in the lab per (event, radius)
        lep_dir = p_lep / pl[:, None]
        q3_tilde = k4[:, None, 1:] - pl_local[..., None] * lep_dir[:, None, :]  # (N,R,3)
        dq_lab = np.linalg.norm(q3_tilde, axis=-1)
        q2tilde = q0t[:, None] ** 2 - dq_lab ** 2
        ok_q2 = (-q2tilde) > K_SMALL

        # Pauli blocking: recoil-species local Fermi momentum
        dens = ws_density(self.r_nodes, self.A)
        rho_recoil = dens * (self.Z if is_neutrino else self.N)
        kf_recoil = (3.0 * PI2 * rho_recoil) ** (1.0 / 3.0) * HBARC   # (R,)
        p_nf_mag = np.linalg.norm(p_nf, axis=1)
        ok_pauli = p_nf_mag[:, None] >= kf_recoil[None, :]

        # boost k, k', qTilde to the nucleon rest frame (beta r-independent)
        beta = -p_ni / e_ni_on[:, None]                       # (N, 3)
        kb = _boost(k4, beta)                                  # (N, 4)
        kpb = _boost(kp4, beta)
        qt4 = np.concatenate([
            np.broadcast_to(q0t[:, None, None], q3_tilde.shape[:2] + (1,)),
            q3_tilde], axis=-1)                                # (N, R, 4)
        qtb = _boost(qt4, beta[:, None, :])                    # (N, R, 4)

        q0t_rest = qtb[..., 0]
        dqt_rest = np.linalg.norm(qtb[..., 1:], axis=-1)
        ok_dq = dqt_rest > K_SMALL

        # components of k, k' along the boosted qTilde direction
        qhat = qtb[..., 1:] / np.where(ok_dq, dqt_rest, 1.0)[..., None]
        kz = np.sum(kb[:, None, 1:] * qhat, axis=-1)           # (N, R)
        kpz = np.sum(kpb[:, None, 1:] * qhat, axis=-1)

        # RPA coefficients at the nucleon-rest-frame qTilde
        CN, CT, CL, imU = cnctcl(
            q0t_rest, dqt_rest, self.r_nodes[None, :], M, is_neutrino,
            hit_is_proton=not is_neutrino, A=self.A, Z=self.Z, N=self.N)
        ok_imu = imU <= K_SMALL
        if not self.rpa:
            CN = np.ones_like(CN)
            CT = np.ones_like(CT)
            CL = np.ones_like(CL)

        ok = (valid[:, None] & ok_c & ok_q2 & ok_pauli & ok_dq & ok_imu)
        common = np.where(ok, coulomb_factor, 0.0)

        # form factors at q2 = -Q2tilde = q2tilde (invariant, (N, R))
        q2t_safe = np.where(ok_q2, q2tilde, -1.0)
        kb_b = np.broadcast_to(kb[:, None, :], qtb.shape)
        kpb_b = np.broadcast_to(kpb[:, None, :], qtb.shape)

        tensors = {}
        for name, fa_model in fa_models.items():
            f1v_ls, xif2v_ls, fa_ls, fp_ls = lwlyn_smith_ccff(
                q2t_safe, fa_model, m_ni)
            # Nieves rescaling of the Llewellyn-Smith FFs
            f1v = 0.5 * f1v_ls
            xif2v = 0.5 * xif2v_ls
            fa = -fa_ls
            fp = -fp_ls / M
            t = tensor_contraction(
                kb_b, kpb_b, q0t_rest, dqt_rest, kz, kpz,
                m_ni, M, f1v, xif2v, fa, fp, CN, CT, CL, sign)
            tensors[name] = np.where(ok, t, 0.0)

        return tensors, common, valid

    def weight_ratio(self, enu, p_lep, p_nf, fa_new, fa_old, is_neutrino=True):
        """Posterior-averaged xsec ratio new/old; 1.0 where undefined."""
        tensors, common, _ = self.tensor_parts(
            enu, p_lep, p_nf, {"new": fa_new, "old": fa_old}, is_neutrino)
        prior = self.r_prior[None, :] * common               # (N, R)
        num = np.sum(prior * tensors["new"], axis=1)
        den = np.sum(prior * tensors["old"], axis=1)
        good = (den > 0) & (num >= 0)
        return np.where(good, num / np.where(good, den, 1.0), 1.0)
