"""BBA2007 elastic (Sachs) nucleon form factors and the Llewellyn-Smith CC
form-factor combination, ported from GENIE (SBNSoftware/Generator@v3_06_02_br,
BBA07ELFormFactorsModel with the "BBA07_25" coefficients, and LwlynSmithFF).

This is the vector-form-factor configuration of the AR23_20i tune
(CommonParam/ElasticFF: ElasticFormFactorsModel = BBA07, transverse
enhancement OFF). Reference: Bodek, Bradford, Budd, Avvakumov,
Eur.Phys.J.C53 (2008), arXiv:0708.1946.

q2 throughout is GENIE's spacelike q2 = -Q2 < 0 (GeV^2).
"""

import numpy as np

M_PROTON = 0.938272081   # GENIE kProtonMass
M_NEUTRON = 0.939565413  # GENIE kNeutronMass
M_PION = 0.13957018      # GENIE kPionMass (pi+/-)
MU_P = 2.7930            # AnomMagnMoment-P (AR23 CommonParam/MagnMoments)
MU_N = -1.913042         # AnomMagnMoment-N

# BBA07 "Default" (BBA07_25) coefficients
GEP_A1, GEP_B = -0.24, (10.98, 12.82, 21.97)
GMP_A1, GMP_B = 0.1717, (11.26, 19.32, 8.33)
GEP_P = (1.0000, 0.9927, 0.9898, 0.9975, 0.9812, 0.9340, 1.0000)
GMP_P = (1.0000, 1.0011, 0.9992, 0.9974, 1.0010, 1.0003, 1.0000)
GEN_P = (1.0000, 1.1011, 1.1392, 1.0203, 1.1093, 1.5429, 0.9706)
GMN_P = (1.0000, 0.9958, 0.9877, 1.0193, 1.0350, 0.9164, 0.7300)

_NODES = np.arange(7) / 6.0


def _an(x, coeffs):
    """7-node Lagrange interpolating polynomial on x = 0, 1/6, ..., 1."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    for i in range(7):
        num = np.ones_like(x)
        den = 1.0
        for j in range(7):
            if j == i:
                continue
            num = num * (x - _NODES[j])
            den = den * (_NODES[i] - _NODES[j])
        out = out + coeffs[i] * num / den
    return out


def _kelly(t, a1, b):
    return (1.0 + a1 * t) / (1.0 + t * (b[0] + t * (b[1] + b[2] * t)))


def sachs_ffs(q2):
    """Return (Gep, Gmp, Gen, Gmn) at spacelike q2 (< 0)."""
    tp = -q2 / (4.0 * M_PROTON ** 2)
    xp = 2.0 / (1.0 + np.sqrt(1.0 + 1.0 / tp))
    tn = -q2 / (4.0 * M_NEUTRON ** 2)
    xn = 2.0 / (1.0 + np.sqrt(1.0 + 1.0 / tn))

    gep = _an(xp, GEP_P) * _kelly(tp, GEP_A1, GEP_B)
    gmp = MU_P * _an(xp, GMP_P) * _kelly(tp, GMP_A1, GMP_B)
    gen = _an(xn, GEN_P) * gep * 1.7 * tn / (1.0 + 3.3 * tn)
    gmn = (MU_N / MU_P) * _an(xn, GMN_P) * gmp
    return gep, gmp, gen, gmn


def lwlyn_smith_ccff(q2, fa_model, m_ni):
    """Llewellyn-Smith CC form factors (GENIE LwlynSmithFF conventions).

    q2: spacelike (< 0); fa_model: callable FA(q2) (e.g. ZExpAxialFF);
    m_ni: on-shell hit-nucleon mass (GENIE uses it in tau and Fp).

    Returns (F1V, xiF2V, FA, Fp) BEFORE the Nieves-model rescaling
    (which applies F1V/2, xiF2V/2, -FA, -Fp/M).
    """
    tau = q2 / (4.0 * m_ni ** 2)
    gep, gmp, gen, gmn = sachs_ffs(q2)
    gve = gep - gen
    gvm = gmp - gmn
    f1v = (gve - tau * gvm) / (1.0 - tau)
    xif2v = (gvm - gve) / (1.0 - tau)
    fa = fa_model(q2)
    fp = 2.0 * m_ni ** 2 * fa / (M_PION ** 2 - q2)
    return f1v, xif2v, fa, fp
