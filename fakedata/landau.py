"""Landau probability density, ported from ROOT.

Numpy port of ``ROOT::Math::landau_pdf`` (CERNLIB G110 ``denlan``; the same
algorithm GSL uses), needed because ``GetSPPTpiMINERvAFittedReweight`` in
sbnana's NuMIXSecSysts.cxx is defined in terms of ``TMath::Landau`` and the
runtime environment has neither ROOT nor scipy.

``TMath::Landau(x, mpv, sigma)`` with its default ``norm = kFALSE`` returns
``landau_pdf((x - mpv) / sigma)`` -- with no 1/sigma factor -- so `landau`
below reproduces that call exactly.
"""

import numpy as np

# CERNLIB denlan rational-approximation coefficients (ROOT PdfFuncMathCore.cxx)
_P1 = (0.4259894875, -0.1249762550, 0.03984243700, -0.006298287635, 0.001511162253)
_Q1 = (1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063)

_P2 = (0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211)
_Q2 = (1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714)

_P3 = (0.1788544503, 0.09359161662, 0.006325387654, 0.00006611667319, -0.000002031049101)
_Q3 = (1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675)

_P4 = (0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186)
_Q4 = (1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511)

_P5 = (1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910)
_Q5 = (1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357)

_P6 = (1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109)
_Q6 = (1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939)

_A1 = (0.04166666667, -0.01996527778, 0.02709538966)
_A2 = (-1.845568670, -4.284640743)


def _poly4(c, x):
    """c[0] + (c[1] + (c[2] + (c[3] + c[4]*x)*x)*x)*x -- Horner, as in ROOT."""
    return c[0] + (c[1] + (c[2] + (c[3] + c[4] * x) * x) * x) * x


def _ratio(p, q, x):
    return _poly4(p, x) / _poly4(q, x)


def landau_pdf(v):
    """The Landau density at v (array or scalar), xi = 1, x0 = 0.

    Elementwise port of ROOT::Math::landau_pdf; every branch of the original
    piecewise approximation is evaluated on its own slice of the input.
    """
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape, dtype=np.float64)

    # v < -5.5 : exponential tail
    m = v < -5.5
    if np.any(m):
        vv = v[m]
        u = np.exp(vv + 1.0)
        d = (0.3989422803 * np.exp(-1.0 / u - 0.5 * (vv + 1.0))
             * (1.0 + (_A1[0] + (_A1[1] + _A1[2] * u) * u) * u))
        out[m] = np.where(u < 1e-10, 0.0, d)

    m = (v >= -5.5) & (v < -1.0)
    if np.any(m):
        vv = v[m]
        out[m] = np.exp(-np.exp(-vv - 1.0) - 0.5 * (vv + 1.0)) * _ratio(_P1, _Q1, vv)

    m = (v >= -1.0) & (v < 1.0)
    if np.any(m):
        out[m] = _ratio(_P2, _Q2, v[m])

    m = (v >= 1.0) & (v < 5.0)
    if np.any(m):
        out[m] = _ratio(_P3, _Q3, v[m])

    for lo, hi, p, q in ((5.0, 12.0, _P4, _Q4),
                         (12.0, 50.0, _P5, _Q5),
                         (50.0, 300.0, _P6, _Q6)):
        m = (v >= lo) & (v < hi)
        if np.any(m):
            u = 1.0 / v[m]
            out[m] = u * u * _ratio(p, q, u)

    # v >= 300 : asymptotic tail
    m = v >= 300.0
    if np.any(m):
        vv = v[m]
        u = 1.0 / (vv - vv * np.log(vv) / (vv + 1.0))
        out[m] = u * u * (1.0 + (_A2[0] + _A2[1] * u) * u)

    return out


def landau(x, mpv, sigma):
    """TMath::Landau(x, mpv, sigma) with the default norm = kFALSE."""
    if sigma <= 0:
        return np.zeros(np.shape(x), dtype=np.float64)
    return landau_pdf((np.asarray(x, dtype=np.float64) - mpv) / sigma)
