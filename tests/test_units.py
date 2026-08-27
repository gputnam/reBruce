"""Unit tests for the fake-data reweighter components.

Run from the repository root:  ./venv/bin/python -m pytest tests/ -v
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fakedata.bba07 import sachs_ffs, MU_P, MU_N                     # noqa: E402
from fakedata.bdt import GBReweighterJSON                            # noqa: E402
from fakedata.nieves import NievesQEXSec, ws_density                 # noqa: E402
from fakedata.tercile import wmode_weights, W_MODES                  # noqa: E402
from fakedata.tki import tki_vars, tki_ptx_pty                       # noqa: E402
from fakedata.xsec_table import (Table1D, load_t2k_table,            # noqa: E402
                                 load_ub_table, ub_observables)
from fakedata.zexp import COEFF_SETS, ZExpAxialFF                    # noqa: E402


# ---------------------------------------------------------------------------
# z-expansion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(COEFF_SETS))
def test_zexp_fa0(name):
    ff = ZExpAxialFF.from_name(name)
    assert abs(ff(0.0) - ff.ga) < 1e-12


def test_zexp_deut_matches_genie_defaults():
    # GENIE Default coefficient set; check FA at a few Q2 against a dipole
    # with MA ~ 1 GeV to ~15% (sanity, not exactness)
    ff = ZExpAxialFF.from_name("deuterium")
    for q2 in [-0.2, -0.5, -1.0]:
        dipole = -1.267 / (1.0 - q2 / 1.0) ** 2
        assert abs(ff(q2) / dipole - 1.0) < 0.15


# ---------------------------------------------------------------------------
# BBA07
# ---------------------------------------------------------------------------

def test_bba07_q2_zero_limits():
    gep, gmp, gen, gmn = sachs_ffs(-1e-12)
    assert abs(gep - 1.0) < 1e-6
    assert abs(gmp - MU_P) < 1e-6
    assert abs(gen) < 1e-6
    assert abs(gmn - MU_N) < 1e-6


# ---------------------------------------------------------------------------
# BDT inference (regression fixture generated from the reference
# json_reweighter implementation in PROfit/MEC-BDT-WGT)
# ---------------------------------------------------------------------------

def test_bdt_regression():
    with open(os.path.join(os.path.dirname(__file__), "bdt_fixture.json")) as f:
        fix = json.load(f)
    model = GBReweighterJSON(
        os.path.join(os.path.dirname(__file__), "..", "data",
                     "mec_bdt_susav2_to_valencia.json"))
    w = model.predict_weights(np.array(fix["features"]))
    assert np.array_equal(w, np.array(fix["weights"]))


def test_bdt_rejects_nonfinite():
    model = GBReweighterJSON(
        os.path.join(os.path.dirname(__file__), "..", "data",
                     "mec_bdt_susav2_to_valencia.json"))
    x = np.zeros((1, 8))
    x[0, 3] = np.nan
    with pytest.raises(ValueError):
        model.predict_weights(x)


# ---------------------------------------------------------------------------
# TKI
# ---------------------------------------------------------------------------

def test_tki_backtoback():
    # muon along +x, proton along -x, equal pT: deltaPT = 0
    vmu = np.array([[0.3, 0.0, 0.5]])
    vp = np.array([[-0.3, 0.0, 0.4]])
    dpt, dat, dphit = tki_vars(vmu, vp)
    assert abs(dpt[0]) < 1e-12
    # deltaPhiT = acos(-muT.hadT/...) = 0 for perfectly back-to-back
    assert abs(dphit[0]) < 1e-9


def test_tki_known_configuration():
    # muon pT along +x, hadron pT along +y (equal magnitude):
    # sumT at 45 deg; deltaAlphaT = angle between -muT and sumT = 135 deg
    vmu = np.array([[0.4, 0.0, 1.0]])
    vp = np.array([[0.0, 0.4, 0.5]])
    dpt, dat, dphit = tki_vars(vmu, vp)
    assert abs(dpt[0] - 0.4 * np.sqrt(2.0)) < 1e-12
    assert abs(dat[0] - 135.0) < 1e-9
    assert abs(dphit[0] - 90.0) < 1e-9
    ptx, pty = tki_ptx_pty(vmu, vp)
    # pty = -(muT.sumT)/|muT| = -0.4; ptx = (-muT_y sumT_x + muT_x sumT_y)/|muT|
    assert abs(pty[0] + 0.4) < 1e-12
    assert abs(ptx[0] - 0.4) < 1e-12


# ---------------------------------------------------------------------------
# Weight tables
# ---------------------------------------------------------------------------

def test_t2k_table_values():
    t = load_t2k_table()
    assert t.n_bins == 9
    # bin 2 of the release: costh [0.5,0.7], p [0.2,0.6], weight 1.6171
    x = np.array([0.3, 0.3, 5.0, 0.1])
    y = np.array([0.6, 0.95, 0.6, 0.6])
    w = t.weight(x, y)
    assert abs(w[0] - 1.6171) < 1e-9
    assert w[2] == 1.0   # outside measured p range
    assert w[3] == 1.0   # below measured p range


def test_ub_1d_table_padding():
    t = load_ub_table("ub_cc1p0pi_xsec.csv", "DeltaPT")
    assert t.n_bins == 13
    w = t.weight(np.array([-0.1, 0.02, 0.89, 0.95]))
    assert w[0] == 1.0            # below range
    assert abs(w[1] - 1.27403) < 1e-9
    assert abs(w[2] - 1.11825) < 1e-9
    assert w[3] == 1.0            # above range


def test_ub_2d_table():
    t = load_ub_table("ub_cc1p0pi_xsec.csv", "DeltaAlphaT_DeltaPT")
    assert t.n_bins == 21   # 3 DeltaPT slices x 7 DeltaAlphaT bins
    w = t.weight(np.array([0.5, 1.5]), np.array([170.0, 90.0]))
    assert w[0] != 1.0
    assert w[1] == 1.0      # DeltaPT outside slices


def test_ub_observable_lists():
    assert "DeltaPT" in ub_observables("ub_cc1p0pi_xsec.csv")
    assert "muon_mom" in ub_observables("ub_cc2p0pi_xsec.csv")
    assert "ThetaMuPi" in ub_observables("ub_ccpi_xsec.csv")


# ---------------------------------------------------------------------------
# W terciles
# ---------------------------------------------------------------------------

def test_tercile_closure():
    rng = np.random.default_rng(7)
    n = 3000
    bin_idx = rng.integers(-1, 3, n)
    bin_weights = np.array([1.4, 0.7, 1.1])
    W = rng.uniform(1.0, 2.0, n)
    w_valid = rng.random(n) < 0.8
    cv = rng.uniform(0.5, 1.5, n)

    nominal = np.where(bin_idx >= 0, bin_weights[np.maximum(bin_idx, 0)], 1.0)
    tot_nom = np.sum(cv * nominal)

    for mode in W_MODES:
        w = wmode_weights(bin_idx, 3, bin_weights, W, w_valid, cv, mode)
        # bin totals preserved
        for b in range(3):
            sel = bin_idx == b
            assert np.isclose(np.sum(cv[sel] * w[sel]),
                              bin_weights[b] * np.sum(cv[sel]))
        # outside bins untouched
        assert np.all(w[bin_idx < 0] == 1.0)
        # unknown-W events get the nominal bin weight
        unk = (bin_idx >= 0) & ~w_valid
        assert np.allclose(w[unk], nominal[unk])
        assert np.isclose(np.sum(cv * w), tot_nom)


def test_tercile_negative_clip():
    # strong downward bin weight: tercile weight would go negative, clip at 0
    n = 300
    rng = np.random.default_rng(3)
    bin_idx = np.zeros(n, dtype=int)
    W = rng.uniform(0.9, 2.1, n)
    cv = np.ones(n)
    w = wmode_weights(bin_idx, 1, np.array([0.5]), W, np.ones(n, bool), cv, "loW")
    assert w.min() == 0.0


def test_tercile_upper_clip():
    # huge upward bin weight: tercile weight capped at WEIGHT_CLIP[1]
    from fakedata.tercile import WEIGHT_CLIP
    n = 300
    rng = np.random.default_rng(5)
    bin_idx = np.zeros(n, dtype=int)
    W = rng.uniform(0.9, 2.1, n)
    cv = np.ones(n)
    # w_bin = 5 -> tercile weight ~ 1 + 3*4 = 13 > 10
    w = wmode_weights(bin_idx, 1, np.array([5.0]), W, np.ones(n, bool), cv, "loW")
    assert w.max() == WEIGHT_CLIP[1]


# ---------------------------------------------------------------------------
# Nieves
# ---------------------------------------------------------------------------

def _toy_qe_events(n=50, seed=11):
    rng = np.random.default_rng(seed)
    enu = rng.uniform(0.4, 2.5, n)
    # generate a plausible lepton: forward-ish, below enu
    el_frac = rng.uniform(0.3, 0.9, n)
    pl = enu * el_frac
    cth = rng.uniform(0.5, 1.0, n)
    sth = np.sqrt(1 - cth ** 2)
    phi = rng.uniform(0, 2 * np.pi, n)
    p_lep = np.stack([pl * sth * np.cos(phi), pl * sth * np.sin(phi), pl * cth], axis=1)
    # nucleon: q + fermi motion
    q3 = np.zeros_like(p_lep)
    q3[:, 2] = enu
    q3 -= p_lep
    pf = rng.uniform(0.0, 0.2, n)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    p_nf = q3 + pf[:, None] * u
    return enu, p_lep, p_nf


def test_nieves_identity_ratio():
    ff = ZExpAxialFF.from_name("deuterium")
    xs = NievesQEXSec(n_r=8)
    enu, p_lep, p_nf = _toy_qe_events()
    w = xs.weight_ratio(enu, p_lep, p_nf, ff, ff)
    assert np.allclose(w, 1.0)


def test_nieves_ratio_positive_finite():
    xs = NievesQEXSec(n_r=8)
    enu, p_lep, p_nf = _toy_qe_events()
    w = xs.weight_ratio(enu, p_lep, p_nf,
                        ZExpAxialFF.from_name("lqcd"),
                        ZExpAxialFF.from_name("minerva_nature"))
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)


def test_ws_density_normalized():
    # integral of rho r^2 4pi dr == 1
    r = np.linspace(0, 20, 4000)
    rho = ws_density(r, 40)
    integral = np.trapezoid(4 * np.pi * rho * r ** 2, r)
    assert abs(integral - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# hA2018 -> hA2025 pion FSI reweight
# ---------------------------------------------------------------------------

def test_ha2025_table_values():
    from fakedata.ha2025 import HA2025Reweighter, FATES
    rw = HA2025Reweighter()
    # first table row: KE=1 MeV, cex -> 0.4651438869139974 (from the CSV)
    w = rw.weight(np.array([1.0]), np.array([FATES.index("cex")]))
    assert abs(w[0] - 0.4651438869139974) < 1e-12
    # unmapped fate index -> exactly 1
    w = rw.weight(np.array([100.0, 500.0]), np.array([-1, -1]))
    assert np.all(w == 1.0)
    # clamping: below/above the table range uses the endpoints
    for f in range(4):
        lo = rw.weight(np.array([0.0]), np.array([f]))
        lo1 = rw.weight(np.array([1.0]), np.array([f]))
        assert lo[0] == lo1[0]


def test_ha2025_weights_reasonable():
    from fakedata.ha2025 import HA2025Reweighter
    rw = HA2025Reweighter()
    ke = np.linspace(1, 999, 500)
    for f in range(4):
        w = rw.weight(ke, np.full(500, f))
        assert np.all(np.isfinite(w))
        assert np.all(w >= 0)


class _StubSBruce:
    """Minimal SBruceFile stand-in for calculator.compute()-level tests."""

    def __init__(self, arrays):
        self._a = arrays
        self.n_entries = len(next(iter(arrays.values())))

    def arrays(self, branches):
        return {b: self._a[b] for b in branches}


# ---------------------------------------------------------------------------
# SPP low-Q2 enhancement x MINERvA Tpi suppression
# ---------------------------------------------------------------------------

def _landau_quadrature(z, umax=40.0, n=400001):
    """Landau density from its integral representation, for an independent
    check of the CERNLIB port: p(z) = 1/pi * int_0^inf e^{-u ln u - z u} sin(pi u) du.
    Composite Simpson on a fine grid (the integrand dies like u^-u)."""
    u = np.linspace(1e-14, umax, n)
    f = np.exp(-u * np.log(u) - z * u) * np.sin(np.pi * u)
    h = u[1] - u[0]
    return (h / 3.0) * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum()
                        + 2.0 * f[2:-2:2].sum()) / np.pi


def test_landau_pdf_matches_integral_representation():
    from fakedata.landau import landau_pdf
    # the range the Tpi fit actually probes is z in [-2.14, 1.79]; check wider
    for z in (-3.0, -2.5, -2.13494, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.79, 4.9):
        assert landau_pdf(z) == pytest.approx(_landau_quadrature(z), rel=1e-6)


def test_landau_is_tmath_convention():
    """TMath::Landau(x, mpv, sigma) with norm=kFALSE has no 1/sigma factor."""
    from fakedata.landau import landau, landau_pdf
    assert landau(0.3, 0.1, 0.05) == pytest.approx(landau_pdf((0.3 - 0.1) / 0.05))


def test_spp_q2_reweight_template():
    """Values straight off NuMIXSecSysts.cxx:94-118."""
    from fakedata.calculators.spp_lowq2 import spp_q2_reweight
    q2 = np.array([-0.1, 0.0, 0.024, 0.025, 0.075, 0.15, 0.65, 1.5, 2.5, 3.0, 50.0])
    want = np.array([1.253255, 1.253255, 1.253255, 1.589738, 1.733869, 1.651728,
                     1.475510, 1.048199, 1.650489, 1.650489, 1.650489])
    assert np.allclose(spp_q2_reweight(q2), want)


def test_spp_tpi_reweight_template_and_fit():
    from fakedata.calculators.spp_lowq2 import spp_tpi_reweight
    # above the 225 MeV cutoff the binned template is used verbatim
    tpi = np.array([0.24, 0.26, 0.31, 0.34, 0.45, 0.8, 1.5, 5.0])
    want = np.array([0.755932, 0.638574, 0.391947, 0.323265, 0.594541,
                     0.658024, 0.873622, 0.873622])
    assert np.allclose(spp_tpi_reweight(tpi), want)
    # the Landau fit is continuous with the template at the cutoff
    assert spp_tpi_reweight(0.2249) == pytest.approx(0.755932, rel=2e-3)
    # and is a suppression at low Tpi, peaking near the MPV
    assert spp_tpi_reweight(0.0) < 0.3
    assert spp_tpi_reweight(0.12235454) > 1.1


def _spp_event(n=1, **over):
    """One in-signal SPP event per branch set; override to break the signal."""
    a = {"genie_q0": np.full(n, 0.3), "genie_q3": np.full(n, 0.5),
         # post-FSI: exactly one charged pion, no pi0, no photon
         "true_npi": np.ones(n), "true_npi0": np.zeros(n),
         "true_g_p": np.full(n, -999.0), "true_cpi_p": np.full(n, 0.25),
         # pre-FSI: a charged pion, no pi0, no photon
         "genie_prefsi_cpi_px": np.zeros(n), "genie_prefsi_cpi_py": np.zeros(n),
         "genie_prefsi_cpi_pz": np.full(n, 0.25),
         "genie_prefsi_pi0_px": np.full(n, -999.0),
         "genie_prefsi_pi0_py": np.full(n, -999.0),
         "genie_prefsi_pi0_pz": np.full(n, -999.0),
         "genie_prefsi_g_px": np.full(n, -999.0),
         "genie_prefsi_g_py": np.full(n, -999.0),
         "genie_prefsi_g_pz": np.full(n, -999.0)}
    a.update({k: np.full(n, v) for k, v in over.items()})
    return _StubSBruce(a)


def _spp_weights(**over):
    from fakedata.calculators.spp_lowq2 import (POSTFSI_SUFFIX, PREFSI_SUFFIX,
                                                SPPLowQ2PiEnhancement)
    calc = SPPLowQ2PiEnhancement()
    out = calc.compute(_spp_event(**over))
    return out[calc.branch + POSTFSI_SUFFIX], out[calc.branch + PREFSI_SUFFIX]


def test_spp_in_signal_reproduces_reference_product():
    """Q2 = 0.5^2 - 0.3^2 = 0.16 -> 1.651728;  Tpi(p=0.25) = 0.1481 GeV."""
    from fakedata.calculators.spp_lowq2 import spp_tpi_reweight
    tpi = np.sqrt(0.25 ** 2 + 0.13957018 ** 2) - 0.13957018
    want = 1.651728 * float(spp_tpi_reweight(tpi))
    post, pre = _spp_weights()
    assert post[0] == pytest.approx(want)
    assert pre[0] == pytest.approx(want)


@pytest.mark.parametrize("over,post_unity,pre_unity", [
    ({"true_npi": 2.0}, True, False),                    # 2 FS pions
    ({"true_npi": 0.0, "true_cpi_p": -999.0}, True, False),  # pion absorbed
    ({"true_npi0": 1.0}, True, False),                   # FS pi0
    ({"true_g_p": 0.05}, True, False),                   # FS photon > 10 MeV
    ({"true_g_p": 0.005}, False, False),                 # FS photon < 10 MeV: kept
    ({"genie_prefsi_pi0_pz": 0.2, "genie_prefsi_pi0_px": 0.0,
      "genie_prefsi_pi0_py": 0.0}, False, True),         # pre-FSI pi0
    ({"genie_prefsi_g_pz": 0.05, "genie_prefsi_g_px": 0.0,
      "genie_prefsi_g_py": 0.0}, False, True),           # pre-FSI photon > 10 MeV
    ({"genie_prefsi_cpi_pz": -999.0, "genie_prefsi_cpi_px": -999.0,
      "genie_prefsi_cpi_py": -999.0}, False, True),      # no pre-FSI pion
    ({"genie_q0": -999.0}, True, True),                  # no Q2
])
def test_spp_signal_definitions(over, post_unity, pre_unity):
    post, pre = _spp_weights(**over)
    assert bool(post[0] == 1.0) == post_unity
    assert bool(pre[0] == 1.0) == pre_unity


def test_spp_weights_are_finite_and_unclipped():
    """Over the whole (Q2, Tpi) plane the product stays inside WEIGHT_CLIP."""
    from fakedata.calculators.spp_lowq2 import spp_q2_reweight, spp_tpi_reweight
    q2 = np.linspace(0.0, 10.0, 2001)
    tpi = np.linspace(0.0, 5.0, 2001)
    w = np.outer(spp_q2_reweight(q2), spp_tpi_reweight(tpi))
    assert np.all(np.isfinite(w))
    assert w.min() >= 0.0 and w.max() < 10.0
