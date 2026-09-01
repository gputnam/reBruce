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
from fakedata.tki import (E_BNB, E_LE_MINERVA, minerva_pz_pt,        # noqa: E402
                          sig_minerva_qelike, sum_tp, tki_ptx_pty,
                          tki_vars)
from fakedata.xsec_table import (Table1D, TableGrid3D,               # noqa: E402
                                 load_minerva_3dqelike_table,
                                 load_t2k_table, load_ub_table,
                                 ub_observables)
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


# ---------------------------------------------------------------------------
# MINERvA LE/ME 3D QE-like table and signal definition
# ---------------------------------------------------------------------------

def _grid_2x2x2():
    # weights flattened in C order (x slowest, z fastest): index = 4i + 2j + k
    return TableGrid3D([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
                       np.arange(8, dtype=float))


def test_grid3d_interior_lookup():
    t = _grid_2x2x2()
    assert t.n_bins == 8 and t.shape == (2, 2, 2)
    for (i, j, k) in [(0, 0, 0), (1, 0, 1), (1, 1, 1), (0, 1, 0)]:
        x = np.array([i + 0.5]), np.array([j + 0.5]), np.array([k + 0.5])
        assert t.bin_index(*x)[0] == 4 * i + 2 * j + k
        assert t.weight(*x)[0] == 4 * i + 2 * j + k


def test_grid3d_out_of_range_each_axis():
    t = _grid_2x2x2()
    mid = np.array([0.5])
    # below the first edge and at/above the last edge, one axis at a time
    for lo, hi in [(-0.1, 2.0), (-0.1, 2.5)]:
        assert t.weight(np.array([lo]), mid, mid)[0] == 1.0
        assert t.weight(mid, np.array([lo]), mid)[0] == 1.0
        assert t.weight(mid, mid, np.array([lo]))[0] == 1.0
        assert t.weight(np.array([hi]), mid, mid)[0] == 1.0
        assert t.weight(mid, np.array([hi]), mid)[0] == 1.0
        assert t.weight(mid, mid, np.array([hi]))[0] == 1.0
    assert t.bin_index(np.array([-1.0]), mid, mid)[0] == -1
    # the last edge is exclusive, the first inclusive
    assert t.weight(np.array([0.0]), mid, mid)[0] == 0.0


def test_minerva_table_loads():
    t = load_minerva_3dqelike_table()
    assert t.shape == (5, 9, 10) and t.n_bins == 450
    np.testing.assert_allclose(t.x_edges, [1.5, 2.0, 2.5, 3.0, 3.5, 4.5])
    np.testing.assert_allclose(t.z_edges[-1], 0.799)
    # spot-check against MvA_LE_ME/results/weights_ptpzsumtp.csv
    def w(ipz, ipt, itp):
        return t.weights[(ipz * 9 + ipt) * 10 + itp]
    assert abs(w(0, 0, 0) - 2.056030984517805) < 1e-12
    assert w(0, 0, 2) == 0.0                       # clipped from -1.605
    assert abs(w(0, 1, 8) - 2.01285432734704) < 1e-12
    assert np.all(t.weights >= 0.0) and np.all(t.weights <= 10.0)
    assert np.count_nonzero(t.weights == 0.0) == 82
    assert np.count_nonzero(t.weights == 10.0) == 4


def test_minerva_excluded_bins_are_unity():
    import csv as _csv
    import os as _os
    path = _os.path.join(_os.path.dirname(__file__), "..", "data",
                         "minerva_3dqelike_bnb.csv")
    with open(path) as f:
        rows = [r for r in _csv.DictReader(x for x in f
                                           if not x.startswith("#"))]
    excluded = [r for r in rows if r["excluded"] == "True"]
    assert len(excluded) == 11
    assert all(float(r["weight"]) == 1.0 for r in excluded)


def test_minerva_pz_scaling():
    # p_z is treated relative to the neutrino energy: a measured edge p_z
    # corresponds at BNB to E_BNB * (p_z / E_LE)
    scale = E_BNB / E_LE_MINERVA
    assert abs(scale - 0.21538461538) < 1e-9
    assert abs(1.5 * scale - 0.32307692) < 1e-7
    # an event at the scaled first edge lands back on the measured first edge
    a = {"true_mu_p": np.array([1.5 * scale]),
         "true_mu_dir_x": np.array([0.0]), "true_mu_dir_y": np.array([0.0]),
         "true_mu_dir_z": np.array([1.0])}
    pz, pt = minerva_pz_pt(a, scale)
    assert abs(pz[0] - 1.5) < 1e-12 and pt[0] == 0.0


def test_sum_tp():
    a = {"true_p_p": np.array([0.5, 0.5, -999.0]),
         "true_p2_p": np.array([-999.0, 0.5, -999.0])}
    tp = sum_tp(a)
    one = np.sqrt(0.5 ** 2 + 0.938272081 ** 2) - 0.938272081
    assert abs(tp[0] - one) < 1e-12          # unfilled second proton -> 0
    assert abs(tp[1] - 2 * one) < 1e-12
    assert tp[2] == 0.0                      # no protons -> first bin


def _minerva_event(**over):
    scale = E_BNB / E_LE_MINERVA
    ev = {"true_isnc": 0, "true_pdg": 14,
          "true_mu_p": 0.6, "true_mu_dir_x": 0.0, "true_mu_dir_y": 0.05,
          "true_mu_dir_z": np.sqrt(1 - 0.05 ** 2),
          "true_p_p": 0.5, "true_p2_p": -999.0, "true_np": 1,
          "true_npi": 0, "true_npi0": 0, "true_g_p": -999.0}
    ev.update(over)
    return {k: np.array([v], dtype=float) for k, v in ev.items()}, scale


def test_sig_minerva_qelike_accepts_signal():
    a, scale = _minerva_event()
    assert sig_minerva_qelike(a, scale)[0]


@pytest.mark.parametrize("over", [
    {"true_isnc": 1},                    # NC
    {"true_pdg": -14},                   # numubar
    {"true_npi": 1},                     # charged pion
    {"true_npi0": 1},                    # pi0
    {"true_g_p": 0.020},                 # photon above 10 MeV
    {"true_mu_p": -999.0},               # unfilled muon
    {"true_mu_dir_y": 0.9,               # outside the scaled 20 deg cone
     "true_mu_dir_z": np.sqrt(1 - 0.9 ** 2), "true_mu_p": 1.5},
])
def test_sig_minerva_qelike_rejects(over):
    a, scale = _minerva_event(**over)
    assert not sig_minerva_qelike(a, scale)[0]


def test_sig_minerva_qelike_keeps_soft_photon():
    a, scale = _minerva_event(true_g_p=0.005)
    assert sig_minerva_qelike(a, scale)[0]


# ---------------------------------------------------------------------------
# MINERvA p_z-marginalized weight
# ---------------------------------------------------------------------------

class _StubSBruce:
    """Minimal SBruceFile stand-in for calculator.compute()-level tests."""

    def __init__(self, arrays):
        self._a = arrays
        self.n_entries = len(next(iter(arrays.values())))

    def arrays(self, branches):
        return {b: self._a[b] for b in branches}

    def has_branch(self, name):
        return name in self._a


def _minerva_sample(rng, n=4000):
    """Synthetic events spread over the measured (p_z, p_T, SumT_p) region
    and outside it, in the SCALED frame (so p_z is small at BNB)."""
    from fakedata.tki import E_BNB, E_LE_MINERVA
    scale = E_BNB / E_LE_MINERVA
    pz = rng.uniform(0.1, 1.2, n)               # scaled: 0.323-0.969 measured
    pt = rng.uniform(0.0, 0.5, n)
    p = np.hypot(pz, pt)
    a = {"true_isnc": np.zeros(n), "true_pdg": np.full(n, 14.0),
         "true_mu_p": p, "true_mu_dir_x": np.zeros(n),
         "true_mu_dir_y": pt / p, "true_mu_dir_z": pz / p,
         "true_p_p": rng.uniform(0.0, 0.9, n), "true_p2_p": np.full(n, -999.0),
         "true_np": np.ones(n), "true_npi": np.zeros(n),
         "true_npi0": np.zeros(n), "true_g_p": np.full(n, -999.0),
         "cvwgt": rng.uniform(0.5, 1.5, n), "genie_W": np.full(n, -999.0)}
    return _StubSBruce(a), scale


def test_minerva_marginalized_reproduces_3d_yield_per_cell():
    """The spectrum-weighted p_z average must preserve, in every
    (p_T, SumT_p) cell, the total yield the 3D weight lookup would give over
    this branch's own in-p_z-range population (no-theta signal)."""
    from fakedata.tki import sig_minerva_qelike
    from fakedata.calculators.minerva_qelike import MARG_SUFFIX, MINERvA3DQELike
    calc = MINERvA3DQELike()
    sb, _ = _minerva_sample(np.random.default_rng(7))
    wm = calc.compute(sb)[calc.branch + MARG_SUFFIX]

    a = sb.arrays(calc.branches_needed())
    cv = a["cvwgt"]
    table = calc.load_table()
    sig = sig_minerva_qelike(a, calc.pz_scale, theta_cut=False)
    obs = calc.observable(a)
    ipz, ipt, itp = table.axis_indices(*(np.where(sig, o, -1e9) for o in obs))
    in3d = (ipz >= 0) & (ipt >= 0) & (itp >= 0)
    assert in3d.sum() > 100, "test sample must populate the measured region"
    w3d = table.weights.reshape(table.shape)

    for j in range(table.shape[1]):
        for k in range(table.shape[2]):
            m = in3d & (ipt == j) & (itp == k)
            if not m.any():
                continue
            lookup = w3d[ipz[m], ipt[m], itp[m]]
            np.testing.assert_allclose(np.sum(cv[m] * wm[m]),
                                       np.sum(cv[m] * lookup), rtol=1e-12)


def test_minerva_marginalized_drops_theta_cut():
    """The marginalized branch reaches large-angle muons the 3D branch's
    theta < 20 deg cut removes; the 3D branch still applies it."""
    from fakedata.tki import sig_minerva_qelike
    from fakedata.calculators.minerva_qelike import MARG_SUFFIX, MINERvA3DQELike
    calc = MINERvA3DQELike()
    sb, _ = _minerva_sample(np.random.default_rng(3))
    out = calc.compute(sb)
    a = sb.arrays(calc.branches_needed())

    with_th = sig_minerva_qelike(a, calc.pz_scale)
    no_th = sig_minerva_qelike(a, calc.pz_scale, theta_cut=False)
    assert np.all(with_th <= no_th)
    fails_theta = no_th & ~with_th
    assert fails_theta.sum() > 0, "sample must contain large-angle muons"

    # those events can only ever be weighted by the marginalized branch
    assert np.all(out[calc.branch][fails_theta] == 1.0)
    assert np.any(out[calc.branch + MARG_SUFFIX][fails_theta] != 1.0)


def test_minerva_marginalized_applies_outside_pz_window():
    """The marginalized weight is applied at any p_z, so it reaches strictly
    more events than the 3D weight, and is 1.0 outside the (p_T, SumT_p)
    measured region."""
    from fakedata.calculators.minerva_qelike import MARG_SUFFIX, MINERvA3DQELike
    calc = MINERvA3DQELike()
    sb, _ = _minerva_sample(np.random.default_rng(11))
    out = calc.compute(sb)
    w3, wm = out[calc.branch], out[calc.branch + MARG_SUFFIX]

    from fakedata.tki import sig_minerva_qelike
    a = sb.arrays(calc.branches_needed())
    table = calc.load_table()
    obs = calc.observable(a)
    marg_sig = sig_minerva_qelike(a, calc.pz_scale, theta_cut=False)
    imz, imt, imk = table.axis_indices(
        *(np.where(marg_sig, o, -1e9) for o in obs))
    in2d = (imt >= 0) & (imk >= 0)
    ipz, ipt, itp = table.axis_indices(
        *(np.where(calc.signal_mask(a), o, -1e9) for o in obs))
    in3d = (ipz >= 0) & (ipt >= 0) & (itp >= 0)

    assert in2d.sum() > in3d.sum()          # reaches beyond the p_z window
    assert np.all(wm[~in2d] == 1.0)
    assert np.all(w3[~in3d] == 1.0)
    # every weight stays inside the table's own range
    assert np.all(wm >= 0.0) and np.all(wm <= 10.0)


def test_minerva_marginalized_empty_cell_is_unity():
    """A (p_T, SumT_p) cell with no in-p_z-range events falls back to 1.0."""
    from fakedata.calculators.minerva_qelike import MARG_SUFFIX, MINERvA3DQELike
    calc = MINERvA3DQELike()
    n = 50
    # all muons well below the scaled p_z window -> no bin has a spectrum
    p = np.full(n, 0.05)
    a = {"true_isnc": np.zeros(n), "true_pdg": np.full(n, 14.0),
         "true_mu_p": p, "true_mu_dir_x": np.zeros(n),
         "true_mu_dir_y": np.zeros(n), "true_mu_dir_z": np.ones(n),
         "true_p_p": np.full(n, 0.3), "true_p2_p": np.full(n, -999.0),
         "true_np": np.ones(n), "true_npi": np.zeros(n),
         "true_npi0": np.zeros(n), "true_g_p": np.full(n, -999.0),
         "cvwgt": np.ones(n), "genie_W": np.full(n, -999.0)}
    out = calc.compute(_StubSBruce(a))
    assert np.all(out[calc.branch + MARG_SUFFIX] == 1.0)
    assert np.all(out[calc.branch] == 1.0)


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


# ---------------------------------------------------------------------------
# Branch declarations and preflight
# ---------------------------------------------------------------------------

def _all_calcs():
    """One instance of every registered calculator, with default options."""
    import fakedata.calculators  # noqa: F401  (importing registers them)
    from fakedata.calculator import REGISTRY

    return [cls() for cls in REGISTRY.values()]


def _full_branch_stub(n=8, **overrides):
    """A _StubSBruce carrying every branch the default calculators declare."""
    a = {b: np.full(n, -999.0) for c in _all_calcs()
         for b in c.branches_needed()}
    a.update(overrides)
    return _StubSBruce(a)


def test_every_registered_calculator_declares_branches():
    calcs = _all_calcs()
    assert len(calcs) == 9
    for c in calcs:
        b = c.branches_needed()
        assert isinstance(b, list) and b, c.type_name
        assert all(isinstance(x, str) for x in b), c.type_name


def test_branches_needed_is_deduplicated_and_stable():
    for c in _all_calcs():
        b = c.branches_needed()
        assert len(set(b)) == len(b), c.type_name
        # calling twice must not accumulate
        assert c.branches_needed() == b, c.type_name


def test_branches_needed_returns_a_copy():
    """A caller doing `branches_needed() + [...]` must not mutate a constant."""
    from fakedata.tki import POSTFSI_BRANCHES

    before = list(POSTFSI_BRANCHES)
    for c in _all_calcs():
        c.branches_needed().append("__scribble__")
    assert POSTFSI_BRANCHES == before
    for c in _all_calcs():
        assert "__scribble__" not in c.branches_needed()


def test_xsec_calculators_declare_cvwgt_and_genie_w():
    """compute() loads both unconditionally, so both must be declared."""
    from fakedata.calculator import REGISTRY

    for name in ("ub_cc1p0pi", "ub_cc2p0pi", "ub_ccpi", "t2k_nc1pi",
                 "minerva_3dqelike"):
        needed = set(REGISTRY[name]().branches_needed())
        assert {"cvwgt", "genie_W"} <= needed, name


def test_divide_out_ff_declares_qe_branches():
    """Regression: divide_out_ff routes through deut_to_minerva_weight(),
    which reads QE_BRANCHES that no subclass declared before."""
    from fakedata.calculators.qe_zexp import QE_BRANCHES
    from fakedata.calculators.xsec_meas import UBCC1p0pi

    assert not set(QE_BRANCHES) <= set(UBCC1p0pi().branches_needed())
    assert set(QE_BRANCHES) <= set(
        UBCC1p0pi(divide_out_ff=True).branches_needed())


class _RecordingSBruce(_StubSBruce):
    """Stub that fails the test if compute() reads an undeclared branch."""

    def __init__(self, arrays, allowed):
        super().__init__(arrays)
        self._allowed = set(allowed)

    def arrays(self, branches):
        extra = [b for b in branches if b not in self._allowed]
        assert not extra, f"compute() read undeclared branches: {extra}"
        return super().arrays(branches)


@pytest.mark.parametrize("type_name", [
    "mec_bdt", "qe_zexp_mva_to_lqcd", "pi_fsi_ha2025",
    "jaesung_lowq2_pi_enhancement", "ub_cc1p0pi", "ub_cc2p0pi", "ub_ccpi",
    "t2k_nc1pi", "minerva_3dqelike",
])
def test_compute_reads_only_declared_branches(type_name):
    """The anti-drift test: what compute() loads must be what it declares.

    Every branch is at the -999 sentinel, so no calculator has any signal
    event; that exercises the load path without needing physical samples.
    """
    from fakedata.calculator import REGISTRY

    calc = REGISTRY[type_name]()
    needed = calc.branches_needed()
    sb = _RecordingSBruce({b: np.full(8, -999.0) for b in needed}, needed)
    out = calc.compute(sb)
    for w in out.values():
        assert len(w) == 8 and np.all(np.isfinite(w))


def test_compute_reads_only_declared_branches_divide_out_ff():
    from fakedata.calculators.xsec_meas import UBCC1p0pi

    calc = UBCC1p0pi(divide_out_ff=True)
    needed = calc.branches_needed()
    sb = _RecordingSBruce({b: np.full(8, -999.0) for b in needed}, needed)
    calc.compute(sb)


def test_check_branches_all_present():
    from fakedata.calculator import blocked, check_branches

    calcs = _all_calcs()
    report = check_branches(_full_branch_stub(), calcs)
    assert len(report) == len(calcs)
    assert all(missing == [] for _, missing in report)
    assert blocked(report) == []


def _drop(*branches):
    """A stub missing the named branches, otherwise complete."""
    sb = _full_branch_stub()
    for b in branches:
        del sb._a[b]
    return sb


def test_check_branches_reports_missing_in_declared_order():
    from fakedata.calculator import blocked, check_branches

    calcs = _all_calcs()
    sb = _drop("genie_prefsi_n_px", "true_cpi_p")
    report = check_branches(sb, calcs)
    bad = {c.type_name: missing for c, missing in blocked(report)}
    assert set(bad) == {"qe_zexp_mva_to_lqcd", "jaesung_lowq2_pi_enhancement",
                        "ub_ccpi", "t2k_nc1pi"}
    assert bad["qe_zexp_mva_to_lqcd"] == ["genie_prefsi_n_px"]
    assert bad["t2k_nc1pi"] == ["true_cpi_p"]
    # each missing list is a subsequence of that calculator's declaration
    for c, missing in report:
        decl = c.branches_needed()
        assert missing == [b for b in decl if b in set(missing)]


def test_check_branches_cvwgt_blocks_nearly_everything():
    """cvwgt is read by every calculator except the three that need no
    per-file normalization or W-tercile population."""
    from fakedata.calculator import blocked, check_branches

    report = check_branches(_drop("cvwgt"), _all_calcs())
    runnable = {c.type_name for c, missing in report if not missing}
    assert runnable == {"pi_fsi_ha2025", "jaesung_lowq2_pi_enhancement",
                        "qe_zexp_mva_to_lqcd"}
    assert len(blocked(report)) == 6


def test_format_branch_report_names_branches_and_calculators():
    from fakedata.calculator import check_branches, format_branch_report

    calcs = _all_calcs()
    txt = format_branch_report(
        check_branches(_drop("genie_prefsi_n_px", "true_cpi_p"), calcs),
        "/some/file.root")
    assert "MISSING" in txt
    assert "genie_prefsi_n_px" in txt and "true_cpi_p" in txt
    assert "qe_zexp_mva_to_lqcd" in txt
    assert "blocked calculators (4 of 9)" in txt
    assert "/some/file.root" in txt

    ok = format_branch_report(check_branches(_full_branch_stub(), calcs), "f")
    assert "all present" in ok and "MISSING" not in ok


def test_sbruce_errors_are_readable(tmp_path):
    import uproot

    from fakedata import ReBruceError
    from fakedata.sbruce import SBruceError, SBruceFile

    assert issubclass(SBruceError, ReBruceError)

    with pytest.raises(SBruceError, match="not a readable ROOT file"):
        SBruceFile(os.path.join(os.path.dirname(__file__), "bdt_fixture.json"))

    with pytest.raises(SBruceError, match="cannot open input file"):
        SBruceFile(str(tmp_path / "does_not_exist.root"))

    other = str(tmp_path / "other.root")
    with uproot.recreate(other) as f:
        f["Events"] = {"x": np.arange(4, dtype=np.float64)}
    with pytest.raises(SBruceError, match="has no 'SelectedEvents' tree"):
        SBruceFile(other)


# ---------------------------------------------------------------------------
# output: cv/ps1 dial format
# ---------------------------------------------------------------------------

_DIAL_WEIGHTS = {
    "fdwgt_alpha": np.array([1.0, 0.5, 1.25, 2.0, 1.0]),
    "fdwgt_beta": np.array([0.75, 1.0, 1.0, 1.0, 3.5]),
}


def _make_input(tmp_path, n=5):
    """A minimal stand-in for an sBruce file: SelectedEvents with n entries."""
    import uproot

    path = str(tmp_path / "in.root")
    with uproot.recreate(path) as f:
        f.mktree("SelectedEvents", {"cvwgt": np.float64}, title="t")
        f["SelectedEvents"].extend({"cvwgt": np.ones(n)})
    return path


def _read_dials(path):
    """{branch: (knots, sigma)} as float64 (n, 2) arrays, either writer."""
    import awkward as ak
    import uproot

    from fakedata.output import SIGMA_SUFFIX

    out = {}
    with uproot.open(path) as f:
        tree = f["fakedataTree"]
        for key in tree.keys():
            if key.endswith(SIGMA_SUFFIX) or key.startswith("n"):
                continue
            rows = ak.to_numpy(ak.to_regular(tree[key].array()))
            sigma = ak.to_numpy(ak.to_regular(tree[key + SIGMA_SUFFIX].array()))
            out[key] = (np.asarray(rows, np.float64),
                        np.asarray(sigma, np.float64))
    return out


def test_dial_name_carries_routing_keyword():
    from fakedata.output import DIAL_PREFIX, dial_name

    assert dial_name("fdwgt_mec_bdt") == "multisigma_fdwgt_mec_bdt"
    # MakesBruceNew.C routes weight branches on this substring
    assert "multisigma" in DIAL_PREFIX
    assert "multisigma" in dial_name("fdwgt_mec_bdt")


def test_build_dials_puts_cv_first():
    from fakedata.output import build_dials

    dials = build_dials(_DIAL_WEIGHTS, 5)
    assert set(dials) == {"multisigma_fdwgt_alpha", "multisigma_fdwgt_beta"}
    for name, block in dials.items():
        assert block.shape == (5, 2)
        assert np.all(block[:, 0] == 1.0)
    np.testing.assert_array_equal(dials["multisigma_fdwgt_alpha"][:, 1],
                                  _DIAL_WEIGHTS["fdwgt_alpha"])


def test_write_output_dial_format(tmp_path):
    import uproot

    from fakedata.output import write_output

    n = 5
    src = _make_input(tmp_path, n)
    dst = str(tmp_path / "out.root")
    write_output(src, dst, _DIAL_WEIGHTS, n)

    with uproot.open(dst) as f:
        # the copied trees survive untouched
        assert f["SelectedEvents"].num_entries == n
        tree = f["fakedataTree"]
        assert tree.num_entries == n
        keys = set(tree.keys())
    # the flat scalar branches are gone -- replaced, not kept alongside
    assert not any(k.startswith("fdwgt_") for k in keys)
    assert {"multisigma_fdwgt_alpha", "multisigma_fdwgt_alpha_sigma",
            "multisigma_fdwgt_beta", "multisigma_fdwgt_beta_sigma"} <= keys

    dials = _read_dials(dst)
    assert set(dials) == {"multisigma_fdwgt_alpha", "multisigma_fdwgt_beta"}
    for branch, weights in _DIAL_WEIGHTS.items():
        knots, sigma = dials["multisigma_" + branch]
        assert np.all(sigma == np.array([0.0, 1.0]))     # cv, ps1
        assert np.all(knots[:, 0] == 1.0)                # cv is always 1
        np.testing.assert_array_equal(knots[:, 1], weights)


def test_write_output_rejects_bad_weights(tmp_path):
    from fakedata.output import write_output

    n = 5
    src = _make_input(tmp_path, n)
    dst = str(tmp_path / "bad.root")

    with pytest.raises(ValueError, match="expected 5"):
        write_output(src, dst, {"fdwgt_x": np.ones(4)}, n)

    w = np.ones(n)
    w[2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        write_output(src, dst, {"fdwgt_x": w}, n)


def test_write_output_stl_vectors(tmp_path):
    """--stl-vectors writes real std::vector<double>, same values as uproot."""
    import uproot

    pytest.importorskip("ROOT")
    from fakedata.output import write_output

    n = 5
    src = _make_input(tmp_path, n)
    plain = str(tmp_path / "plain.root")
    stl = str(tmp_path / "stl.root")
    write_output(src, plain, _DIAL_WEIGHTS, n)
    write_output(src, stl, _DIAL_WEIGHTS, n, stl_vectors=True)

    with uproot.open(stl) as f:
        assert f["SelectedEvents"].num_entries == n
        tree = f["fakedataTree"]
        assert tree.num_entries == n
        # no counter branches: PyROOT writes the vectors directly
        assert set(tree.keys()) == {
            "multisigma_fdwgt_alpha", "multisigma_fdwgt_alpha_sigma",
            "multisigma_fdwgt_beta", "multisigma_fdwgt_beta_sigma"}
        for key in tree.keys():
            assert tree[key].typename == "std::vector<double>"

    a, b = _read_dials(plain), _read_dials(stl)
    assert set(a) == set(b)
    for name in a:
        np.testing.assert_array_equal(a[name][0], b[name][0])
        np.testing.assert_array_equal(a[name][1], b[name][1])
