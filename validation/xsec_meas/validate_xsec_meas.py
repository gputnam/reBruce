#!/usr/bin/env python3
"""Validation plots for the cross-section-measurement calculators
ub_cc1p0pi, ub_cc2p0pi, ub_ccpi, t2k_nc1pi (fakedata/calculators/xsec_meas.py,
fakedata/tki.py, fakedata/tercile.py, fakedata/xsec_table.py, data/*.csv).

Each calculator gives a signal event in a measured bin of its observable the
bin's weight sigma_data / sigma_AR23 (the AR23 side extracted in
fake-data-studies from a GENIE v3.6.2 AR23_20i GHEP sample with a signal
definition written on ALL final-state particles, see build_ub_*.py there),
and 1 to everything else; the loW / midW / hiW modes move the whole
data-MC difference of a bin into one cvwgt-weighted genie_W tercile of it.
On the CV MC sBruce files this script runs the real calculators (default
variables, all four W modes, plus divide_out_ff: true) and, per detector
(cvwgt-weighted):

  domain_<calc>       w != 1 by flavour x CC/NC x mode, one column per W mode
  weights_<calc>      weight distributions of the four branches
  closure_<calc>      per published bin, sum(cvwgt*w)/sum(cvwgt) over the
                      events of the bin, per W mode, over the CSV weight
                      (bins and weights read here from the CSV, not via
                      fakedata.xsec_table; clipped W-mode bins marked)
  audit_<calc>        signal-definition audit. The calculator counts with
                      the threshold-less true_np / true_npi / true_npi0;
                      the AR23 tables were built with the papers' window
                      definitions (CC1p: exactly one proton in 0.3-1.0
                      GeV/c, ANY number outside; charged pions allowed
                      below 70 (CC1p) / 65 (CC2p) MeV/c). The "paper"
                      signal is rebuilt from the two stored protons and
                      the leading pion (exact whenever the third proton
                      cannot be in the window; otherwise "ambiguous"),
                      and the cvwgt-weighted yield lost / gained by the
                      calculator's definition is broken down by cause;
                      also the leading-proton / count consistency
                      (true_np vs filled true_p / true_p2; p2 dropped
                      when |p2| == |p1|; npi vs filled true_cpi).
  frame               detector z vs neutrino direction. cos theta_mu to
                      the neutrino is exact from genie_prefsi_lep
                      (lepton = (0, pT, pL) in the neutrino frame). The
                      neutrino direction itself is estimated per event:
                      a point source S fitted per detector to
                      n(S)=(vtx-S)/|vtx-S| . mu_hat == cos theta_mu(nu),
                      then n projected onto the exact cone around mu_hat.
                      Shown: cos theta_mu(det) - cos theta_mu(nu), the
                      fit residual, delta_pT(det) - delta_pT(nu) and the
                      fraction of signal events changing bin (and the
                      change of the weighted yield) per variable.
  phi                 muon azimuth (cc2p muon_phi) in the detector frame
                      vs about the estimated neutrino direction
                      (x' = y_det x n), with the CSV phi weights
  divide_out_ff       (a) CC QE nu_mu: median cvwgt / ccqe_xsec_corr vs
                      Q^2 against the median deuterium->MINERvA FF weight
                      (deut_to_minerva_weight): is the MvA FF change in
                      cvwgt? (b) w(divide_out_ff: true)/w(false) = 1/w_ff
                      on signal events per calculator
  shapes_<calc>       every CSV observable of the calculator: MC signal
                      fraction per bin vs the AR23 table's fraction
                      (sigma_ar23 x width), with the fraction of MC signal
                      outside the table range -- unit / convention check
                      (deg vs rad, cos vs angle, phi range)
  spectra_<calc>      nominal vs weighted signal spectra of the default
                      observable(s), E_nu and genie_W, ratio per W mode

Checks (PASS / WARN / FAIL in summary.txt): calculator weights == weights
recomputed here (own observables, own CSV parsing, own signal mask);
independent signal mask == calculator signal_mask; w == 1 outside the
measured bins in every mode; per-bin yield closure in every mode (WARN
where WEIGHT_CLIP engaged); genie_W filled for all neutrino events and
equal to the on-shell-nucleon-at-rest formula; divide_out_ff weight ==
w_nominal / w_ff; table integrals (sum sigma*width per observable)
consistent with the published / AR23 totals.

Writes validation/plots/xsec_meas/*.png, xsec_meas.pdf and summary.txt.

Usage: ./venv/bin/python validation/xsec_meas/validate_xsec_meas.py
           [--max-files N] [files...]
"""

import contextlib
import csv
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

import common                                                # noqa: E402
from common import C, GRID, INK, INK2, NOM_COLOR, RW_COLOR, plt  # noqa: E402

from fakedata.calculator import WEIGHT_CLIP, build           # noqa: E402
from fakedata.calculators.qe_zexp import deut_to_minerva_weight  # noqa: E402
from fakedata.sbruce import valid                            # noqa: E402

DATA = os.path.join(HERE, "..", "..", "data")
CALCS = ["ub_cc1p0pi", "ub_cc2p0pi", "ub_ccpi", "t2k_nc1pi"]
MODES = ["nominal", "loW", "midW", "hiW"]
MODE_COLOR = dict(zip(MODES, [INK, C[0], C[2], C[1]]))
CSV_NAME = {"ub_cc1p0pi": "ub_cc1p0pi_xsec.csv", "ub_cc2p0pi": "ub_cc2p0pi_xsec.csv",
            "ub_ccpi": "ub_ccpi_xsec.csv", "t2k_nc1pi": "t2k_nc1pi_xsec.csv"}
DEFAULT_VAR = {"ub_cc1p0pi": "dpt_dat", "ub_cc2p0pi": "delta_PT",
               "ub_ccpi": "PionMomentum", "t2k_nc1pi": "p_costh"}
# AR23 / data signal totals [1e-38 cm^2/Ar40], fake-data-studies/VALIDATION.md
TOTALS = {"ub_cc1p0pi": (9.490, 11.628), "ub_cc2p0pi": (2.644, 2.684),
          "ub_ccpi": (3.510, 3.745)}
M_MU, M_P, M_N = 0.10565837, 0.938272081, 0.93956542
P_WIN = (0.3, 1.0)

RAW = ["true_pdg", "true_isnc", "genie_mode", "nu_E", "cvwgt", "genie_W",
       "genie_q0", "genie_q3", "true_np", "true_npi", "true_npi0", "true_nmu",
       "true_vtx_x_x", "true_vtx_y_x", "true_vtx_z_x",
       "genie_prefsi_lep_px", "genie_prefsi_lep_py", "genie_prefsi_lep_pz"]
for _p in ("true_mu", "true_p", "true_p2", "true_cpi"):
    RAW += [f"{_p}_p", f"{_p}_dir_x", f"{_p}_dir_y", f"{_p}_dir_z"]


# ------------------------------------------------------------------ tables

class Rects:
    """Measured bins as rectangles (1D: y unbounded), read from the CSV."""

    def __init__(self, rows, xk, yk=None):
        f = lambda k: np.array([float(r[k]) for r in rows])   # noqa: E731
        self.xlo, self.xhi = f(xk[0]), f(xk[1])
        if yk:
            self.ylo, self.yhi = f(yk[0]), f(yk[1])
        else:
            self.ylo = np.full(len(rows), -np.inf)
            self.yhi = np.full(len(rows), np.inf)
        self.w, self.sd, self.sa = f("weight"), f("sigma_data"), f("sigma_ar23")
        self.is2d = yk is not None

    def __len__(self):
        return len(self.w)

    def index(self, x, y=None):
        idx = np.full(len(x), -1)
        y = np.zeros(len(x)) if y is None else y
        with np.errstate(invalid="ignore"):
            for i in range(len(self)):
                s = ((x >= self.xlo[i]) & (x < self.xhi[i])
                     & (y >= self.ylo[i]) & (y < self.yhi[i]))
                idx[s] = i
        return idx

    def width(self):
        """Width the CSV sigma is differential in (for the CC1p 2D slices:
        deltaAlphaT only -- the slices are integrated over deltaPT)."""
        return (self.yhi - self.ylo) if self.is2d else (self.xhi - self.xlo)


def read_tables():
    """{calc: {obs: Rects}} straight from data/*.csv."""
    out = {}
    for c in CALCS:
        with open(os.path.join(DATA, CSV_NAME[c])) as f:
            rows = list(csv.DictReader(x for x in f if not x.startswith("#")))
        if c == "t2k_nc1pi":
            out[c] = {"p_costh": Rects(rows, ("p_lo", "p_hi"),
                                       ("costh_lo", "costh_hi"))}
            continue
        by = {}
        for r in rows:
            by.setdefault(r["obs"], []).append(r)
        tabs = {}
        for obs, rs in by.items():
            if obs == "DeltaAlphaT_DeltaPT":
                tabs["dpt_dat"] = Rects(rs, ("x_lo", "x_hi"), ("y_lo", "y_hi"))
            else:
                tabs[obs] = Rects(rs, ("x_lo", "x_hi"))
        out[c] = tabs
    return out


# ------------------------------------------------------- own observables

def vec(d, pre):
    return d[f"{pre}_p"][:, None] * np.stack(
        [d[f"{pre}_dir_x"], d[f"{pre}_dir_y"], d[f"{pre}_dir_z"]], 1)


def dot(a, b):
    return np.sum(a * b, 1)


def norm(a):
    return np.sqrt(dot(a, a))


def unit(a):
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / norm(a)[:, None]


def acosd(c):
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def basis(n):
    """Transverse basis about n: x' = y_det x n, y' = n x x' (== x, y for
    n == z)."""
    yd = np.zeros_like(n)
    yd[:, 1] = 1.0
    ex = unit(np.cross(yd, n))
    return ex, np.cross(n, ex)


def obs_value(c, var, d, n, proton=("true_p", "true_p2")):
    """Observable `var` of calculator c, computed here from the raw
    branches about the direction n (N,3). Returns a tuple (x,) or (x, y)."""
    vmu = vec(d, "true_mu")
    v1, v2 = vec(d, proton[0]), vec(d, proton[1])
    cos = lambda v: dot(v, n) / norm(v)                       # noqa: E731
    ex, ey = basis(n)

    def phi(v):
        return np.arctan2(dot(v, ey), dot(v, ex))

    def tki(vh):
        mT = vmu - dot(vmu, n)[:, None] * n
        hT = vh - dot(vh, n)[:, None] * n
        sT = mT + hT
        with np.errstate(invalid="ignore", divide="ignore"):
            dpt = norm(sT)
            dat = acosd(-dot(mT, sT) / (norm(mT) * dpt))
            dphit = acosd(-dot(mT, hT) / (norm(mT) * norm(hT)))
            pty = -dot(mT, sT) / norm(mT)
            ptx = (-dot(mT, ey) * dot(sT, ex) + dot(mT, ex) * dot(sT, ey)) / norm(mT)
        return dict(dpt=dpt, dat=dat, dphit=dphit, ptx=ptx, pty=pty)

    def ocos(a, b):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.clip(dot(a, b) / (norm(a) * norm(b)), -1, 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        if c == "ub_cc1p0pi":
            t = tki(v1)
            m = {"dpt_dat": (t["dpt"], t["dat"]), "DeltaPT": (t["dpt"],),
                 "DeltaAlphaT": (t["dat"],), "DeltaPhiT": (t["dphit"],),
                 "DeltaPtx": (t["ptx"],), "DeltaPty": (t["pty"],),
                 "MuonCosTheta": (cos(vmu),), "ProtonCosTheta": (cos(v1),),
                 "MuonMomentum": (norm(vmu),), "ProtonMomentum": (norm(v1),),
                 "ECal": (np.sqrt(norm(vmu) ** 2 + M_MU ** 2)
                          + np.sqrt(norm(v1) ** 2 + M_P ** 2) - M_P + 0.04,)}
            return m[var]
        if c == "ub_cc2p0pi":
            t = tki(v1 + v2)
            m = {"delta_PT": t["dpt"], "delta_alphaT": t["dat"],
                 "delta_phiT": t["dphit"], "muon_mom": norm(vmu),
                 "muon_costheta": cos(vmu), "muon_phi": phi(vmu),
                 "leading_mom": norm(v1), "leading_costheta": cos(v1),
                 "leading_phi": phi(v1), "recoil_mom": norm(v2),
                 "recoil_costheta": cos(v2), "recoil_phi": phi(v2),
                 "opening_angle_protons_lab": ocos(v1, v2),
                 "opening_angle_mu_both": ocos(vmu, v1 + v2)}
            return (m[var],)
        vpi = vec(d, "true_cpi")
        if c == "ub_ccpi":
            m = {"MuonCosTheta": cos(vmu), "MuonMomentum": norm(vmu),
                 "PionCosTheta": cos(vpi), "PionMomentum": norm(vpi),
                 "ThetaMuPi": np.arccos(ocos(vmu, vpi))}
            return (m[var],)
        return norm(vpi), cos(vpi)                             # t2k p_costh


def zdir(n):
    z = np.zeros((n, 3))
    z[:, 2] = 1.0
    return z


# ------------------------------------------------------ signal definitions

def sig_indep(c, d):
    """The calculator's signal definition, re-implemented from the raw
    branches (independent of fakedata.tki)."""
    numucc = (d["true_pdg"] == 14) & (d["true_isnc"] == 0)
    pmu, p1, p2, ppi = d["true_mu_p"], d["true_p_p"], d["true_p2_p"], d["true_cpi_p"]
    inw = lambda p: (p > P_WIN[0]) & (p < P_WIN[1])          # noqa: E731
    novmes = (d["true_npi"] == 0) & (d["true_npi0"] == 0)
    with np.errstate(invalid="ignore"):
        if c == "ub_cc1p0pi":
            return (numucc & (d["nu_E"] > 0) & (d["nu_E"] < 6.8) & (pmu > 0.1)
                    & (pmu < 1.2) & (d["true_np"] == 1) & inw(p1) & novmes
                    & valid(d["true_mu_dir_z"], d["true_p_dir_z"]))
        if c == "ub_cc2p0pi":
            return (numucc & (d["nu_E"] > 0) & (d["nu_E"] < 6.8) & (pmu > 0.1)
                    & (pmu < 1.2) & (d["true_np"] == 2) & inw(p1) & inw(p2)
                    & novmes & valid(d["true_mu_dir_z"], d["true_p_dir_z"],
                                     d["true_p2_dir_z"]))
        onepi = (d["true_npi"] == 1) & (d["true_npi0"] == 0) & valid(ppi, d["true_cpi_dir_z"])
        if c == "ub_ccpi":
            th = np.arccos(np.clip(dot(unit(vec(d, "true_mu")), unit(vec(d, "true_cpi"))), -1, 1))
            return (numucc & (d["nu_E"] > 0) & onepi & valid(d["true_mu_dir_z"])
                    & (pmu > 0.15) & (ppi > 0.10) & (th < 2.65))
        return (d["true_isnc"] == 1) & onepi


def paper_signal(c, d):
    """The AR23-table (paper / NUISANCE window) signal rebuilt from the
    stored particles. Returns (definite, ambiguous, proton-prefix pair for
    the observables, cause dict of definite-but-not-calculator events)."""
    numucc = (d["true_pdg"] == 14) & (d["true_isnc"] == 0)
    pmu = d["true_mu_p"]
    np_ = d["true_np"]
    p1 = np.where(valid(d["true_p_p"]), d["true_p_p"], 0.0)
    p2 = np.where(valid(d["true_p2_p"]), d["true_p2_p"], 0.0)
    inw = lambda p: (p > P_WIN[0]) & (p < P_WIN[1])          # noqa: E731
    stored = (p1 > 0).astype(int) + (p2 > 0).astype(int)
    # the third (and further) protons are below p2 in momentum: they cannot
    # be in the window when p2 <= 0.3; a p2 missing although np >= 2 (dropped
    # duplicate / count filter) leaves the count unknown
    known = ((np_ <= 2) & (stored >= np.minimum(np_, 2))) | ((p2 > 0) & (p2 <= P_WIN[0]))
    nin = inw(p1).astype(int) + inw(p2).astype(int)
    ppi = np.where(valid(d["true_cpi_p"]), d["true_cpi_p"], 0.0)
    if c in ("ub_cc1p0pi", "ub_cc2p0pi"):
        pi_cut = 0.07 if c == "ub_cc1p0pi" else 0.065
        base = (numucc & (d["nu_E"] > 0) & (d["nu_E"] < 6.8) & (pmu > 0.1)
                & (pmu < 1.2) & (d["true_npi0"] == 0) & (ppi <= pi_cut)
                & valid(d["true_mu_dir_z"]))
        want = 1 if c == "ub_cc1p0pi" else 2
        definite = base & known & (nin == want)
        ambiguous = base & ~known & (nin == want)
    else:
        s = sig_indep(c, d)
        return s, np.zeros_like(s), None
    return definite, ambiguous, (p1, p2)


# ------------------------------------------------------------------ loading

def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def process(sb, tables):
    a = sb.arrays(RAW)
    d = {k: np.asarray(v, dtype=np.float64) for k, v in a.items()}
    for c in CALCS:
        calc = build({"type": c})
        res = quiet(calc.compute, sb)
        for m in MODES:
            d[f"w_{c}_{m}"] = res[calc.branch + ("" if m == "nominal" else f"_{m}")]
        dff = build({"type": c, "w_modes": ["nominal"], "divide_out_ff": True})
        d[f"wdff_{c}"] = quiet(dff.compute, sb)[dff.branch]
        ca = sb.arrays(calc.branches_needed())
        d[f"sigc_{c}"] = np.asarray(calc.signal_mask(ca))
        # every CSV observable through the calculator's own observable()
        for v in tables[c]:
            cv = build({"type": c, "variable": v, "w_modes": ["nominal"]})
            o = cv.observable(sb.arrays(cv.branches_needed()))
            for j, x in enumerate(o):
                d[f"obs_{c}_{v}_{j}"] = np.asarray(x, dtype=np.float64)
    d["w_ff"] = deut_to_minerva_weight(sb)
    cq = build({"type": "ccqe_xsec_corr", "match_tool": True})
    d["w_ccqe"] = quiet(cq.compute, sb)[cq.branch]
    return d


# ------------------------------------------------------- neutrino direction

def fit_source(d):
    """Point source S (cm, detector coords) with (vtx-S)/|vtx-S| . mu_hat ==
    cos theta_mu(nu) from genie_prefsi_lep; Gauss-Newton."""
    m = d["_ccmu"]
    v = d["_vtx"][m][:200000]
    mu = d["_muhat"][m][:200000]
    cn = d["_cosnu"][m][:200000]

    def res(S):
        return dot(unit(v - S), mu) - cn
    S = np.array([0.0, 0.0, -1.0e4])
    for _ in range(40):
        r = res(S)
        J = np.stack([(res(S + 10.0 * np.eye(3)[k]) - r) / 10.0 for k in range(3)], 1)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        S = S + step
        if np.max(np.abs(step)) < 1e-3:
            break
    return S


def nu_frame(d):
    lep = np.stack([d["genie_prefsi_lep_px"], d["genie_prefsi_lep_py"],
                    d["genie_prefsi_lep_pz"]], 1)
    d["_vtx"] = np.stack([d["true_vtx_x_x"], d["true_vtx_y_x"], d["true_vtx_z_x"]], 1)
    d["_muhat"] = unit(vec(d, "true_mu"))
    with np.errstate(invalid="ignore", divide="ignore"):
        d["_cosnu"] = lep[:, 2] / norm(lep)
    d["_ccmu"] = ((np.abs(d["true_pdg"]) == 14) & (d["true_isnc"] == 0)
                  & valid(d["genie_prefsi_lep_pz"], d["true_mu_dir_z"], d["true_vtx_x_x"])
                  & (d["true_mu_p"] > 0))
    S = fit_source(d)
    nm = unit(d["_vtx"] - S)
    n = nm.copy()
    m = d["_ccmu"]
    u = d["_muhat"][m]
    c = np.clip(d["_cosnu"][m], -1, 1)
    perp = unit(nm[m] - dot(nm[m], u)[:, None] * u)
    n[m] = c[:, None] * u + np.sqrt(1 - c ** 2)[:, None] * perp
    bad = ~np.all(np.isfinite(n), 1)
    n[bad] = zdir(bad.sum())
    d["_n_model"], d["_n"], d["_S"] = nm, n, S


# ------------------------------------------------------------------- plots

def fig_closure(c, d, det, tab, rows):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(tab))
    ax.plot(x, tab.w, color=INK2, lw=0, marker="_", ms=22, mew=2.5, label="CSV weight")
    for k, m in enumerate(MODES):
        r = rows[m]
        ax.plot(x + 0.12 * (k - 1.5), r["ach"], lw=0, marker="o", ms=4,
                color=MODE_COLOR[m], label=m)
        cl = r["clipped"]
        if cl.any():
            ax.plot(x[cl] + 0.12 * (k - 1.5), r["ach"][cl], lw=0, marker="x",
                    ms=9, color=MODE_COLOR[m])
    ax.axhline(1, color=GRID, lw=1)
    ax.set_xlabel("measured bin (CSV order)")
    ax.set_ylabel("sum(cvwgt w) / sum(cvwgt) in bin")
    ax.set_title(f"{det}: {c} [{DEFAULT_VAR[c]}] per-bin yield closure "
                 "(x = a W-mode weight hit WEIGHT_CLIP)", fontsize=10, color=INK)
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    return fig


def fig_audit(c, d, det, aud):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8),
                            gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axs[0]
    labs = list(aud["bars"])
    vals = [aud["bars"][k] for k in labs]
    cols = [C[0], C[5], C[1], C[1], C[1], C[1], C[3], C[4]][:len(labs)]
    ax.barh(range(len(labs)), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(labs)), labs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("cvwgt-weighted events")
    ax.set_title(f"{det}: {c} signal: calculator vs paper windows", fontsize=10, color=INK)
    ax = axs[1]
    if c in ("ub_cc1p0pi", "ub_cc2p0pi"):
        # sub-leading proton momentum of numu CC 0pi events with the leading
        # proton in the window (CC1p) / both stored in window (CC2p)
        m, xl = aud["p2sel"], aud["p2label"]
        bins = np.linspace(0, 1.2, 49)
        p2 = d["true_p2_p"][m]
        ax.hist(np.where(p2 > 0, p2, -1), bins, weights=d["cvwgt"][m],
                histtype="step", color=C[0], lw=2, label="stored true_p2_p")
        ax.axvspan(*P_WIN, color=GRID, alpha=0.5, label="signal window")
        ax.set_xlabel(xl)
        ax.set_ylabel("events")
        ax.legend(fontsize=8)
        ax.set_title(aud["p2title"], fontsize=9, color=INK)
    else:
        txt = "\n".join(aud["notes"])
        ax.axis("off")
        ax.text(0, 1, txt, va="top", fontsize=8.5, family="monospace", color=INK)
    fig.tight_layout()
    return fig


def fig_frame(d, det, mig):
    m = d["_ccmu"] & (d["true_pdg"] == 14)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
    ax = axs[0, 0]
    dc = (d["_muhat"][:, 2] - d["_cosnu"])[m]
    lim = max(0.01, np.percentile(np.abs(dc), 99.5))
    bins = np.linspace(-lim, lim, 81)
    ax.hist(dc, bins, weights=d["cvwgt"][m], histtype="step", lw=2, color=C[0],
            label="cos theta_mu(det z) - cos theta_mu(nu)")
    rm = (dot(d["_n_model"], d["_muhat"]) - d["_cosnu"])[m]
    ax.hist(rm, bins, weights=d["cvwgt"][m], histtype="step", lw=2, color=C[1],
            label="point-source model - exact (fit residual)")
    p = np.percentile(dc, [1, 99])
    ax.set_title(f"nu_mu CC: 1-99% of det-nu = [{p[0]:+.4f}, {p[1]:+.4f}];\n"
                 f"source S = ({d['_S'][0]:.0f}, {d['_S'][1]:.0f}, {d['_S'][2]:.0f}) cm",
                 fontsize=9, color=INK)
    ax.legend(fontsize=7)
    ax.set_xlabel("delta cos theta_mu")
    ax = axs[0, 1]
    s = d["sigc_ub_cc1p0pi"]
    for c, key, col in (("ub_cc1p0pi", "sigc_ub_cc1p0pi", C[0]),
                        ("ub_cc2p0pi", "sigc_ub_cc2p0pi", C[2])):
        s = d[key]
        x0 = obs_value(c, "DeltaPT" if c == "ub_cc1p0pi" else "delta_PT", d, zdir(len(s)))[0][s]
        x1 = obs_value(c, "DeltaPT" if c == "ub_cc1p0pi" else "delta_PT", d, d["_n"])[0][s]
        ax.hist(x0 - x1, np.linspace(-0.08, 0.08, 81), weights=d["cvwgt"][s],
                histtype="step", lw=2, color=col, label=f"{c} signal")
    ax.set_xlabel("delta_pT(det z) - delta_pT(nu est.) [GeV/c]")
    ax.legend(fontsize=8)
    ax.set_title("TKI: detector frame vs estimated neutrino frame", fontsize=9, color=INK)
    ax = axs[1, 0]
    labs = list(mig)
    ax.barh(range(len(labs)), [100 * mig[k][0] for k in labs], color=C[0])
    for i, k in enumerate(labs):
        ax.text(100 * mig[k][0], i, f" {100 * mig[k][0]:.1f}%  (dY {100 * mig[k][1]:+.2f}%)",
                va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(labs)), labs, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("% of signal (cvwgt) changing bin, det z -> nu  "
                  "(dY: weighted-yield change)")
    ax.set_xlim(0, max(5, 1.6 * max(100 * v[0] for v in mig.values())))
    ax = axs[1, 1]
    cosb = np.linspace(-1, 1, 41)
    cosd = d["_muhat"][:, 2]
    prof = []
    for lo, hi in zip(cosb[:-1], cosb[1:]):
        sel = m & (d["_cosnu"] >= lo) & (d["_cosnu"] < hi)
        prof.append(np.percentile((cosd - d["_cosnu"])[sel], [16, 50, 84]) if sel.sum() > 20
                    else [np.nan] * 3)
    prof = np.array(prof)
    xc = 0.5 * (cosb[1:] + cosb[:-1])
    ax.fill_between(xc, prof[:, 0], prof[:, 2], color=C[0], alpha=0.25, lw=0)
    ax.plot(xc, prof[:, 1], color=C[0])
    ax.axhline(0, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("cos theta_mu(nu)")
    ax.set_ylabel("cos(det z) - cos(nu): median, 16-84%")
    fig.suptitle(f"{det}: observables w.r.t. detector z vs the neutrino direction",
                 color=INK)
    fig.tight_layout()
    return fig


def fig_phi(d, det, tab):
    s = d["sigc_ub_cc2p0pi"] | d["sigc_ub_cc1p0pi"]
    p0 = obs_value("ub_cc2p0pi", "muon_phi", d, zdir(len(s)))[0][s]
    p1 = obs_value("ub_cc2p0pi", "muon_phi", d, d["_n"])[0][s]
    t = tab["ub_cc2p0pi"]["muon_phi"]
    edges = np.r_[t.xlo, t.xhi[-1]]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.2, 1]))
    cw = d["cvwgt"][s]
    for x, lab, col in ((p0, "detector frame (calculator)", C[0]),
                        (p1, "about estimated nu direction", C[1])):
        h, _ = np.histogram(x, edges, weights=cw)
        ax.stairs(h / h.sum(), edges, color=col, lw=2, label=lab)
    ax.set_ylabel("fraction (CC1p + CC2p signal muons)")
    ax.legend(fontsize=8)
    axr.stairs(t.w, edges, color=INK, lw=2, label="CSV muon_phi weight")
    axr.axhline(1, color=INK2, lw=0.8, ls=":")
    axr.set_xlabel("muon phi [rad] (atan2(y, x))")
    axr.set_ylabel("weight")
    mig = np.average(t.index(p0) != t.index(p1), weights=cw)
    ax.set_title(f"{det}: muon azimuth; {100 * mig:.1f}% change muon_phi bin "
                 "det -> nu frame", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def fig_ff(d, det, out_lines):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axs[0]
    qe = ((d["true_pdg"] == 14) & (d["true_isnc"] == 0) & (d["genie_mode"] == 0)
          & (d["w_ccqe"] > 0) & (d["cvwgt"] > 0) & valid(d["genie_q0"]))
    Q2 = (d["genie_q3"] ** 2 - d["genie_q0"] ** 2)[qe]
    r = (d["cvwgt"] / d["w_ccqe"])[qe]
    ff = d["w_ff"][qe]
    qb = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 2.0])
    xc = 0.5 * (qb[1:] + qb[:-1])
    for y, lab, col in ((r, "cvwgt / ccqe_xsec_corr(match_tool)", C[0]),
                        (ff, "deut -> MINERvA FF weight (Nieves port)", C[1]),
                        (r / ff, "cvwgt / (ccqe_corr x FF)", C[2])):
        med = [np.median(y[(Q2 >= lo) & (Q2 < hi)]) if np.sum((Q2 >= lo) & (Q2 < hi)) > 10
               else np.nan for lo, hi in zip(qb[:-1], qb[1:])]
        ax.plot(xc, med, marker="o", color=col, label=lab)
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Q^2 [GeV^2]")
    ax.set_ylabel("median weight")
    ax.set_title("nu_mu CC QE: is the MvA FF change inside cvwgt?", fontsize=10, color=INK)
    ax.legend(fontsize=8)
    ax = axs[1]
    bins = np.linspace(0.5, 1.2, 71)
    for i, c in enumerate(CALCS):
        s = d["sigc_" + c]
        rr = 1.0 / d["w_ff"][s]
        ax.hist(rr, bins, weights=d["cvwgt"][s], histtype="step", lw=2, color=C[i],
                label=f"{c}: <w(true)/w(false)> = {np.average(rr, weights=d['cvwgt'][s]):.4f}")
    ax.set_yscale("log")
    ax.set_xlabel("w(divide_out_ff: true) / w(false) = 1 / w_ff (signal events)")
    ax.legend(fontsize=7.5)
    fig.suptitle(f"{det}: divide_out_ff", color=INK)
    fig.tight_layout()
    return fig


def fig_shapes(c, d, det, tabs, rng_info):
    vars_ = list(tabs)
    ncol = 4
    nrow = int(np.ceil(len(vars_) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.9 * nrow), squeeze=False)
    s = d["sigc_" + c]
    cw = d["cvwgt"][s]
    for ax, v in zip(axs.flat, vars_):
        t = tabs[v]
        o = [d[f"obs_{c}_{v}_{j}"][s] for j in range(2 if t.is2d else 1)]
        idx = t.index(*o)
        hm = np.bincount(idx[idx >= 0], weights=cw[idx >= 0], minlength=len(t))
        fin = 1 - np.sum(cw[idx >= 0]) / np.sum(cw)
        fa = t.sa * t.width()
        fd = t.sd * t.width()
        x = np.arange(len(t) + 1)
        ax.stairs(hm / hm.sum(), x, color=C[0], lw=2, label="MC signal")
        ax.stairs(fa / fa.sum(), x, color=INK2, lw=1.5, ls="--", label="AR23 table")
        ax.stairs(fd / fd.sum(), x, color=C[1], lw=1.2, label="data table")
        ax.set_title(f"{v}: {100 * fin:.1f}% MC out of range", fontsize=8.5,
                     color=INK if fin < 0.2 else C[1])
        ax.tick_params(labelsize=7)
        rng_info[(c, v)] = (fin, np.nanmin(o[0]) if len(o[0]) else np.nan,
                            np.nanmax(o[0]) if len(o[0]) else np.nan)
    for ax in list(axs.flat)[len(vars_):]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=7)
    fig.suptitle(f"{det}: {c} -- per-bin fraction, MC signal vs table (x = bin index)",
                 color=INK)
    fig.tight_layout()
    return fig


def fig_spectra(c, d, det, tab):
    s = d["sigc_" + c]
    cw = d["cvwgt"][s]
    specs = []
    t = tab
    if c == "ub_cc1p0pi":
        specs += [(d[f"obs_{c}_dpt_dat_0"][s], np.linspace(0, 1.0, 41), "delta_pT [GeV/c]"),
                  (d[f"obs_{c}_dpt_dat_1"][s], np.linspace(0, 180, 37), "delta_alphaT [deg]")]
    elif c == "t2k_nc1pi":
        specs += [(d[f"obs_{c}_p_costh_0"][s], np.linspace(0, 1.5, 31), "p_pi [GeV/c]"),
                  (d[f"obs_{c}_p_costh_1"][s], np.linspace(-1, 1, 41), "cos theta_pi")]
    else:
        v = DEFAULT_VAR[c]
        lo, hi = t.xlo[0], t.xhi[-1]
        pad = 0.3 * (hi - lo)
        specs.append((d[f"obs_{c}_{v}_0"][s], np.linspace(max(lo - pad, 0), hi + pad, 41), v))
    specs += [(d["nu_E"][s], np.linspace(0, 3, 31), "E_nu [GeV]"),
              (d["genie_W"][s], np.linspace(0.8, 2.0, 49), "genie_W [GeV]")]
    fig, axs = plt.subplots(2, len(specs), figsize=(4.3 * len(specs), 6.2),
                            gridspec_kw=dict(height_ratios=[2.2, 1]), squeeze=False)
    for j, (x, bins, xl) in enumerate(specs):
        ax, axr = axs[0, j], axs[1, j]
        h0, _ = np.histogram(x, bins, weights=cw)
        ax.stairs(h0, bins, color=NOM_COLOR, lw=2, label="nominal")
        for m in MODES:
            h1, _ = np.histogram(x, bins, weights=cw * d[f"w_{c}_{m}"][s])
            if m == "nominal":
                ax.stairs(h1, bins, color=RW_COLOR, lw=2, label="weighted")
            r = np.divide(h1, h0, out=np.full(len(h0), np.nan), where=h0 > 0)
            axr.stairs(r, bins, color=MODE_COLOR[m], lw=1.5, label=m, baseline=None)
        axr.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_ylabel("signal events")
        axr.set_ylabel("weighted / nominal")
        axr.set_xlabel(xl)
        axr.set_ylim(0, 3)
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{det}: {c} signal spectra", color=INK)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ analysis

def default_bin(c, d, tab, n=None, proton=("true_p", "true_p2"), sig=None):
    n = zdir(len(d["cvwgt"])) if n is None else n
    o = obs_value(c, DEFAULT_VAR[c], d, n, proton)
    idx = tab.index(*o)
    if sig is not None:
        idx = np.where(sig, idx, -1)
    return idx


def closure(out, c, d, tab, det):
    s_ind = sig_indep(c, d)
    nd = np.sum(s_ind != d["sigc_" + c])
    out.check(nd == 0, f"{c}: independent signal mask == calculator signal_mask "
                       f"({nd} disagreements of {s_ind.sum()} signal)")
    idx = default_bin(c, d, tab, sig=s_ind)
    inb = idx >= 0
    w_ref = np.where(inb, tab.w[np.maximum(idx, 0)], 1.0)
    dw = np.max(np.abs(d[f"w_{c}_nominal"] - w_ref))
    out.check(dw < 1e-9, f"{c}: nominal weight == own recomputation (max |dw| = {dw:.1e}; "
                         f"{inb.sum()} events in measured bins, "
                         f"{100 * np.mean(inb[s_ind]) if s_ind.any() else 0:.1f}% of signal)")
    rows = {}
    cv = d["cvwgt"]
    for m in MODES:
        w = d[f"w_{c}_{m}"]
        off = np.sum(w[~inb] != 1.0)
        out.check(off == 0, f"{c} {m}: w == 1 outside the measured bins ({off} violate)")
        num = np.bincount(idx[inb], weights=(cv * w)[inb], minlength=len(tab))
        den = np.bincount(idx[inb], weights=cv[inb], minlength=len(tab))
        ach = np.divide(num, den, out=np.full(len(tab), np.nan), where=den > 0)
        clipped = np.zeros(len(tab), bool)
        if m != "nominal":
            atclip = inb & ((w <= WEIGHT_CLIP[0]) | (w >= WEIGHT_CLIP[1])) & (w != 1.0)
            clipped = np.bincount(idx[atclip], minlength=len(tab)) > 0
        dev = np.abs(ach - tab.w)
        ok = ~clipped & (den > 0)
        mx = np.nanmax(dev[ok]) if ok.any() else 0.0
        out.check(mx < 1e-6, f"{c} {m}: per-bin yield closure, max |achieved - CSV| = "
                             f"{mx:.1e} over {ok.sum()} unclipped bins")
        if clipped.any():
            out.check(False, f"{c} {m}: WEIGHT_CLIP engaged in bins "
                             f"{list(np.flatnonzero(clipped))}: achieved "
                             f"{np.round(ach[clipped], 3).tolist()} vs CSV "
                             f"{np.round(tab.w[clipped], 3).tolist()}", warn=True)
        nobin = np.flatnonzero(den == 0)
        if len(nobin):
            out.text(f"    ({c} {m}: no MC in bins {list(nobin)})")
        rows[m] = dict(ach=ach, clipped=clipped)
    # divide_out_ff
    r = d[f"wdff_{c}"] * d["w_ff"] - d[f"w_{c}_nominal"]
    out.check(np.max(np.abs(r)) < 1e-9, f"{c}: divide_out_ff weight == w_nominal / w_ff")
    return rows, idx


def audit(out, c, d, tab):
    cv = d["cvwgt"]
    s = d["sigc_" + c]
    bars = {}
    notes = []
    aud = dict(bars=bars, notes=notes)
    ncalc = cv[s].sum()
    bars["calculator signal"] = ncalc
    if c in ("ub_cc1p0pi", "ub_cc2p0pi"):
        dfn, amb, (p1, p2) = paper_signal(c, d)
        bars["paper signal (definite)"] = cv[dfn].sum()
        bars["paper signal (ambiguous, 3rd p)"] = cv[amb].sum()
        lost = dfn & ~s
        inw = lambda p: (p > P_WIN[0]) & (p < P_WIN[1])      # noqa: E731
        npi = d["true_npi"]
        if c == "ub_cc1p0pi":
            c_sub = lost & (d["true_np"] > 1) & inw(p1) & (p2 <= P_WIN[0])
            c_hi = lost & ~c_sub & (p1 >= P_WIN[1])
        else:
            c_sub = lost & (d["true_np"] > 2)
            c_hi = lost & ~c_sub & (p1 >= P_WIN[1])
        c_pi = lost & ~c_sub & ~c_hi & (npi >= 1)
        c_oth = lost & ~c_sub & ~c_hi & ~c_pi
        thr = "< 0.3" if c == "ub_cc1p0pi" else "(np>2, p3<0.3 implied)"
        bars[f"lost: extra protons {thr}"] = cv[c_sub].sum()
        bars["lost: proton above 1 GeV/c"] = cv[c_hi].sum()
        bars["lost: soft pi+- below cut"] = cv[c_pi].sum()
        bars["lost: other / count mismatch"] = cv[c_oth].sum()
        gained = s & ~dfn & ~amb
        bars["gained (calc sig, not paper)"] = cv[gained].sum()
        # intended vs achieved weighted-yield change on the paper signal
        prot = np.where(inw(p1), 0, 1)  # in-window leading proton for CC1p
        idx_cal = np.where(s, default_bin(c, d, tab), -1)
        if c == "ub_cc1p0pi":
            # paper observable: the in-window proton (p2 when p1 > 1 GeV/c)
            dd = dict(d)
            for k in ("p", "dir_x", "dir_y", "dir_z"):
                dd[f"true_pw_{k}"] = np.where(prot == 0, d[f"true_p_{k}"], d[f"true_p2_{k}"])
            idx_pap = np.where(dfn, default_bin(c, dd, tab, proton=("true_pw", "true_p2")), -1)
        else:
            idx_pap = np.where(dfn, default_bin(c, d, tab), -1)
        w_ideal = np.where(idx_pap >= 0, tab.w[np.maximum(idx_pap, 0)], 1.0)
        w_cal = d[f"w_{c}_nominal"]
        Y0 = cv[dfn].sum()
        want = np.sum((cv * w_ideal)[dfn]) - Y0
        got = np.sum((cv * w_cal)[dfn]) - Y0
        extra = np.sum((cv * (w_cal - 1))[s & ~dfn])
        aud["frac_lost"] = cv[lost].sum() / max(cv[dfn].sum(), 1e-12)
        out.text(f"  {c}: calculator signal {ncalc:.1f}, paper definite {Y0:.1f} "
                 f"(+{cv[amb].sum():.1f} ambiguous); lost {100 * aud['frac_lost']:.1f}% of "
                 f"paper signal [extra-p {cv[c_sub].sum():.1f}, p>1 GeV {cv[c_hi].sum():.1f}, "
                 f"soft pi {cv[c_pi].sum():.1f}, other {cv[c_oth].sum():.1f}], "
                 f"gained {cv[gained].sum():.1f}")
        out.text(f"  {c}: weighted-yield change on paper signal: intended {want:+.1f} "
                 f"({100 * want / Y0:+.2f}%), achieved {got:+.1f} ({100 * got / Y0:+.2f}%); "
                 f"applied to non-paper events {extra:+.1f}")
        out.check(aud["frac_lost"] < 0.05, f"{c}: < 5% of the paper (AR23-table) signal "
                  f"missed by the true_n* counting ({100 * aud['frac_lost']:.1f}%)", warn=True)
        # leading-proton bookkeeping
        nm = valid(d["true_mu_p"]) & (d["true_pdg"] == 14) & (d["true_isnc"] == 0)
        np_ = d["true_np"]
        p2miss = nm & (np_ >= 2) & ~(d["true_p2_p"] > 0)
        p1miss = nm & (np_ >= 1) & ~(d["true_p_p"] > 0)
        extra_st = nm & (np_ == 0) & (d["true_p_p"] > 0)
        eq = nm & (d["true_p2_p"] > 0) & (np.abs(d["true_p2_p"] - d["true_p_p"]) < 1e-6)
        order = nm & (d["true_p2_p"] > d["true_p_p"] + 1e-6)
        out.text(f"  {c}: nu_mu CC bookkeeping: np>=2 but no p2 {p2miss.sum()}, np>=1 but "
                 f"no p {p1miss.sum()}, np==0 but p filled {extra_st.sum()}, "
                 f"|p2|==|p1| {eq.sum()}, p2>p1 {order.sum()}")
        if c == "ub_cc1p0pi":
            aud["p2sel"] = (nm & inw(p1) & (np_ >= 2) & (d["true_npi"] == 0)
                            & (d["true_npi0"] == 0) & (d["true_mu_p"] > 0.1)
                            & (d["true_mu_p"] < 1.2))
            aud["p2label"] = "true_p2_p [GeV/c] (leading p in window, np >= 2, 0 pi)"
        else:
            aud["p2sel"] = (nm & inw(p1) & inw(p2) & (np_ >= 2) & (d["true_npi"] == 0)
                            & (d["true_npi0"] == 0) & (d["true_mu_p"] > 0.1)
                            & (d["true_mu_p"] < 1.2))
            aud["p2label"] = "true_p2_p [GeV/c] (both stored p in window, 0 pi)"
        m = aud["p2sel"]
        fb = np.sum(cv[m & (d["true_p2_p"] <= P_WIN[0])]) / max(np.sum(cv[m]), 1e-12)
        aud["p2title"] = (f"{100 * fb:.1f}% have p2 <= 0.3 GeV/c (paper signal, "
                          "calculator rejects)" if c == "ub_cc1p0pi" else
                          f"np distribution: " + ", ".join(
                              f"np={k}: {np.sum(cv[m & (np_ == k)]):.0f}" for k in (2, 3, 4)))
    else:
        cnt = d["true_npi"]
        filled = d["true_cpi_p"] > 0
        sel = (d["true_isnc"] == 1) if c == "t2k_nc1pi" else \
            ((d["true_pdg"] == 14) & (d["true_isnc"] == 0))
        mism0 = sel & (cnt == 0) & filled
        mism1 = sel & (cnt >= 1) & ~filled
        notes.append(f"pi counts vs true_cpi: npi==0 & cpi filled {mism0.sum()}, "
                     f"npi>=1 & cpi empty {mism1.sum()}")
        if c == "t2k_nc1pi":
            fl = {k: cv[s & (d['true_pdg'] == k)].sum() for k in (14, -14, 12, -12)}
            tot = max(sum(fl.values()), 1e-12)
            notes.append("NC signal by flavour (cvwgt): " + ", ".join(
                f"{k}: {100 * v / tot:.1f}%" for k, v in fl.items()))
            for k, v in fl.items():
                if k != 14:
                    bars[f"NC1pi signal from pdg {k}"] = v
            inb = d[f"w_{c}_nominal"] != 1.0
            notes.append(f"in measured RoI: {cv[s & inb].sum():.1f} of {ncalc:.1f}")
            notes.append("pi charge not stored: pi- NC1pi weighted with the")
            notes.append("NC1pi+ table (isospin: Delta0 -> p pi- vs Delta+ -> n pi+;")
            notes.append("Ar has 22 n / 18 p, so ~half the weighted events are pi-)")
            notes.append("T2K table is CH target, any-flavour applied")
        else:
            nmu = d["true_nmu"]
            notes.append(f"signal with true_nmu != 1: {np.sum(s & (nmu != 1))}")
            notes.append("CC1pi+- definition matches the AR23 builder")
            notes.append("(exactly one pi+-, no pi0, any nucleons) -- counts only")
            vpi = d[f"obs_{c}_PionMomentum_0"][s]
            vmu = d[f"obs_{c}_MuonMomentum_0"][s]
            f6 = np.average(vpi >= 0.6, weights=cv[s])
            f15 = np.average(vmu >= 1.5, weights=cv[s])
            notes.append(f"signal with p_pi >= 0.60 (beyond table, w=1): {100 * f6:.1f}%")
            notes.append(f"signal with p_mu >= 1.50 (beyond table, w=1): {100 * f15:.1f}%")
            aud["f_pi_over"] = f6
        bars["count/leading mismatch (sel)"] = mism0.sum() + mism1.sum()
        for n_ in notes:
            out.text(f"  {c}: {n_}")
    return aud


def migrations(d, tables):
    """Fraction of (calculator) signal changing bin det-z -> nu frame and
    the relative change of the weighted signal yield."""
    plan = [("ub_cc1p0pi", v) for v in ("dpt_dat", "DeltaPT", "DeltaAlphaT", "DeltaPhiT",
                                         "DeltaPtx", "DeltaPty", "MuonCosTheta",
                                         "ProtonCosTheta")]
    plan += [("ub_cc2p0pi", v) for v in ("delta_PT", "delta_alphaT", "muon_costheta",
                                          "muon_phi")]
    plan += [("ub_ccpi", v) for v in ("MuonCosTheta", "PionCosTheta")]
    plan += [("t2k_nc1pi", "p_costh")]
    res = {}
    z = zdir(len(d["cvwgt"]))
    for c, v in plan:
        s = d["sigc_" + c]
        if not s.any():
            continue
        t = tables[c][v]
        if v in ("MuonCosTheta", "muon_costheta"):
            # exact: cos theta_mu to the neutrino from genie_prefsi_lep
            i0 = t.index(d["_muhat"][:, 2])
            i1 = t.index(d["_cosnu"])
        else:
            i0 = t.index(*obs_value(c, v, d, z))
            i1 = t.index(*obs_value(c, v, d, d["_n"]))
        cw = d["cvwgt"][s]
        frac = np.average(i0[s] != i1[s], weights=cw)
        w0 = np.where(i0 >= 0, t.w[np.maximum(i0, 0)], 1.0)[s]
        w1 = np.where(i1 >= 0, t.w[np.maximum(i1, 0)], 1.0)[s]
        dy = np.sum(cw * w0) / np.sum(cw * w1) - 1
        res[f"{c.replace('ub_', '')}: {v}"] = (frac, dy)
    return res


def table_integrals(out, tables):
    out.text("== table integrals sum(sigma x width) [1e-38 cm^2/Ar]: data/AR23 per "
             "observable vs published totals")
    for c, (ta, td) in TOTALS.items():
        for v, t in tables[c].items():
            ia, idd = np.sum(t.sa * t.width()), np.sum(t.sd * t.width())
            fa, fd = ia / ta, idd / td
            out.check(abs(fa - fd) < 0.03,
                      f"{c} {v}: data integral {idd:.3f} ({100 * fd:.0f}% of total), "
                      f"AR23 {ia:.3f} ({100 * fa:.0f}%) -- same phase-space coverage",
                      warn=True)


def wcheck(out, d, det):
    nu = d["true_pdg"] != -1
    vw = valid(d["genie_W"])
    out.text(f"  genie_W filled: {100 * vw[nu].mean():.2f}% of neutrino events; by mode: " +
             ", ".join(f"{common.MODE_NAMES[m]} {100 * vw[nu & (d['genie_mode'] == m)].mean():.1f}%"
                       for m in common.MODE_NAMES if np.any(nu & (d["genie_mode"] == m))))
    out.check(vw[nu].mean() > 0.999, f"{det}: genie_W filled for (almost) all neutrino "
                                     "events -- tercile.py's 'most SBND events lack genie_W' "
                                     "is stale if PASS")
    for c in CALCS:
        s = d["sigc_" + c]
        out.text(f"    {c}: signal events without genie_W: {np.sum(s & ~vw)} of {s.sum()}")
    m = nu & vw & valid(d["genie_q0"], d["genie_q3"])
    Mavg = 0.5 * (M_P + M_N)
    W2 = Mavg ** 2 + 2 * Mavg * d["genie_q0"] - (d["genie_q3"] ** 2 - d["genie_q0"] ** 2)
    dw = d["genie_W"] - np.sqrt(np.maximum(W2, 0))
    p = np.percentile(np.abs(dw[m]), 99.9)
    out.check(p < 2e-3, f"{det}: genie_W == sqrt(M^2 + 2 M q0 - Q^2) (nucleon at rest, "
                        f"M = <m_N>) to |dW| < 2 MeV (99.9% = {1000 * p:.2f} MeV) -- "
                        "the gst 'W', not GENIE's generated (off-shell) W")


# ------------------------------------------------------------------ main

def main():
    args = common.parse_args(__doc__.splitlines()[0])
    by_det = common.files_by_detector(args)
    tables = read_tables()
    out = common.Output("xsec_meas", "ub_cc1p0pi / ub_cc2p0pi / ub_ccpi / t2k_nc1pi "
                        "validation (validation/xsec_meas/validate_xsec_meas.py)")
    table_integrals(out, tables)
    for det, files in by_det.items():
        out.text("")
        out.text(f"== {det}: {len(files)} files")
        d = common.load(files, process, tables)
        nu_frame(d)
        out.text(f"  {len(d['cvwgt'])} events; fitted nu source S = "
                 f"({d['_S'][0]:.0f}, {d['_S'][1]:.0f}, {d['_S'][2]:.0f}) cm")
        wcheck(out, d, det)
        for c in CALCS:
            tab = tables[c][DEFAULT_VAR[c]]
            out.text(f"-- {c} [{DEFAULT_VAR[c]}]")
            ws = {m: d[f"w_{c}_{m}"] for m in MODES}
            for m in MODES:
                common.weight_summary(out, ws[m], d["cvwgt"], m)
            out.save(common.fig_domain(d, ws, det, c), f"domain_{c}_{det}")
            out.save(common.fig_weight_hists(ws, det, c), f"weights_{c}_{det}")
            rows, _ = closure(out, c, d, tab, det)
            out.save(fig_closure(c, d, det, tab, rows), f"closure_{c}_{det}")
            aud = audit(out, c, d, tab)
            out.save(fig_audit(c, d, det, aud), f"audit_{c}_{det}")
            rng = {}
            out.save(fig_shapes(c, d, det, tables[c], rng), f"shapes_{c}_{det}")
            for (cc, v), (fin, lo, hi) in rng.items():
                if fin > 0.25:
                    out.text(f"    {cc} {v}: {100 * fin:.1f}% of MC signal outside the "
                             f"table range (MC range [{lo:.3g}, {hi:.3g}])")
            out.save(fig_spectra(c, d, det, tab), f"spectra_{c}_{det}")
            s = d["sigc_" + c]
            if s.any():
                r = 1.0 / d["w_ff"][s]
                out.text(f"  {c}: w(divide_out_ff true)/w(false) on signal: cvwgt-mean "
                         f"{np.average(r, weights=d['cvwgt'][s]):.4f}, range "
                         f"[{r.min():.3f}, {r.max():.3f}], {100 * np.mean(r != 1):.1f}% != 1")
        # frame
        mig = migrations(d, tables)
        out.text("-- frame: detector z vs neutrino direction")
        m = d["_ccmu"] & (d["true_pdg"] == 14)
        dc = (d["_muhat"][:, 2] - d["_cosnu"])[m]
        rm = (dot(d["_n_model"], d["_muhat"]) - d["_cosnu"])[m]
        p = np.percentile(dc, [1, 50, 99])
        out.text(f"  cos theta_mu(det) - cos theta_mu(nu): 1/50/99% = "
                 f"{p[0]:+.4f}/{p[1]:+.4f}/{p[2]:+.4f}; point-source residual RMS "
                 f"{np.std(rm):.4f} (vs {np.std(dc):.4f} for det z)")
        for k, (f, dy) in mig.items():
            out.text(f"  {k:32s} {100 * f:5.1f}% of signal change bin, weighted yield "
                     f"det/nu - 1 = {100 * dy:+.2f}%")
        worst = max(mig.items(), key=lambda kv: kv[1][0])
        out.check(worst[1][0] < 0.05, f"{det}: frame-induced bin migration < 5% for every "
                  f"variable (worst {worst[0]}: {100 * worst[1][0]:.1f}%)", warn=True)
        out.save(fig_frame(d, det, mig), f"frame_{det}")
        out.save(fig_phi(d, det, tables), f"phi_{det}")
        # FF
        qe = ((d["true_pdg"] == 14) & (d["true_isnc"] == 0) & (d["genie_mode"] == 0)
              & (d["w_ccqe"] > 0) & (d["cvwgt"] > 0))
        r = np.log((d["cvwgt"] / d["w_ccqe"])[qe])
        f = np.log(d["w_ff"][qe])
        cc = np.corrcoef(r, f)[0, 1]
        slope = np.polyfit(f, r, 1)[0]
        out.text(f"  divide_out_ff: nu_mu CC QE log(cvwgt/ccqe_corr) vs log(w_ff): corr "
                 f"{cc:.3f}, slope {slope:.3f}; median cvwgt/ccqe {np.exp(np.median(r)):.4f}, "
                 f"median w_ff {np.exp(np.median(f)):.4f}")
        out.text(f"  divide_out_ff: nu_mu CC QE with cvwgt == 0: "
                 f"{np.sum((d['true_pdg'] == 14) & (d['true_isnc'] == 0) & (d['genie_mode'] == 0) & (d['cvwgt'] == 0))}")
        contained = cc > 0.5 and 0.7 < slope < 1.4
        out.check(not contained,
                  f"{det}: default divide_out_ff: false consistent with the inputs, i.e. "
                  f"cvwgt does NOT contain the deut->MINERvA FF change (found corr "
                  f"{cc:.2f}, slope {slope:.2f}: FAIL = it does, so data/AR23 is applied "
                  "on top of an FF-modified CV)")
        out.save(fig_ff(d, det, None), f"divide_out_ff_{det}")
    return out.close()


if __name__ == "__main__":
    sys.exit(main())
