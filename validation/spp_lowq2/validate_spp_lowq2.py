#!/usr/bin/env python3
"""Validation plots for the jaesung_lowq2_pi_enhancement calculator.

The weight is CVCorr = GetSPPQ2Reweight(Q2) * GetSPPTpiMINERvAFittedReweight(
T_pi) for events passing IsSPP (fakedata/calculators/spp_lowq2.py, a port of
sbnana NuMIXSecSysts.cxx, see README). Two branches: _postfsi (signal from
true_npi / true_npi0 / true_g_p, T_pi from true_cpi_p) and _prefsi (signal
from the leading pre-FSI cpi / pi0 / g, T_pi from |genie_prefsi_cpi|). On the
CV MC sBruce files this script runs the real calculator and, per detector
(cvwgt-weighted), draws / checks:

  domain            w != 1 per flavour x CC/NC x mode, both branches (the
                    calculator has no CC / flavour / mode cut by design)
  composition       share of the weighted (signal) cvwgt per category: numu CC
                    RES / DIS / COH / QE / MEC, NC, nue -- how much of the
                    correction lands outside numu CC RES
  weights           weight distributions of both branches
  weight_vs_q2      w / TpiRW(T_pi) vs Q2 over the GetSPPQ2Reweight steps, and
  weight_vs_tpi     w / Q2RW(Q2) vs T_pi over the Landau + template curve
                    (factorization closure: every event on its curve)
  overlap           2x2 postfsi x prefsi signal matrix (events and cvwgt)
  photon_veto       true_g_p spectrum for post-FSI signal candidates (all cuts
                    but the photon veto), split by whether GENIE has a pre-FSI
                    (non-de-excitation) photon; the 10 MeV veto, and the
                    reference's effective 11 MeV veto (it casts genE*1000 to
                    int before "> 10.0")
  count_leading     true_npi / true_npi0 (G4ID>=0 & cryostat>=0 filtered)
                    vs the unfiltered leading-particle branches true_cpi_p /
                    true_pi0_p, inside the signal definition
  q2_check          Q2 = q3^2 - q0^2 vs Q2 rebuilt from the lepton
                    (2 E (E_l - p_l cos theta_l) - m_l^2 with the pre-FSI
                    lepton in the neutrino frame, E = nu_E), and q0 vs
                    E - E_l -- the reference uses SRTrueInteraction.Q2
                    (GENIE2ART's -q.M2() from the lepton 4-vectors)
  spectra           nominal vs weighted Q2, T_pi, E_nu of all events

Checks (PASS/WARN/FAIL; none aborts): calculator == independent
recomputation (bin tables typed in from NuMIXSecSysts.cxx here, Landau from
its integral representation, not fakedata.landau); w == 1 off-signal;
pion-mass choice (M_PION vs the reference's M_CHARGEDPION = 0.13957039);
Q2 equivalence; count/leading consistency; photon-veto details.

Reference facts (NuMIXSecSysts.cxx IsSPP / NuMIXSecTruthVars.cxx):
  - loops over nu.prim with start_process == 0, NO G4ID / cryostat filter;
  - photon: `int energy = prim.genE*1000.; if (pdg==22 && energy > 10.0)` --
    the int cast makes the veto genE >= 11 MeV;
  - nPip counts pdg == 211 only; n_mesons == 1 over pi+-, pi0, K, K*, eta, eta';
  - T_pi = genE - 0.13957039 of the leading-genE |pdg|==211 primary;
  - Q2 = SRTrueInteraction.Q2 (kTruth_Q2 = SIMPLETRUTHVAR(Q2)).
GENIE de-excitation photons are status-1 GHEP particles, hence G4 primaries
with start_process 0: the reference counts them exactly as true_g_p does.

Writes validation/plots/spp_lowq2/*.png, spp_lowq2.pdf and summary.txt.

Usage: ./venv/bin/python validation/spp_lowq2/validate_spp_lowq2.py
           [--max-files N] [files...]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

import common                                                # noqa: E402
from common import C, INK, INK2, NOM_COLOR, RW_COLOR         # noqa: E402

import fakedata.calculators  # noqa: E402,F401  (registers calculators)
from fakedata import calculator                             # noqa: E402
from fakedata.calculators.spp_lowq2 import BRANCH, M_PION    # noqa: E402
from fakedata.sbruce import valid                            # noqa: E402

NAME = "spp_lowq2"
B_POST = BRANCH + "_postfsi"
B_PRE = BRANCH + "_prefsi"

M_CHARGEDPION_REF = 0.13957039     # NuMIXSecTruthVars.h
M_MU, M_E = 0.1056583755, 0.00051099895

# NuMIXSecSysts.cxx, typed in independently of spp_lowq2.py
Q2_EDGES = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.3, 2.0, 3.0]
Q2_VALS = [1.253255, 1.589738, 1.733869, 1.651728, 1.659705, 1.584229,
           1.703793, 1.475510, 1.456727, 1.252215, 1.048199, 1.650489]
TPI_EDGES = [0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.4, 0.5, 0.7, 1.0, 2.0]
TPI_VALS = [0.755932, 0.638574, 0.493987, 0.391947, 0.323265, 0.452765,
            0.594541, 0.768459, 0.658024, 0.873622]
LANDAU = (6.70797696, 0.12235454, 0.05731087)

PRE = "genie_prefsi_"
BRANCHES = (["genie_q0", "genie_q3", "true_npi", "true_npi0", "true_g_p",
             "true_cpi_p", "true_pi0_p", "true_pdg", "true_isnc", "genie_mode",
             "cvwgt", "nu_E", "true_mu_p"]
            + [f"{PRE}{s}_{c}" for s in ("cpi", "pi0", "g", "lep")
               for c in ("px", "py", "pz")])


# ------------------------------------------------------------- reference

def ref_q2rw(q2):
    x = np.where(q2 >= 3.0, 2.5, q2)
    out = np.full(len(x), Q2_VALS[-1])
    lo = -np.inf
    for e, v in zip(Q2_EDGES, Q2_VALS):
        out[(x >= lo) & (x < e)] = v
        lo = e
    return out


_U = np.linspace(1e-12, 30.0, 30001)


def landau_integral(z):
    """p(z) = 1/pi int_0^inf exp(-u ln u - z u) sin(pi u) du (trapezoid)."""
    z = np.atleast_1d(z)
    base = -_U * np.log(_U)
    s = np.sin(np.pi * _U)
    f = np.exp(base[None, :] - z[:, None] * _U[None, :]) * s[None, :]
    return np.trapezoid(f, _U, axis=1) / np.pi


def ref_tpirw(t):
    out = np.full(len(t), TPI_VALS[-1])
    lo = TPI_EDGES[0]
    for e, v in zip(TPI_EDGES[1:], TPI_VALS):
        out[(t >= lo) & (t < e)] = v
        lo = e
    low = t < TPI_EDGES[0]
    if low.any():
        # evaluate on a fine grid, interpolate (the integral is slow per point)
        grid = np.linspace(min(t[low].min(), 0.0), TPI_EDGES[0], 2251)
        pg = LANDAU[0] * landau_integral((grid - LANDAU[1]) / LANDAU[2])
        out[low] = np.interp(t[low], grid, pg)
    return out


def mag(a, s):
    return np.sqrt(a[f"{PRE}{s}_px"].astype(np.float64) ** 2
                   + a[f"{PRE}{s}_py"].astype(np.float64) ** 2
                   + a[f"{PRE}{s}_pz"].astype(np.float64) ** 2)


def has(a, s):
    return valid(a[f"{PRE}{s}_px"], a[f"{PRE}{s}_py"], a[f"{PRE}{s}_pz"])


def process(sb):
    a = sb.arrays(BRANCHES)
    out = calculator.build({"type": "jaesung_lowq2_pi_enhancement"}).compute(sb)
    d = {k: np.asarray(v) for k, v in a.items()}
    d["w_post"] = out[B_POST]
    d["w_pre"] = out[B_PRE]
    q0 = a["genie_q0"].astype(np.float64)
    q3 = a["genie_q3"].astype(np.float64)
    d["has_q2"] = valid(a["genie_q0"], a["genie_q3"])
    d["q2"] = np.where(d["has_q2"], q3 ** 2 - q0 ** 2, np.nan)
    for s in ("cpi", "pi0", "g", "lep"):
        d[f"has_{s}"] = has(a, s)
        d[f"pre_{s}_p"] = np.where(d[f"has_{s}"], mag(a, s), np.nan)
    d["lep_cos"] = np.where(d["has_lep"],
                            a[f"{PRE}lep_pz"] / np.where(d["has_lep"], d["pre_lep_p"], 1), np.nan)
    for k in [k for k in d if k.startswith(PRE)]:
        del d[k]
    return d


# ------------------------------------------------------------- derived

def derive(d):
    cpi = d["true_cpi_p"].astype(np.float64)
    d["tpi_post"] = np.where(valid(cpi), np.sqrt(cpi ** 2 + M_PION ** 2) - M_PION, np.nan)
    pp = d["pre_cpi_p"]
    d["tpi_pre"] = np.where(d["has_cpi"], np.sqrt(pp ** 2 + M_PION ** 2) - M_PION, np.nan)
    gp = d["true_g_p"].astype(np.float64)
    # post-FSI signal pieces
    d["cand_post"] = ((d["true_npi"] == 1) & (d["true_npi0"] == 0)
                      & valid(d["true_cpi_p"]) & d["has_q2"])
    d["gveto"] = gp > 0.010
    d["gveto_ref"] = np.floor(gp * 1000.0) > 10.0      # int cast, > 10.0
    d["sig_post"] = d["cand_post"] & ~d["gveto"]
    pg = np.where(d["has_g"], d["pre_g_p"], 0.0)
    d["cand_pre"] = d["has_cpi"] & ~d["has_pi0"] & d["has_q2"]
    d["sig_pre"] = d["cand_pre"] & ~(pg > 0.010)
    return d


def recompute(d, tpi_key, sig_key, mass=M_PION):
    w = np.ones(len(d["cvwgt"]))
    m = d[sig_key]
    if tpi_key == "tpi_post":
        p = d["true_cpi_p"].astype(np.float64)
    else:
        p = d["pre_cpi_p"]
    t = np.sqrt(p[m] ** 2 + mass ** 2) - mass
    w[m] = ref_q2rw(d["q2"][m]) * ref_tpirw(t)
    return w


# ------------------------------------------------------------- plots

def comp_cats(d):
    pdg, isnc, mode = d["true_pdg"], d["true_isnc"], d["genie_mode"]
    numucc = (pdg == 14) & (isnc == 0)
    cats = []
    for m, mn in common.MODE_NAMES.items():
        cats.append((f"numu CC {mn}", numucc & (mode == m)))
    cats += [("numubar CC", (pdg == -14) & (isnc == 0)),
             ("nue/nuebar CC", (np.abs(pdg) == 12) & (isnc == 0)),
             ("NC (all flav.)", (pdg != -1) & (isnc == 1))]
    return cats


def fig_composition(d, det):
    cats = comp_cats(d)
    cv = d["cvwgt"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(cats))
    for j, (lab, sk, wk) in enumerate((("postfsi", "sig_post", "w_post"),
                                       ("prefsi", "sig_pre", "w_pre"))):
        tot = np.sum(cv[d[sk]])
        sh = [np.sum(cv[d[sk] & m]) / tot for _, m in cats]
        ax.barh(y + (j - 0.5) * 0.38, sh, 0.38, color=C[j], label=f"{lab} signal")
        for yy, s in zip(y, sh):
            ax.text(s + 0.005, yy + (j - 0.5) * 0.38, f"{100 * s:.1f}%",
                    va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [c for c, _ in cats])
    ax.invert_yaxis()
    ax.set_xlabel("share of signal cvwgt")
    ax.set_title(f"{det}: where the SPP correction lands (signal composition)",
                 fontsize=10, color=INK)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_weight_vs(d, det, rng):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    qg = np.logspace(-3, np.log10(5), 2000)
    tg = np.linspace(0.0, 2.2, 2000)
    for r, (lab, sk, wk, tk) in enumerate(
            (("postfsi", "sig_post", "w_post", "tpi_post"),
             ("prefsi", "sig_pre", "w_pre", "tpi_pre"))):
        idx = np.flatnonzero(d[sk])
        show = rng.choice(idx, min(len(idx), 15000), replace=False) if len(idx) else idx
        q2, t, w = d["q2"][show], d[tk][show], d[wk][show]
        ax = axs[r, 0]
        ax.plot(qg, ref_q2rw(qg), color=INK2, lw=1.5, label="GetSPPQ2Reweight")
        ax.scatter(q2, w / ref_tpirw(t), s=3, color=C[r], alpha=0.4, lw=0, label="w / TpiRW")
        ax.set_xscale("log")
        ax.set_xlabel("Q2 = q3^2 - q0^2 [GeV^2]")
        ax.set_ylabel("w / TpiRW(T_pi)")
        ax.set_title(f"{lab}: Q2 factor (N shown {len(show)})", fontsize=10, color=INK)
        ax.legend(fontsize=8)
        ax = axs[r, 1]
        ax.plot(tg, ref_tpirw(tg), color=INK2, lw=1.5, label="Landau + template")
        ax.scatter(t, w / ref_q2rw(q2), s=3, color=C[r], alpha=0.4, lw=0, label="w / Q2RW")
        ax.set_xlabel("T_pi [GeV]")
        ax.set_ylabel("w / Q2RW(Q2)")
        ax.set_title(f"{lab}: T_pi factor", fontsize=10, color=INK)
        ax.legend(fontsize=8)
    fig.suptitle(f"{det}: factorized weight vs the reference curves", color=INK)
    fig.tight_layout()
    return fig


def fig_overlap(d, det):
    cv = d["cvwgt"]
    sp, sq = d["sig_post"], d["sig_pre"]
    cells = [[sp & sq, sp & ~sq], [~sp & sq, ~sp & ~sq]]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (lab, fn) in zip(axs, (("events", lambda m: m.sum()),
                                   ("cvwgt", lambda m: cv[m].sum()))):
        mat = np.array([[fn(c) for c in row] for row in cells], dtype=float)
        frac = mat / mat.sum()
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=min(1, frac[:, :].max() * 1.2 + 1e-9))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.0f}\n({100 * frac[i, j]:.2f}%)",
                        ha="center", va="center", fontsize=9, color=INK)
        ax.set_xticks([0, 1], ["prefsi signal", "prefsi not"])
        ax.set_yticks([0, 1], ["postfsi signal", "postfsi not"])
        ax.grid(False)
        ax.set_title(lab, fontsize=10, color=INK)
    fig.suptitle(f"{det}: postfsi vs prefsi SPP signal overlap", color=INK)
    fig.tight_layout()
    return fig


def fig_photon(d, det):
    cand = d["cand_post"]
    gp = d["true_g_p"].astype(np.float64) * 1000.0
    preg = d["has_g"] & (np.nan_to_num(d["pre_g_p"]) > 0.010)
    bins = np.logspace(-1, 3, 81)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axs[0]
    for lab, m, col in (("no pre-FSI photon > 10 MeV", cand & ~preg, C[0]),
                        ("pre-FSI photon > 10 MeV", cand & preg, C[1])):
        ax.hist(gp[m & (gp > 0)], bins, weights=d["cvwgt"][m & (gp > 0)],
                histtype="step", lw=1.8, color=col, label=f"{lab} (N={m.sum()})")
    ax.axvline(10, color=INK, ls="--", lw=1, label="calc veto 10 MeV")
    ax.axvline(11, color=INK2, ls=":", lw=1, label="ref (int cast) 11 MeV")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true_g_p (leading final-state photon) [MeV]")
    ax.set_ylabel("cvwgt")
    ax.set_title("post-FSI SPP candidates (all cuts but the photon veto)",
                 fontsize=9, color=INK)
    ax.legend(fontsize=7)
    ax = axs[1]
    # same for the prefsi-signal events: are they post-FSI vetoed by photons?
    m = d["sig_pre"]
    ax.hist(gp[m & (gp > 0)], bins, weights=d["cvwgt"][m & (gp > 0)],
            histtype="step", lw=1.8, color=C[2], label="prefsi-signal events")
    m = d["sig_post"] | (cand & d["gveto"])
    ax.hist(gp[cand & (gp > 0)], bins, weights=d["cvwgt"][cand & (gp > 0)],
            histtype="step", lw=1.8, color=C[0], label="post-FSI candidates")
    ax.axvline(10, color=INK, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true_g_p [MeV]")
    ax.set_title("true_g_p, prefsi signal vs post-FSI candidates", fontsize=9, color=INK)
    ax.legend(fontsize=7)
    fig.suptitle(f"{det}: SPP photon veto", color=INK)
    fig.tight_layout()
    return fig


def fig_q2(d, det):
    lep = d["has_lep"] & d["has_q2"] & (d["true_pdg"] != -1)
    isnc = d["true_isnc"] == 1
    ml = np.where(isnc, 0.0, np.where(np.abs(d["true_pdg"]) == 14, M_MU, M_E))
    E = d["nu_E"].astype(np.float64)
    pl = d["pre_lep_p"]
    El = np.sqrt(pl ** 2 + ml ** 2)
    q2l = 2 * E * (El - pl * d["lep_cos"]) - ml ** 2
    q0l = E - El
    q0 = d["genie_q0"].astype(np.float64)
    r1 = np.where(lep, (d["q2"] - q2l), np.nan)
    r2 = np.where(lep, q0 - q0l, np.nan)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, r, xl in ((axs[0], r1, "Q2(q3,q0) - Q2(lepton) [GeV^2]"),
                      (axs[1], r2, "genie_q0 - (nu_E - E_lep) [GeV]")):
        rr = r[lep]
        lim = max(1e-6, np.nanpercentile(np.abs(rr), 99.9) * 1.5)
        ax.hist(np.clip(rr, -lim, lim), np.linspace(-lim, lim, 101),
                histtype="step", color=C[0], lw=1.8)
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_title(f"N={lep.sum()}, 99.9% |r| < {np.nanpercentile(np.abs(rr), 99.9):.1e}",
                     fontsize=9, color=INK)
    fig.suptitle(f"{det}: Q2 = q3^2 - q0^2 vs -q^2 from the lepton", color=INK)
    fig.tight_layout()
    return fig, r1, r2, lep


def fig_spectra(d, det):
    cv = d["cvwgt"]
    nu = d["true_pdg"] != -1
    specs = [("q2", "Q2 [GeV^2]", np.linspace(0, 1.5, 31)),
             ("tpi_post", "post-FSI T_pi [GeV]", np.linspace(0, 1.0, 26)),
             ("nu_E", "E_nu [GeV]", np.linspace(0, 3, 31))]
    fig, axs = plt.subplots(2, 3, figsize=(14, 6.5), sharex="col",
                            gridspec_kw=dict(height_ratios=[2.5, 1]))
    for c, (k, xl, bins) in enumerate(specs):
        x = np.nan_to_num(d[k].astype(np.float64), nan=-1.0)
        m = nu
        common.ratio_panel(axs[0, c], axs[1, c], x[m], bins, cv[m], d["w_post"][m],
                           [("postfsi", np.ones(m.sum(), bool), C[0])], xl)
        axr = axs[1, c]
        h0, _ = np.histogram(x[m], bins, weights=cv[m])
        h1, _ = np.histogram(x[m], bins, weights=(cv * d["w_pre"])[m])
        axr.stairs(np.divide(h1, h0, out=np.full(len(h0), np.nan), where=h0 > 0),
                   bins, color=C[1], lw=1.5, label="prefsi", baseline=None)
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=8)
    fig.suptitle(f"{det}: nominal vs weighted (_postfsi on top; all nu events)",
                 color=INK)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------- main

def pct(m, cv, base):
    return f"{m.sum()} ({100 * cv[m].sum() / max(cv[base].sum(), 1e-12):.2f}% cvwgt)"


def run_det(out, det, files, rng):
    d = derive(common.load(files, process))
    cv = d["cvwgt"]
    n = len(cv)
    out.text(f"== {det}: {len(files)} files, {n} slices")
    for b, w in ((B_POST, d["w_post"]), (B_PRE, d["w_pre"])):
        common.weight_summary(out, w, cv, b)

    # -- recomputation
    for lab, wk, tk, sk in (("postfsi", "w_post", "tpi_post", "sig_post"),
                            ("prefsi", "w_pre", "tpi_pre", "sig_pre")):
        wr = recompute(d, tk, sk)
        dw = np.max(np.abs(d[wk] - wr))
        out.check(dw < 1e-4, f"{lab}: calculator == independent recomputation "
                             f"(max |dw| = {dw:.2e})")
        out.check(bool(np.all(d[wk][~d[sk]] == 1.0)),
                  f"{lab}: w == 1 exactly outside the signal "
                  f"({np.sum(d[wk] != 1)} vs {d[sk].sum()} signal)")
        wm = recompute(d, tk, sk, mass=M_CHARGEDPION_REF)
        out.text(f"    {lab}: pion mass {M_PION} vs ref {M_CHARGEDPION_REF}: "
                 f"max |dw| = {np.max(np.abs(wm - wr)):.1e}")
        # events sitting exactly on a step edge would flip with float noise
        tpi = d[tk][d[sk]]
        out.text(f"    {lab}: T_pi < 0.225 GeV (Landau region): "
                 f"{100 * np.mean(tpi < 0.225):.1f}% of signal; "
                 f"Q2 >= 3 GeV^2: {np.sum(d['q2'][d[sk]] >= 3)}")

    # -- composition
    out.text("  signal composition (share of signal cvwgt):")
    for lab, sk in (("postfsi", "sig_post"), ("prefsi", "sig_pre")):
        tot = cv[d[sk]].sum()
        parts = ", ".join(f"{c} {100 * cv[d[sk] & m].sum() / tot:.1f}%"
                          for c, m in comp_cats(d) if cv[d[sk] & m].sum() > 0)
        out.text(f"    {lab}: {parts}")
    nc_share = cv[d["sig_post"] & (d["true_isnc"] == 1)].sum() / cv[d["sig_post"]].sum()
    out.check(nc_share < 0.05, f"postfsi: NC share of signal {100 * nc_share:.1f}% "
              "(by design: no CC requirement; the template was fit to numu CC)",
              warn=True)

    # -- overlap
    sp, sq = d["sig_post"], d["sig_pre"]
    out.text(f"  overlap: both {pct(sp & sq, cv, sp | sq)}, post only "
             f"{pct(sp & ~sq, cv, sp | sq)}, pre only {pct(~sp & sq, cv, sp | sq)} "
             "(% of union)")
    diffw = np.abs(d["w_post"] - d["w_pre"]) > 1e-9
    out.text(f"  weights differ on {100 * diffw.mean():.1f}% of slices")

    # -- photon veto
    cand = d["cand_post"]
    gv = cand & d["gveto"]
    preg = d["has_g"] & (np.nan_to_num(d["pre_g_p"]) > 0.010)
    out.text(f"  photon veto: post-FSI candidates {cand.sum()}, vetoed by "
             f"true_g_p > 10 MeV: {pct(gv, cv, cand)}")
    out.text(f"    of which without any pre-FSI photon > 10 MeV (de-excitation / "
             f"FSI origin): {pct(gv & ~preg, cv, cand)}")
    gp = d["true_g_p"]
    band = gv & ~d["gveto_ref"]
    out.text(f"    true_g_p in (10, 11) MeV, vetoed here but not by the reference's "
             f"int cast: {pct(band, cv, cand)}")
    out.text(f"    candidates with 0 < true_g_p <= 10 MeV (de-excitation range, kept): "
             f"{pct(cand & (gp > 0) & (gp <= 0.010), cv, cand)}; median true_g_p "
             f"{1000 * np.median(gp[cand & (gp > 0)]) if np.any(cand & (gp > 0)) else 0:.2f} MeV")
    out.check(band.sum() <= 0.001 * max(cand.sum(), 1),
              "photon veto threshold matches reference int cast (<0.1% of candidates "
              "in the 10-11 MeV band)", warn=True)

    # -- counts vs leading
    sig_like = (d["true_npi0"] == 0) & d["has_q2"] & ~d["gveto"]
    c1 = sig_like & (d["true_npi"] == 0) & valid(d["true_cpi_p"])
    c2 = (d["true_npi"] == 1) & ~valid(d["true_cpi_p"]) & (d["true_pdg"] != -1)
    c3 = d["sig_post"] & valid(d["true_pi0_p"])
    c4 = (d["true_npi"] >= 2) & (d["true_npi0"] == 0) & d["has_q2"]
    out.text(f"  counts vs leading branches (G4ID/cryostat filter on counts only):")
    out.text(f"    npi==0 but true_cpi_p filled (would-be signal, rejected; ref "
             f"counts that pion): {c1.sum()}")
    out.text(f"    npi==1 but true_cpi_p unfilled (rejected by valid()): {c2.sum()}")
    out.text(f"    postfsi signal with npi0==0 but true_pi0_p filled (ref would veto "
             f"the pi0): {c3.sum()}")
    out.check(c1.sum() + c3.sum() <= 1e-3 * max(d["sig_post"].sum(), 1),
              "count/leading inconsistencies < 0.1% of postfsi signal", warn=True)

    # -- Q2
    fq, r1, r2, lep = fig_q2(d, det)
    q99 = np.nanpercentile(np.abs(r1[lep]), 99)
    out.check(q99 < 1e-3, f"Q2 = q3^2-q0^2 equals -q^2 from the lepton "
              f"(99% |dQ2| = {q99:.1e} GeV^2, max {np.nanmax(np.abs(r1[lep])):.1e})",
              warn=True)
    sigl = lep & sp
    if sigl.any():
        out.text(f"    on postfsi signal: 99% |dQ2| = "
                 f"{np.nanpercentile(np.abs(r1[sigl]), 99):.1e}, Q2 bin changes "
                 f"{np.sum(ref_q2rw(d['q2'][sigl]) != ref_q2rw(d['q2'][sigl] - r1[sigl]))}")

    figs = {"domain": common.fig_domain(d, {"postfsi": d["w_post"], "prefsi": d["w_pre"]},
                                        det, "jaesung_lowq2_pi_enhancement"),
            "composition": fig_composition(d, det),
            "weights": common.fig_weight_hists({"postfsi": d["w_post"],
                                                "prefsi": d["w_pre"]}, det, "SPP"),
            "weight_vs": fig_weight_vs(d, det, rng),
            "overlap": fig_overlap(d, det),
            "photon_veto": fig_photon(d, det),
            "q2_check": fq,
            "spectra": fig_spectra(d, det)}
    for k, f in figs.items():
        out.save(f, f"{k}_{det}")
    out.text("")


def main():
    args = common.parse_args(__doc__.splitlines()[0])
    out = common.Output(NAME, "jaesung_lowq2_pi_enhancement validation "
                              "(validation/spp_lowq2/validate_spp_lowq2.py)")
    rng = np.random.default_rng(1)
    for det, files in common.files_by_detector(args).items():
        run_det(out, det, files, rng)
    return out.close()


if __name__ == "__main__":
    sys.exit(main())
