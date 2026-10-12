#!/usr/bin/env python3
"""Validation plots for osc_sterile_ic2024 and osc_sterile_ic2024_unfolded.

osc_sterile_ic2024 (fakedata/calculators/osc_sterile.py) weights every
nu_mu / nu_mu-bar event (CC and NC; true_pdg is the unoscillated flavour) by
the 3+1 survival probability

    P = 1 - sin^2(2 theta_24) sin^2(1.267 dm2[eV^2] L[km] / E[GeV])

at a FIXED L = 110 m for SBND and ICARUS alike (documented choice), E =
true_E. osc_sterile_ic2024_unfolded applies a tabulated piecewise-constant
true-E weight built offline (scripts/build_osc_ic2024_unfolded.py) by
unfolding the SBND reco-E (nu_E_calo) signature of the oscillation at each
event's own baseline. Per detector, cvwgt-weighted, this script runs both
calculators and draws:

  domain            w != 1 by flavour x CC/NC x mode, six branches
                    (expected: nu_mu / nu_mu-bar CC AND NC only)
  weight_hists      weight distributions
  weight_vs_E       per point: event weights vs true_E over the analytic P
                    at 110 m (the calculator's curve), the P each event
                    would get at its OWN baseline (scatter), and the
                    unfolded step function
  baseline          the `baseline` branch (metres) per detector
  icarus_L          per point: cvwgt-weighted nu_mu true_E spectrum
                    weighted by P(110 m) vs P(own baseline), and their
                    ratio -- the size of the fixed-110 m choice (biggest for
                    ICARUS, ~600 m)
  unfolded_vs_direct  per point: the unfolded table vs the direct true-E
                    ratio sum(cv P(own L)) / sum(cv) in the same true bins
                    (SBND = what the table should approximate) and vs the
                    110 m direct ratio

Checks: calculator == independent recomputation; w == 1 off nu_mu /
nu_mu-bar and on cosmics; NC nu_mu weighted; true_E == genie_Enu == nu_E;
baseline in metres (SBND 50-140: decay-pipe origin, ICARUS 520-620); the 1.267 constant and
units (vs hbar*c from first principles, and the weight at the first
oscillation maximum equals 1 - sin^2 2theta); unfolded table within a few %
of the direct ratio in the well-constrained 0.3-1.25 GeV bins.

Writes validation/plots/osc_sterile/*.png, osc_sterile.pdf and summary.txt.

Usage: ./venv/bin/python validation/osc_sterile/validate_osc.py
           [--max-files N] [files...]
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

import common                                                # noqa: E402
from common import C, DET_COLOR, INK, INK2, NOM_COLOR, plt   # noqa: E402

from fakedata.calculator import build                        # noqa: E402
from fakedata.calculators.osc_sterile import (               # noqa: E402
    BASELINE_KM, OSC_CONST, POINTS, UNFOLD_TRUE_EDGES, UNFOLD_WEIGHTS)

BRANCHES = ["true_pdg", "true_isnc", "genie_mode", "true_E", "genie_Enu",
            "nu_E", "baseline", "cvwgt"]
HBARC_EV_M = 1.973269804e-7          # hbar c [eV m]
E_BINS = np.linspace(0, 3, 61)


def p_surv(e, dm2, s22, l_km):
    """Independent recomputation (first principles: Delta = dm2 L / 4E with
    natural units, here written out without the 1.267 shortcut)."""
    phase = dm2 * (l_km * 1e3) / (4.0 * e * 1e9 * HBARC_EV_M)
    return 1.0 - s22 * np.sin(phase) ** 2


def process(sb):
    a = sb.arrays(BRANCHES)
    d = {k: a[k] for k in BRANCHES}
    d["cvwgt"] = d["cvwgt"].astype(np.float64)
    w = build({"type": "osc_sterile_ic2024"}).compute(sb)
    wu = build({"type": "osc_sterile_ic2024_unfolded"}).compute(sb)
    for p in POINTS:
        d[f"w_{p}"] = w[f"fdwgt_osc_ic2024_{p}"]
        d[f"u_{p}"] = wu[f"fdwgt_osc_ic2024_unf_{p}"]
    return d


def numu(d):
    return np.isin(d["true_pdg"], (14, -14)) & (d["true_E"] > 0)


# ---------------------------------------------------------------- plots

def fig_weight_vs_E(d, det, rng):
    m = np.flatnonzero(numu(d))
    show = rng.choice(m, min(len(m), 15000), replace=False)
    e = d["true_E"][show].astype(np.float64)
    L = d["baseline"][show] / 1000.0
    ee = np.geomspace(0.05, 10, 600)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (p, (dm2, s22)) in zip(axs, POINTS.items()):
        ax.scatter(e, p_surv(e, dm2, s22, L), s=3, lw=0, alpha=0.3,
                   color=C[1], label="P at the event's own baseline")
        ax.scatter(e, d[f"w_{p}"][show], s=5, lw=0, color=C[0],
                   label="osc_sterile_ic2024 weight")
        ax.plot(ee, p_surv(ee, dm2, s22, BASELINE_KM), color=INK, lw=1,
                label="analytic P(110 m)")
        ue = UNFOLD_TRUE_EDGES.copy()
        ue[-1] = 10
        ax.stairs(UNFOLD_WEIGHTS[p], ue, color=C[2], lw=2, baseline=None,
                  label="unfolded table")
        ax.set_xscale("log")
        ax.set_xlim(0.05, 10)
        ax.set_xlabel("true_E [GeV]")
        ax.set_title(f"{p}: dm2={dm2}, s22={s22}", fontsize=10)
    axs[0].set_ylabel("weight / P_survival")
    axs[0].legend(fontsize=7, loc="lower right", markerscale=3)
    fig.suptitle(f"{det}: nu_mu(bar) weights vs true E", color=INK)
    fig.tight_layout()
    return fig


def fig_baseline(dd):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for det, d in dd.items():
        m = numu(d) & (d["baseline"] > 0)
        ax.hist(d["baseline"][m], np.linspace(0, 700, 141), histtype="step",
                lw=2, color=DET_COLOR[det], label=f"{det} (N={m.sum()})")
    ax.axvline(BASELINE_KM * 1e3, color=INK2, ls="--", lw=1,
               label="calculator L = 110 m")
    ax.set_yscale("log")
    ax.set_xlabel("baseline branch [m]")
    ax.set_ylabel("nu_mu(bar) events")
    ax.legend(fontsize=8)
    ax.set_title("baseline per detector", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def fig_own_L(d, det):
    m = numu(d) & (d["baseline"] > 0)
    e = d["true_E"][m].astype(np.float64)
    cv = d["cvwgt"][m]
    L = d["baseline"][m] / 1000.0
    fig, axs = plt.subplots(2, 3, figsize=(16, 7), sharex=True,
                            gridspec_kw=dict(height_ratios=[2.2, 1]))
    h0, _ = np.histogram(e, E_BINS, weights=cv)
    for j, (p, (dm2, s22)) in enumerate(POINTS.items()):
        ax, axr = axs[0, j], axs[1, j]
        hf, _ = np.histogram(e, E_BINS, weights=cv * d[f"w_{p}"][m])
        ho, _ = np.histogram(e, E_BINS, weights=cv * p_surv(e, dm2, s22, L))
        hu, _ = np.histogram(e, E_BINS, weights=cv * d[f"u_{p}"][m])
        ax.stairs(h0, E_BINS, color=NOM_COLOR, lw=2, label="null")
        ax.stairs(hf, E_BINS, color=C[0], lw=1.6, label="calculator (110 m)")
        ax.stairs(ho, E_BINS, color=C[1], lw=1.6, label="own baseline")
        ax.stairs(hu, E_BINS, color=C[2], lw=1.6, label="unfolded table")
        ax.set_title(f"{p}", fontsize=10)
        for h, col in ((hf, C[0]), (ho, C[1]), (hu, C[2])):
            axr.stairs(np.divide(h, h0, out=np.full(len(h0), np.nan),
                                 where=h0 > 0), E_BINS, color=col, lw=1.5,
                       baseline=None)
        axr.axhline(1, color=INK2, lw=0.8, ls=":")
        axr.set_xlabel("true_E [GeV]")
    axs[0, 0].set_ylabel("cvwgt nu_mu(bar) events")
    axs[1, 0].set_ylabel("/ null")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"{det}: fixed 110 m vs each event's own baseline", color=INK)
    fig.tight_layout()
    return fig


def direct_ratio(d, p, own=True):
    m = numu(d) & (d["baseline"] > 0)
    e = d["true_E"][m].astype(np.float64)
    cv = d["cvwgt"][m]
    dm2, s22 = POINTS[p]
    L = d["baseline"][m] / 1000.0 if own else BASELINE_KM
    h0, _ = np.histogram(e, UNFOLD_TRUE_EDGES, weights=cv)
    h1, _ = np.histogram(e, UNFOLD_TRUE_EDGES, weights=cv * p_surv(e, dm2, s22, L))
    return np.divide(h1, h0, out=np.full(len(h0), np.nan), where=h0 > 0)


def fig_unfolded(d, det):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    ue = UNFOLD_TRUE_EDGES.copy()
    ue[-1] = 4.0
    for ax, p in zip(axs, POINTS):
        ax.stairs(UNFOLD_WEIGHTS[p], ue, color=C[2], lw=2, baseline=None,
                  label="unfolded table")
        ax.stairs(direct_ratio(d, p, True), ue, color=C[1], lw=1.6,
                  baseline=None, label="direct ratio, own baseline")
        ax.stairs(direct_ratio(d, p, False), ue, color=C[0], lw=1.6, ls="--",
                  baseline=None, label="direct ratio, 110 m")
        ax.axvspan(0.3, 1.25, color=common.GRID, alpha=0.35, lw=0)
        ax.set_xlabel("true_E [GeV] (last bin open, drawn to 4)")
        ax.set_title(p, fontsize=10)
    axs[0].set_ylabel("weight per true-E bin")
    axs[0].legend(fontsize=8)
    fig.suptitle(f"{det}: unfolded table vs direct true-E ratio "
                 "(shaded: well-constrained bins)", color=INK)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- main

def main():
    args = common.parse_args(__doc__.splitlines()[0])
    out = common.Output("osc_sterile", "osc_sterile_ic2024 / _unfolded "
                        "(validation/osc_sterile/validate_osc.py)")
    rng = np.random.default_rng(1)

    out.text("units: 1.267 = 1/(4 hbar c) in eV^2 km / GeV")
    k = 1e3 / (4 * 1e9 * HBARC_EV_M)
    out.check(abs(k - OSC_CONST) / k < 1e-3,
              f"OSC_CONST {OSC_CONST} vs 1e3 m/(4 * 1e9 eV * hbar c) = {k:.5f}")

    dd = {}
    for det, files in common.files_by_detector(args).items():
        out.text(f"\n== {det}: {len(files)} files")
        d = common.load(files, process)
        dd[det] = d
        cv = d["cvwgt"]
        W = {}
        for p in POINTS:
            W[f"{p} (110 m)"] = d[f"w_{p}"]
        for p in POINTS:
            W[f"{p} unf"] = d[f"u_{p}"]
        out.save(common.fig_domain(d, W, det, "osc_sterile"), f"domain_{det}")
        out.save(common.fig_weight_hists(W, det, "osc_sterile"), f"weight_hists_{det}")
        out.save(fig_weight_vs_E(d, det, rng), f"weight_vs_E_{det}")
        out.save(fig_own_L(d, det), f"own_baseline_{det}")
        out.save(fig_unfolded(d, det), f"unfolded_vs_direct_{det}")

        nu = d["true_pdg"] != -1
        mu = numu(d)
        for lab, w in W.items():
            common.weight_summary(out, w, cv, lab)

        # energy branches
        e, g, n = (d[x].astype(np.float64) for x in ("true_E", "genie_Enu", "nu_E"))
        out.check(bool(np.all(e[nu] == g[nu]) and np.all(e[nu] == n[nu])),
                  f"true_E == genie_Enu == nu_E on all {nu.sum()} neutrino rows "
                  f"(max |dE| {np.max(np.abs(e[nu] - g[nu])):.2e}, "
                  f"{np.max(np.abs(e[nu] - n[nu])):.2e})")
        out.check(bool(np.all(e[nu] > 0)), "true_E > 0 on every neutrino row")
        bl = d["baseline"][nu]
        lo, hi = (50, 140) if det == "SBND" else (520, 620)
        out.text(f"  baseline [m]: min {bl.min():.1f} median {np.median(bl):.1f} "
                 f"max {bl.max():.1f}")
        out.check(bool(np.all((bl > lo) & (bl < hi))),
                  f"baseline in metres, within [{lo}, {hi}]")

        # closure vs independent recomputation
        for p, (dm2, s22) in POINTS.items():
            ref = np.ones(len(e))
            ref[mu] = p_surv(e[mu], dm2, s22, BASELINE_KM)
            dw = np.max(np.abs(d[f"w_{p}"] - ref))
            out.check(dw < 1e-3, f"{p}: calculator == first-principles P(110 m) "
                      f"(max |dw| {dw:.1e}; 1.267 vs 1.26693 rounding)")
            e1 = OSC_CONST * dm2 * BASELINE_KM / (np.pi / 2)
            from fakedata.calculators.osc_sterile import numu_survival
            out.check(abs(numu_survival(e1, dm2, s22) - (1 - s22)) < 1e-12,
                      f"{p}: weight at first max E = {1000 * e1:.0f} MeV is 1 - s22")

        # domain
        off = ~mu
        for lab, w in W.items():
            out.check(bool(np.all(w[off] == 1.0)),
                      f"{lab}: w == 1 off nu_mu(bar) (nue, cosmics)")
        for lab, sel in (("numu CC", mu & (d["true_isnc"] == 0)),
                         ("numu NC", mu & (d["true_isnc"] == 1))):
            if sel.any():
                f = np.mean(d["w_bestfit"][sel] != 1.0)
                out.check(f > 0.99, f"bestfit (110 m): {lab} weighted ({100 * f:.2f}%)")

        # size of the fixed 110 m choice
        m = mu & (d["baseline"] > 0)
        L = d["baseline"][m] / 1000.0
        for p, (dm2, s22) in POINTS.items():
            wf = d[f"w_{p}"][m]
            wo = p_surv(e[m], dm2, s22, L)
            c = cv[m]
            sf, so = np.sum(c * wf) / np.sum(c), np.sum(c * wo) / np.sum(c)
            h0, _ = np.histogram(e[m], E_BINS, weights=c)
            hf, _ = np.histogram(e[m], E_BINS, weights=c * wf)
            ho, _ = np.histogram(e[m], E_BINS, weights=c * wo)
            good = h0 > 0.002 * h0.sum()
            rel = (hf[good] - ho[good]) / ho[good]
            su = np.sum(c * d[f"u_{p}"][m]) / np.sum(c)
            out.text(f"  {p}: nu_mu(bar) cvwgt-mean P: 110 m {sf:.4f}, own baseline "
                     f"{so:.4f}, unfolded {su:.4f}; per-0.05-GeV-bin (110 m - own)/own "
                     f"in [{100 * rel.min():+.1f}%, {100 * rel.max():+.1f}%]")
            if det == "ICARUS":
                out.check(np.max(np.abs(rel)) < 0.02,
                          f"{p}: ICARUS fixed-110 m spectrum within 2% of the "
                          f"real-baseline one (documented choice; size "
                          f"{100 * np.max(np.abs(rel)):.1f}%)", warn=True)

        # unfolded table vs direct ratio
        cen = (UNFOLD_TRUE_EDGES[:-1] >= 0.3) & (UNFOLD_TRUE_EDGES[1:] <= 1.25)
        for p in POINTS:
            dr = direct_ratio(d, p, True)
            diff = UNFOLD_WEIGHTS[p] - dr
            out.text(f"  {p} unfolded - direct(own L) per true bin: "
                     + " ".join(f"{x:+.3f}" for x in diff))
            if det == "SBND":
                out.check(np.nanmax(np.abs(diff[cen])) < 0.03,
                          f"{p}: unfolded table within 0.03 of the direct ratio in "
                          f"0.3-1.25 GeV (max {np.nanmax(np.abs(diff[cen])):.3f})",
                          warn=True)
    out.save(fig_baseline(dd), "baseline")
    sys.exit(out.close())


if __name__ == "__main__":
    main()
