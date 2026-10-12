#!/usr/bin/env python3
"""Is the ccqe_xsec_corr fake-data weight already inside cvwgt?

ccqe_xsec_corr applies the nusystematics CCQEXSecCorr correction (GENIE
v3_04_02a -> v3_06_02 CCQE bugfix, fakedata/calculators/ccqe_corr.py) as a
fake-data weight on top of the CV MC. But the sBruce `cvwgt` is built by
cafpyana analysis_village/gump/loaddf.py (reweight_aFF, called by
gumple/run_gumple_pipeline.py with --weights) as the product of the CAF
weights

    xsec_cv_rwgt = [ZExpPCAWeighter_SBN_v3_MvA_b1,      ("cv" column)
                    CCQEXSecCorr_SBN_v3_CCQEXSecCorr,    ("morph": nuniv 1)
                    GENIEReWeight_SBN_v3_FrKin_PiProFix_N]

i.e. it already CONTAINS a CCQEXSecCorr weight. If so, reweighting the CV
MC by fdwgt_ccqe_xsec_corr applies the bugfix twice. This script tests it on
the data: on CC QE events it divides cvwgt by the deuterium->MINERvA ZExp
CV weight (computed with the validated Nieves port, exactly the
xsec_meas.py divide_out_ff path, fakedata.calculators.qe_zexp
.deut_to_minerva_weight; extended here to nu_e / nu_e-bar with the electron
mass) and compares the remainder r = cvwgt / zexp_cv with the calculator's
weight in both modes. Plots per detector (cvwgt-weighted where a rate):

  domain            w != 1 by flavour x CC/NC x mode for cvwgt, zexp_cv,
                    r, ccqe_xsec_corr (match_tool) and (default)
  weight_hists      the same weights' distributions
  scatter           r vs calculator weight (match_tool / default), CC QE
                    nu_mu; the diagonal is "cvwgt already contains it"
  ratio_hist        r / w for both modes, and r / 1 (the "cvwgt holds
                    only ZExp" hypothesis), per CC QE flavour
  binned_means      <r> vs <w> in coarse (E_nu, q3, q0) cells (>= 30
                    events): agreement of the AVERAGES even where the
                    per-event values scatter
  nonqe_cvwgt       cvwgt per mode for non-QE / NC events (FrKin only:
                    expected 1)
  q3q0              mean r, mean w(match_tool) and their ratio vs (q3, q0)
  enu               mean cvwgt, zexp_cv, r, w(tool), w(default) vs E_nu

Checks: the code-path facts above (printed); fraction of nu_mu CC QE with
|r / w - 1| < 1%; coarse-cell agreement; hyperon events (match_tool weight
0) vs cvwgt; cvwgt == 1 off CC QE.

Writes validation/plots/ccqe_cvwgt/*.png, ccqe_cvwgt.pdf and summary.txt.

Usage: ./venv/bin/python validation/ccqe_xsec_corr/validate_ccqe_cvwgt.py
           [--max-files N] [files...]
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

import common                                                # noqa: E402
from common import C, INK, INK2, GRID, plt                   # noqa: E402

from fakedata.calculator import build                        # noqa: E402
from fakedata.calculators.qe_zexp import (QE_BRANCHES, _ff,  # noqa: E402
                                          deut_to_minerva_weight)
from fakedata.nieves import NievesQEXSec                     # noqa: E402
from fakedata.sbruce import MODE_QE, valid                   # noqa: E402

M_E = 0.51099895e-3
BRANCHES = ["cvwgt", "genie_mode", "true_isnc", "true_pdg", "genie_Enu",
            "genie_q0", "genie_q3"]
FLAV = {14: "numu", -14: "numubar", 12: "nue", -12: "nuebar"}
FLAV_COLOR = {14: C[0], -14: C[1], 12: C[2], -12: C[3]}


def nue_zexp(a, n):
    """deut->MINERvA Nieves weight for nu_e / nu_e-bar CC QE (electron
    mass), same convention as deut_to_minerva_weight (which is nu_mu only)."""
    w = np.ones(n)
    xs = NievesQEXSec(m_lep=M_E)
    fa_d, fa_m = _ff("deuterium", "nusyst"), _ff("minerva_nature", "nusyst")
    for pdg, rec in ((12, "p"), (-12, "n")):
        m = ((a["genie_mode"] == MODE_QE) & (a["true_isnc"] == 0)
             & (a["true_pdg"] == pdg)
             & valid(a["genie_Enu"], a["genie_prefsi_lep_py"],
                     a["genie_prefsi_lep_pz"], a[f"genie_prefsi_{rec}_px"],
                     a[f"genie_prefsi_{rec}_py"], a[f"genie_prefsi_{rec}_pz"]))
        if not m.any():
            continue
        enu = a["genie_Enu"][m].astype(np.float64)
        pl = np.stack([a[f"genie_prefsi_lep_p{c}"][m] for c in "xyz"],
                      1).astype(np.float64)
        pn = np.stack([a[f"genie_prefsi_{rec}_p{c}"][m] for c in "xyz"],
                      1).astype(np.float64)
        w[m] = xs.weight_ratio(enu, pl, pn, fa_m, fa_d, is_neutrino=pdg > 0)
    return w


def process(sb):
    a = sb.arrays(BRANCHES + QE_BRANCHES)
    n = sb.n_entries
    z = deut_to_minerva_weight(sb)
    zn = nue_zexp(a, n)
    z = np.where(np.abs(a["true_pdg"]) == 12, zn, z)
    wt = build({"type": "ccqe_xsec_corr", "match_tool": True}).compute(sb)
    wd = build({"type": "ccqe_xsec_corr"}).compute(sb)
    cv = a["cvwgt"].astype(np.float64)
    return dict(cvwgt=cv, zexp=z, r=cv / z,
                w_tool=wt["fdwgt_ccqe_xsec_corr"], w_def=wd["fdwgt_ccqe_xsec_corr"],
                true_pdg=a["true_pdg"], true_isnc=a["true_isnc"],
                genie_mode=a["genie_mode"],
                E=a["genie_Enu"].astype(np.float64),
                q0=a["genie_q0"].astype(np.float64),
                q3=a["genie_q3"].astype(np.float64))


def ccqe(d, pdg=None):
    m = (d["genie_mode"] == MODE_QE) & (d["true_isnc"] == 0) & (d["true_pdg"] != -1)
    return m & (d["true_pdg"] == pdg) if pdg is not None else m & np.isin(
        d["true_pdg"], list(FLAV))


# ---------------------------------------------------------------- plots

def fig_scatter(d, det, rng):
    m = np.flatnonzero(ccqe(d, 14))
    show = rng.choice(m, min(len(m), 30000), replace=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    for ax, k, lab in ((axs[0], "w_tool", "match_tool: true"),
                       (axs[1], "w_def", "default")):
        ax.scatter(d[k][show], d["r"][show], s=3, alpha=0.3, lw=0, color=C[0])
        ax.plot([0, 2.5], [0, 2.5], color=INK2, lw=1, ls="--",
                label="r = w (cvwgt already has it)")
        ax.axhline(1, color=C[1], lw=1, ls=":",
                   label="r = 1 (cvwgt = ZExp only)")
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 2.5)
        ax.set_xlabel(f"ccqe_xsec_corr weight ({lab})")
        ax.set_ylabel("r = cvwgt / zexp_cv(Nieves)")
        ax.legend(fontsize=8, loc="upper left")
        cc = np.corrcoef(d[k][m], d["r"][m])[0, 1]
        ax.set_title(f"nu_mu CC QE, N={len(m)}, corr={cc:.3f}", fontsize=10)
    fig.suptitle(f"{det}: cvwgt / ZExp CV vs the fake-data CCQE weight",
                 color=INK)
    fig.tight_layout()
    return fig


def fig_ratio_hist(d, det):
    bins = np.linspace(-1.0, 1.0, 161)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (k, lab) in zip(axs, (("w_tool", "r / w(match_tool)"),
                                  ("w_def", "r / w(default)"),
                                  (None, "r / 1"))):
        for pdg, fl in FLAV.items():
            m = ccqe(d, pdg)
            den = d[k][m] if k else np.ones(m.sum())
            ok = (den > 0) & (d["r"][m] > 0)
            if not ok.any():
                continue
            x = np.log(d["r"][m][ok] / den[ok])
            f1 = np.mean(np.abs(np.exp(x) - 1) < 0.01)
            ax.hist(np.clip(x, bins[0], bins[-1]), bins, histtype="step",
                    lw=1.6, color=FLAV_COLOR[pdg],
                    label=f"{fl} (N={ok.sum()}, {100 * f1:.1f}% within 1%)")
        ax.axvline(0, color=INK2, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel(f"ln({lab})")
        ax.legend(fontsize=7)
    axs[0].set_ylabel("CC QE events")
    fig.suptitle(f"{det}: per-event ratios (clipped to +-1)", color=INK)
    fig.tight_layout()
    return fig


def coarse_cells(d, m, min_n=30):
    b3 = np.digitize(d["q3"][m], np.arange(0, 2.0, 0.1))
    b0 = np.digitize(d["q0"][m], np.arange(0, 1.5, 0.05))
    bE = np.digitize(d["E"][m], np.arange(0.2, 3.01, 0.2))
    key = (b3 * 100 + b0) * 100 + bE
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    mr = np.bincount(inv, d["r"][m]) / cnt
    mw = np.bincount(inv, d["w_tool"][m]) / cnt
    big = cnt >= min_n
    return mr[big], mw[big], cnt[big]


def fig_binned_means(d, det):
    m = ccqe(d, 14) & (d["E"] >= 0.2) & (d["E"] <= 3.0)
    mr, mw, cnt = coarse_cells(d, m)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axs[0]
    sc = ax.scatter(mw, mr, c=np.log10(cnt), s=12, cmap="viridis")
    ax.plot([0, 2], [0, 2], color=INK2, lw=1, ls="--", label="<r> = <w>")
    ax.axhline(1, color=C[1], lw=1, ls=":", label="<r> = 1")
    ax.set_xlabel("<ccqe_xsec_corr weight (match_tool)> in cell")
    ax.set_ylabel("<cvwgt / zexp_cv> in cell")
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.6)
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label="log10 events in cell")
    ax.set_title(f"nu_mu CC QE, {len(mr)} (E, q3, q0) cells with >= 30 events",
                 fontsize=9)
    ax = axs[1]
    ax.hist(mr / mw, np.linspace(0.8, 1.2, 81), histtype="step", lw=2,
            color=C[0], label="<r>/<w>")
    ax.hist(mr, np.linspace(0.8, 1.2, 81), histtype="step", lw=2,
            color=C[1], label="<r>/1")
    ax.set_xlabel("cell ratio")
    ax.set_ylabel("cells")
    ax.legend(fontsize=8)
    fig.suptitle(f"{det}: coarse-cell averages (0.2 GeV E x 0.1 q3 x 0.05 q0)",
                 color=INK)
    fig.tight_layout()
    return fig


def fig_nonqe_cvwgt(d, det):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 3, 121)
    nu = d["true_pdg"] != -1
    groups = [("CC " + common.MODE_NAMES[k], nu & (d["true_isnc"] == 0)
               & (d["genie_mode"] == k)) for k in (1, 2, 3, 10)]
    groups += [("NC (all modes)", nu & (d["true_isnc"] == 1)),
               ("cosmic/unmatched", ~nu)]
    for i, (lab, m) in enumerate(groups):
        if m.any():
            f = np.mean(d["cvwgt"][m] != 1)
            ax.hist(np.clip(d["cvwgt"][m], 0, 3), bins, histtype="step",
                    lw=1.8, color=C[i % len(C)],
                    label=f"{lab}: N={m.sum()}, {100 * f:.2f}% != 1")
    ax.set_yscale("log")
    ax.set_xlabel("cvwgt")
    ax.set_ylabel("events")
    ax.legend(fontsize=8)
    ax.set_title(f"{det}: cvwgt off CC QE (FrKin_PiProFix_N cv only)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def fig_q3q0(d, det):
    m = ccqe(d, 14) & (d["E"] >= 0.2) & (d["E"] <= 3.0)
    b3, b0 = np.linspace(0, 1.6, 33), np.linspace(0, 1.0, 41)
    n, _, _ = np.histogram2d(d["q3"][m], d["q0"][m], (b3, b0))
    out = []
    for k in ("r", "w_tool"):
        s, _, _ = np.histogram2d(d["q3"][m], d["q0"][m], (b3, b0),
                                 weights=d[k][m])
        out.append(np.where(n >= 10, s / np.maximum(n, 1), np.nan))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, h, t, cm, lim in ((axs[0], out[0], "<cvwgt / zexp_cv>", "RdBu_r", (0.5, 1.5)),
                              (axs[1], out[1], "<w (match_tool)>", "RdBu_r", (0.5, 1.5)),
                              (axs[2], out[0] / out[1], "ratio", "RdBu_r", (0.8, 1.2))):
        im = ax.pcolormesh(b3, b0, h.T, cmap=cm, vmin=lim[0], vmax=lim[1])
        fig.colorbar(im, ax=ax)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("q3 [GeV]")
        ax.set_ylabel("q0 [GeV]")
        ax.grid(False)
    fig.suptitle(f"{det}: nu_mu CC QE, mean weights vs (q3, q0), cells >= 10 events",
                 color=INK)
    fig.tight_layout()
    return fig


def fig_enu(d, det):
    bins = np.linspace(0, 3.5, 36)
    x = 0.5 * (bins[1:] + bins[:-1])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, pdg in zip(axs, (14, -14)):
        m = ccqe(d, pdg)
        n, _ = np.histogram(d["E"][m], bins)
        for i, (k, lab) in enumerate((("cvwgt", "cvwgt"), ("zexp", "zexp_cv (Nieves)"),
                                      ("r", "cvwgt / zexp_cv"),
                                      ("w_tool", "ccqe_xsec_corr match_tool"),
                                      ("w_def", "ccqe_xsec_corr default"))):
            s, _ = np.histogram(d["E"][m], bins, weights=d[k][m])
            ax.plot(x, np.where(n >= 5, s / np.maximum(n, 1), np.nan),
                    color=C[i], lw=1.6, marker="o", ms=3, label=lab)
        ax.axvspan(0.2, 3.0, color=GRID, alpha=0.3, lw=0)
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("E_nu [GeV]")
        ax.set_ylabel("mean weight")
        ax.set_title(f"{FLAV[pdg]} CC QE (N={m.sum()}); shaded = map E range",
                     fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"{det}: weights vs E_nu", color=INK)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- main

def main():
    args = common.parse_args(__doc__.splitlines()[0])
    out = common.Output("ccqe_cvwgt", "ccqe_xsec_corr vs cvwgt "
                        "(validation/ccqe_xsec_corr/validate_ccqe_cvwgt.py)")
    out.text("cvwgt provenance (code): cafpyana analysis_village/gump/loaddf.py "
             "xsec_cv_rwgt = [ZExpPCAWeighter_SBN_v3_MvA_b1, "
             "CCQEXSecCorr_SBN_v3_CCQEXSecCorr, GENIEReWeight_SBN_v3_FrKin_PiProFix_N];")
    out.text("  cvwgt = prod(cv col, or 'morph' for the 1-universe correction dial); "
             "gumple/run_gumple_pipeline.py passes reweight_aFF=--weights.")
    out.text("  nusystematics fcl/CCQEXSecCorr.ToolConfig.fcl: 'Correction dial: "
             "central_value only' (v3_04_02a -> v3_06_02 maps).")
    rng = np.random.default_rng(1)
    for det, files in common.files_by_detector(args).items():
        out.text(f"\n== {det}: {len(files)} files")
        d = common.load(files, process)
        cv = d["cvwgt"]
        W = {"cvwgt": cv, "zexp_cv": d["zexp"], "cvwgt/zexp": d["r"],
             "ccqe (match_tool)": d["w_tool"], "ccqe (default)": d["w_def"]}
        out.save(common.fig_domain(d, W, det, "cvwgt factors vs ccqe_xsec_corr"),
                 f"domain_{det}")
        out.save(common.fig_weight_hists(
            {k: W[k] for k in ("cvwgt", "cvwgt/zexp", "ccqe (match_tool)",
                               "ccqe (default)")}, det, "weights"), f"weight_hists_{det}")
        out.save(fig_scatter(d, det, rng), f"scatter_{det}")
        out.save(fig_ratio_hist(d, det), f"ratio_hist_{det}")
        out.save(fig_binned_means(d, det), f"binned_means_{det}")
        out.save(fig_nonqe_cvwgt(d, det), f"nonqe_cvwgt_{det}")
        out.save(fig_q3q0(d, det), f"q3q0_{det}")
        out.save(fig_enu(d, det), f"enu_{det}")

        common.weight_summary(out, d["w_tool"], cv, "ccqe_xsec_corr match_tool")
        common.weight_summary(out, d["w_def"], cv, "ccqe_xsec_corr default")

        for pdg, fl in FLAV.items():
            m = ccqe(d, pdg)
            if not m.any():
                continue
            inr = m & (d["E"] >= 0.2) & (d["E"] <= 3.0)
            ok = inr & (d["w_tool"] > 0)
            f_t = np.mean(np.abs(d["r"][ok] / d["w_tool"][ok] - 1) < 0.01)
            okd = inr & (d["w_def"] > 0)
            f_d = np.mean(np.abs(d["r"][okd] / d["w_def"][okd] - 1) < 0.01)
            f_1 = np.mean(np.abs(d["r"][inr] - 1) < 0.01)
            f_c1 = np.mean(cv[m] == 1)
            out.text(f"  {fl} CC QE: N={m.sum()} (in 0.2-3 GeV: {inr.sum()}), "
                     f"cvwgt==1: {100 * f_c1:.1f}%; |r/w-1|<1%: tool {100 * f_t:.1f}%, "
                     f"default {100 * f_d:.1f}%; |r-1|<1%: {100 * f_1:.1f}%")
            ok &= d["r"] > 0
            if ok.sum() > 10:
                x = np.log(d["r"][ok] / d["w_tool"][ok])
                y = np.log(d["r"][ok])
                out.text(f"    std ln(r/w_tool)={np.std(x):.3f} vs std ln(r)="
                         f"{np.std(y):.3f}; cvwgt-weighted sum r / sum w_tool = "
                         f"{np.sum(d['zexp'][ok] * d['r'][ok]) / np.sum(d['zexp'][ok] * d['w_tool'][ok]):.4f}")
        m14 = ccqe(d, 14) & (d["E"] >= 0.2) & (d["E"] <= 3.0) & (d["w_tool"] > 0)
        f_t = np.mean(np.abs(d["r"][m14] / d["w_tool"][m14] - 1) < 0.01)
        out.check(f_t > 0.95, f"nu_mu CC QE: cvwgt/zexp_cv == ccqe weight "
                  f"event-by-event (within 1%: {100 * f_t:.1f}%; needs > 95%)",
                  warn=True)
        mr, mw, cnt = coarse_cells(d, ccqe(d, 14) & (d["E"] >= 0.2) & (d["E"] <= 3.0))
        rr = mr / mw
        q = np.percentile(rr, [16, 50, 84])
        q1 = np.percentile(mr, [16, 50, 84])
        cw = np.sum(cnt * (rr - 1) ** 2) / np.sum(cnt)
        c1 = np.sum(cnt * (mr - 1) ** 2) / np.sum(cnt)
        out.text(f"  coarse cells (N={len(rr)}): <r>/<w> 16/50/84% = "
                 f"{q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}, rms dev {np.sqrt(cw):.3f}; "
                 f"<r>/1: {q1[0]:.3f}/{q1[1]:.3f}/{q1[2]:.3f}, rms dev {np.sqrt(c1):.3f}")
        already = np.sqrt(cw) < 0.5 * np.sqrt(c1)
        out.check(not already,
                  "fake-data dial not already in cvwgt: " + (
                      "VIOLATED -- coarse-cell averages of cvwgt/zexp_cv track the "
                      "ccqe_xsec_corr weight, so applying the dial on top of "
                      "cvwgt DOUBLE COUNTS the CCQE correction" if already else
                      "cvwgt/zexp_cv does not track the ccqe_xsec_corr weight"))
        hyp = ccqe(d) & (d["w_tool"] == 0)
        if hyp.any():
            out.text(f"  match_tool weight-0 CC QE events (hyperons etc.): N={hyp.sum()}, "
                     f"cvwgt == 0: {np.sum(cv[hyp] == 0)}, cvwgt 5/50/95% = "
                     f"{np.round(np.percentile(cv[hyp], [5, 50, 95]), 3)}")
        offqe = (d["true_pdg"] != -1) & ~((d["genie_mode"] == MODE_QE) & (d["true_isnc"] == 0))
        f_off = np.mean(cv[offqe] != 1)
        out.check(f_off < 1e-3, f"cvwgt == 1 off CC QE (FrKin_PiProFix_N cv): "
                  f"{100 * f_off:.3f}% != 1", warn=True)
        neg = cv < 0
        out.check(not neg.any(), f"cvwgt >= 0 (negative: {neg.sum()}, values "
                  f"{np.round(cv[neg], 3)}, pdg {d['true_pdg'][neg]}, mode "
                  f"{d['genie_mode'][neg]})", warn=True)
        out.check(bool(np.all(cv[d["true_pdg"] == -1] == 1)),
                  "cvwgt == 1 for cosmic / unmatched slices")
    sys.exit(out.close())


if __name__ == "__main__":
    main()
