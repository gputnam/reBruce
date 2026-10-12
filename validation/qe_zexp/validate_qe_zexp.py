#!/usr/bin/env python3
"""Validation plots for the qe_zexp_mva_to_lqcd calculator (CCQE axial form
factor MINERvA -> LQCD, as a ratio of Nieves cross sections).

w = xsec_Nieves(FA = LQCD) / xsec_Nieves(FA = MINERvA) at the event's
pre-FSI kinematics (fakedata/calculators/qe_zexp.py, nieves.py, zexp.py),
for numu / numubar CC QE events with the pre-FSI recoil nucleon stored
(genie_prefsi_p for nu, genie_prefsi_n for nubar). On the CV MC sBruce
files this script runs the real calculator and checks / plots, per detector
(cvwgt-weighted):

  domain            w != 1 by flavour x CC/NC x mode (NC elastic is mode 0
                    with true_isnc 1 and must stay at 1; nue excluded)
  weight_hists      weights in the default ("tune") and "nusyst" gA conventions
  weight_vs_q2      w vs Q2 = q3^2 - q0^2 (genie_q0/q3), nu and nubar, over the
                    FA(LQCD)^2 / FA(MINERvA)^2 curves of both conventions (a
                    reference only: the weight is NOT an FA^2 ratio)
  spectra           Q2 and cos theta_mu (pre-FSI lepton, nu frame) nominal vs
                    weighted, nu and nubar
  recoil            pre-FSI nucleon availability for numu (p) and numubar (n)
                    CC QE: which events can be reweighted at all
  knob_orientation  the stored multisigma_ZExpPCAWeighter_SBN_v3_MvA_b1..b4
                    weights at labels +1 and -1 against an independent Nieves
                    computation at b_i = +1 with the PCA built from EIGEN's
                    eigenvectors (nusystematics ZExpPCAWeighter_tool.cc uses
                    Eigen::SelfAdjointEigenSolver). Eigen 3.4.0 (compiled
                    separately for this check) returns exactly MINUS numpy's
                    eigh eigenvectors for this matrix, column by column; they
                    are hard-coded below. If label +1 matches, the v3 labels
                    are NOT reversed and validation/validate_nieves.py's
                    "numpy vs Eigen sign" explanation is the right one.
  cvwgt_zexp        is the deuterium->MINERvA ZExp CV weight inside cvwgt?
                    In Q2 bins, <cvwgt / CCQEXSecCorr(tool)> vs the computed
                    deuterium->MINERvA weight (deut_to_minerva_weight), numu CC
                    QE: if cvwgt carries it, the two track each other (the
                    calculator's "from MINERvA" starting point is right).

The gA convention: the input MC is AR23 (FA(0) = -1.2670) reweighted by the
nusystematics ZExpPCAWeighter CV, which keeps FA(0) = -1.2670 with the
MINERvA a1..a4. The default "tune" convention divides by MINERvA with
FA(0) = -1.2723, so the effective result is LQCD(-1.2754) x
[MvA(-1.2670) / MvA(-1.2723)]; this script quantifies that against a
"consistent" weight xsec(LQCD, -1.2754) / xsec(MvA, -1.2670).

Writes validation/plots/qe_zexp/*.png, qe_zexp.pdf and summary.txt.

Usage: ./venv/bin/python validation/qe_zexp/validate_qe_zexp.py
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
from common import C, INK, INK2                              # noqa: E402

from fakedata.calculators.ccqe_corr import CCQEXSecCorr      # noqa: E402
from fakedata.calculators.qe_zexp import (                   # noqa: E402
    GA_CV, QE_BRANCHES, QEZexpMvaToLQCD, _ff, deut_to_minerva_weight,
    qe_kinematics, qe_numu_prefsi_mask)
from fakedata.nieves import NievesQEXSec                     # noqa: E402
from fakedata.sbruce import MODE_QE, valid                   # noqa: E402
from fakedata.zexp import COEFF_SETS, ZExpAxialFF            # noqa: E402

# nusystematics ZExpPCAWeighter_tool.cc, namespace Nature_614_102522
A_CV = np.array([1.50, -1.2, -0.1, 0.2])
COV = np.array([[0.0961, 0.002604, -0.54777, 0.403],
                [0.002604, 0.49, -0.4256, -1.365],
                [-0.54777, -0.4256, 3.61, -1.2825],
                [0.403, -1.365, -1.2825, 6.25]])
# Eigen 3.4.0 SelfAdjointEigenSolver(COV).eigenvectors() (columns; eigenvalues
# ascending 0.0039440275 0.040381133 3.3915923 7.0101826)
EIGEN_EVECS = np.array([
    [-0.96651734, -0.21462162, -0.11531504, 0.080524788],
    [-0.18791241, 0.92823685, -0.27185832, -0.17075808],
    [-0.17413798, 0.1685598, 0.90895803, -0.33920327],
    [-0.014461272, 0.25278746, 0.29426314, 0.92157397]])
KNOB = "multisigma_ZExpPCAWeighter_SBN_v3_MvA_b{}"
N_KNOB_NU = 1500        # numu events per file for the knob comparison

EXTRA = ["cvwgt", "genie_q0", "genie_q3"]


def eigen_pca():
    evals, evecs = np.linalg.eigh(COV)
    return EIGEN_EVECS * np.sqrt(evals), evals, evecs


def process(sb, xs, pca):
    a = sb.arrays(QE_BRANCHES + EXTRA)
    n = sb.n_entries
    w = QEZexpMvaToLQCD().compute(sb)["fdwgt_qe_zexp_mva_to_lqcd"]
    w_ns = QEZexpMvaToLQCD(ga_convention="nusyst").compute(sb)["fdwgt_qe_zexp_mva_to_lqcd"]
    d2m = deut_to_minerva_weight(sb, xs)
    ccqe = CCQEXSecCorr(match_tool=True).compute(sb)["fdwgt_ccqe_xsec_corr"]

    # "consistent" weight: LQCD (tune gA) over MINERvA with the AR23 CV gA
    fa_lqcd = _ff("lqcd", "tune")
    fa_mva_cv = ZExpAxialFF(**dict(COEFF_SETS["minerva_nature"], ga=GA_CV))
    w_cons = np.ones(n)
    masks = {}
    for pdg in (14, -14):
        m = qe_numu_prefsi_mask(a, pdg)
        masks[pdg] = m
        if m.any():
            w_cons[m] = xs.weight_ratio(*qe_kinematics(a, m, pdg), fa_lqcd,
                                        fa_mva_cv, is_neutrino=pdg > 0)

    # stored ZExp knobs vs Eigen-convention PCA at b_i = +1
    sub = np.zeros(n, dtype=bool)
    sub[np.flatnonzero(masks[14])[:N_KNOB_NU]] = True
    sub |= masks[-14]
    rp = np.full((n, 4), np.nan)
    rm = np.full((n, 4), np.nan)
    fa_cv = ZExpAxialFF(a=list(A_CV), t0=-0.75, tcut=0.1764, ga=GA_CV)
    for b in range(4):
        fa_b = ZExpAxialFF(a=list(A_CV + pca[:, b]), t0=-0.75, tcut=0.1764, ga=GA_CV)
        sp = common.knob(sb, KNOB.format(b + 1), 1.0)
        sm = common.knob(sb, KNOB.format(b + 1), -1.0)
        for pdg in (14, -14):
            m = sub & masks[pdg]
            if m.any():
                wc = xs.weight_ratio(*qe_kinematics(a, m, pdg), fa_b, fa_cv,
                                     is_neutrino=pdg > 0)
                rp[m, b] = wc / sp[m]
                rm[m, b] = wc / sm[m]

    lep = np.stack([a[f"genie_prefsi_lep_p{c}"] for c in "xyz"], 1).astype(np.float64)
    q0 = a["genie_q0"].astype(np.float64)
    q3 = a["genie_q3"].astype(np.float64)
    out = dict(genie_mode=a["genie_mode"], true_isnc=a["true_isnc"],
               true_pdg=a["true_pdg"], cvwgt=a["cvwgt"].astype(np.float64),
               w=w, w_ns=w_ns, w_cons=w_cons, d2m=d2m, ccqe=ccqe,
               m_nu=masks[14], m_nub=masks[-14],
               hasl=valid(a["genie_prefsi_lep_py"], a["genie_prefsi_lep_pz"]),
               hasp=valid(a["genie_prefsi_p_px"]), hasn=valid(a["genie_prefsi_n_px"]),
               Q2=np.where(valid(q0, q3), q3 ** 2 - q0 ** 2, np.nan),
               cos_mu=lep[:, 2] / np.linalg.norm(lep, axis=1))
    for b in range(4):
        out[f"rp{b}"] = rp[:, b]
        out[f"rm{b}"] = rm[:, b]
    return out


def cc_qe(d, pdg):
    return ((d["genie_mode"] == MODE_QE) & (d["true_isnc"] == 0)
            & (d["true_pdg"] == pdg))


# ---------------------------------------------------------------- plots

def fa2_ratio(q2v, conv):
    fa_new, fa_old = _ff("lqcd", conv), _ff("minerva_nature", conv)
    return (fa_new(-q2v) / fa_old(-q2v)) ** 2


def fig_weight_vs_q2(d, det, rng):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    qq = np.linspace(0, 2.0, 201)
    for ax, (lab, m) in zip(axs, (("numu CC QE", d["m_nu"]), ("numubar CC QE", d["m_nub"]))):
        idx = np.flatnonzero(m)
        if len(idx):
            show = rng.choice(idx, min(len(idx), 15000), replace=False)
            ax.scatter(d["Q2"][show], d["w"][show], s=3, lw=0, alpha=0.4,
                       color=C[0], label=f"events, tune (N={len(idx)})")
            bins = np.linspace(0, 2.0, 21)
            ib = np.digitize(d["Q2"][idx], bins) - 1
            prof = [np.average(d["w"][idx][ib == i], weights=d["cvwgt"][idx][ib == i])
                    if np.sum(ib == i) >= 3 else np.nan for i in range(len(bins) - 1)]
            ax.plot(0.5 * (bins[1:] + bins[:-1]), prof, "o-", color=C[4], ms=4,
                    lw=1.5, label="cvwgt mean (tune)")
        ax.plot(qq, fa2_ratio(qq, "tune"), color=INK, lw=1.5, label="FA^2 ratio, tune gA")
        ax.plot(qq, fa2_ratio(qq, "nusyst"), color=INK2, lw=1.2, ls="--",
                label="FA^2 ratio, nusyst gA")
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_xlim(0, 2.0)
        ax.set_ylim(0.6, 1.8)
        ax.set_xlabel("Q2 = q3^2 - q0^2 [GeV^2]")
        ax.set_ylabel("weight")
        ax.set_title(lab, fontsize=10, color=INK)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{det}: qe_zexp weight vs Q2 (LQCD / MINERvA)", color=INK)
    fig.tight_layout()
    return fig


def fig_spectra(d, det):
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.5), sharex="col",
                            gridspec_kw=dict(height_ratios=[2.5, 1]))
    specs = [("Q2", "Q2 [GeV^2]", np.linspace(0, 1.5, 31)),
             ("cos_mu", "cos theta_mu (nu frame)", np.linspace(-1, 1, 21))]
    col = 0
    for lab, pdg in (("numu", 14), ("numubar", -14)):
        sel = cc_qe(d, pdg) & np.isfinite(d["Q2"])
        for key, xl, bins in specs:
            ax, axr = axs[0, col], axs[1, col]
            common.ratio_panel(ax, axr, d[key][sel], bins, d["cvwgt"][sel], d["w"][sel],
                               [(f"all {lab} CC QE", np.ones(sel.sum(), bool), C[0]),
                                ("nusyst gA", np.ones(sel.sum(), bool), C[1])][:1],
                               f"{xl} ({lab} CC QE)")
            g0, _ = np.histogram(d[key][sel], bins, weights=d["cvwgt"][sel])
            g1, _ = np.histogram(d[key][sel], bins, weights=(d["cvwgt"] * d["w_ns"])[sel])
            axr.stairs(np.divide(g1, g0, out=np.full(len(g0), np.nan), where=g0 > 0),
                       bins, color=C[1], lw=1.2, ls="--", baseline=None, label="nusyst gA")
            axr.set_ylim(0.85, 1.25)
            col += 1
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=7)
    fig.suptitle(f"{det}: CC QE spectra, nominal (MINERvA FA) vs weighted (LQCD)", color=INK)
    fig.tight_layout()
    return fig


def fig_recoil(d, det):
    rows = [("numu CC QE", cc_qe(d, 14)), ("numubar CC QE", cc_qe(d, -14))]
    fig, ax = plt.subplots(figsize=(9, 2.8))
    for i, (lab, s) in enumerate(rows):
        cats = [("p only", s & d["hasp"] & ~d["hasn"]),
                ("n only", s & d["hasn"] & ~d["hasp"]),
                ("p and n", s & d["hasp"] & d["hasn"]),
                ("neither", s & ~d["hasp"] & ~d["hasn"])]
        left = 0.0
        for j, (cl, m) in enumerate(cats):
            f = m.sum() / max(s.sum(), 1)
            ax.barh(i, f, left=left, color=C[j], label=cl if i == 0 else None)
            if f > 0.05:
                ax.text(left + f / 2, i, f"{f:.3f}", ha="center", va="center",
                        fontsize=8, color="white")
            left += f
    ax.set_yticks(range(2), [f"{lab} (N={s.sum()})" for lab, s in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of events: pre-FSI nucleon stored")
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title(f"{det}: recoil availability (nu needs p, nubar needs n)",
                 fontsize=10, color=INK, pad=24)
    fig.tight_layout()
    return fig


def fig_knob(d, det):
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.5))
    bins = np.linspace(0.8, 1.2, 81)
    for row, (lab, m) in enumerate((("numu", d["m_nu"]), ("numubar", d["m_nub"]))):
        for b in range(4):
            ax = axs[row, b]
            for key, c, sl in ((f"rp{b}", C[0], "label +1"), (f"rm{b}", C[1], "label -1")):
                r = d[key][m]
                r = r[np.isfinite(r)]
                if len(r):
                    ax.hist(np.clip(r, bins[0], bins[-1]), bins, histtype="step", lw=1.8,
                            color=c, label=f"{sl}: median {np.median(r):.4f}")
            ax.axvline(1, color=INK2, lw=0.8, ls=":")
            ax.set_xlabel(f"computed(Eigen b{b + 1}=+1) / stored")
            ax.set_title(f"{lab} b{b + 1}", fontsize=10, color=INK)
            ax.legend(fontsize=7)
    fig.suptitle(f"{det}: ZExpPCAWeighter MvA dial orientation (Eigen eigenvector signs)",
                 color=INK)
    fig.tight_layout()
    return fig


def fig_cvwgt(d, det):
    m = d["m_nu"] & (d["ccqe"] > 0) & np.isfinite(d["Q2"])
    r = d["cvwgt"][m] / d["ccqe"][m]
    bins = np.linspace(0, 1.6, 17)
    ib = np.digitize(d["Q2"][m], bins) - 1
    xc = 0.5 * (bins[1:] + bins[:-1])
    mr = np.array([np.median(r[ib == i]) if np.sum(ib == i) > 10 else np.nan
                   for i in range(len(xc))])
    md = np.array([np.median(d["d2m"][m][ib == i]) if np.sum(ib == i) > 10 else np.nan
                   for i in range(len(xc))])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axs[0]
    ax.plot(xc, mr, "o-", color=C[0], label="median cvwgt / CCQEXSecCorr(tool)")
    ax.plot(xc, md, "s--", color=C[1], label="median computed deut->MINERvA")
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Q2 [GeV^2]")
    ax.set_ylabel("weight")
    ax.legend(fontsize=8)
    ax.set_title("numu CC QE, in Q2 bins", fontsize=10, color=INK)
    ax = axs[1]
    lr = np.log(r)
    ax.hist(lr - np.log(d["d2m"][m]), np.linspace(-1, 1, 101), histtype="step", lw=2,
            color=C[0], label="ln(cvwgt / ccqe / d2m)")
    ax.hist(lr, np.linspace(-1, 1, 101), histtype="step", lw=1.5, color=C[1],
            label="ln(cvwgt / ccqe)")
    ax.set_xlabel("log ratio")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("event by event", fontsize=10, color=INK)
    fig.suptitle(f"{det}: does cvwgt contain the deuterium->MINERvA ZExp weight?", color=INK)
    fig.tight_layout()
    return fig, xc, mr, md


# ---------------------------------------------------------------- summary

def summarize(out, d, det, nfiles, slope):
    cv, w = d["cvwgt"], d["w"]
    out.text(f"== {det}: {nfiles} files, {len(w)} slices")
    common.weight_summary(out, w, cv, "qe_zexp (tune)")
    common.weight_summary(out, d["w_ns"], cv, "qe_zexp (nusyst)")
    qe = d["genie_mode"] == MODE_QE
    ncqe = qe & (d["true_isnc"] == 1) & (d["true_pdg"] != -1)
    out.check(bool(np.all(w[ncqe] == 1)), f"NC QE / NC elastic (mode 0, isnc 1): w == 1 (N={ncqe.sum()})")
    nue = np.abs(d["true_pdg"]) == 12
    out.check(bool(np.all(w[nue] == 1)), f"nue / nuebar excluded: w == 1 (N={nue.sum()})")
    out.check(bool(np.all(w[~qe] == 1)), "non-QE modes: w == 1")
    dom = d["m_nu"] | d["m_nub"]
    out.check(bool(np.all(w[~dom] == 1) and np.mean(w[dom] != 1) > 0.999),
              f"w != 1 exactly on the independent domain mask (N={dom.sum()})")
    for lab, pdg, rec, m in (("numu", 14, "hasp", d["m_nu"]), ("numubar", -14, "hasn", d["m_nub"])):
        s = cc_qe(d, pdg)
        out.text(f"  {lab} CC QE: N={s.sum()}, reweighted {m.sum()} ({m.sum() / max(s.sum(), 1):.4f}); "
                 f"missing pre-FSI {rec[3:]}: {np.sum(s & ~d[rec])}; "
                 f"have p {np.sum(s & d['hasp'])}, have n {np.sum(s & d['hasn'])}")
        if m.any():
            out.text(f"    mean w (cvwgt) tune {np.average(w[m], weights=cv[m]):.4f}, "
                     f"nusyst {np.average(d['w_ns'][m], weights=cv[m]):.4f}, "
                     f"consistent (LQCD tune gA / MvA AR23 gA) {np.average(d['w_cons'][m], weights=cv[m]):.4f}")
    out.check(np.mean(d["m_nub"][cc_qe(d, -14)]) > 0.5,
              "numubar CC QE included (majority reweighted)", warn=True)
    m = d["m_nu"]
    dev = np.average(w[m] / d["w_cons"][m], weights=cv[m]) - 1
    out.check(abs(dev) < 1e-3,
              f"gA convention: tune-convention weight / consistent weight (MC sits at MvA "
              f"a1..a4 with FA(0) = {GA_CV}) = {1 + dev:.4f} (cvwgt mean, numu)", warn=True)
    for b in range(4):
        for lab, mm in (("numu", d["m_nu"]), ("numubar", d["m_nub"])):
            rp = d[f"rp{b}"][mm]
            rm = d[f"rm{b}"][mm]
            ok = np.isfinite(rp) & np.isfinite(rm)
            if not ok.any():
                continue
            mp, mmed = np.median(rp[ok]), np.median(rm[ok])
            sp, sm = np.std(rp[ok]), np.std(rm[ok])
            tol = 0.003 if lab == "numu" else 0.03
            out.check(abs(mp - 1) < tol and abs(mp - 1) < abs(mmed - 1),
                      f"ZExp b{b + 1} {lab}: Eigen-PCA b=+1 matches stored LABEL +1 "
                      f"(median ratio {mp:.4f}, rms {sp:.4f}) not -1 ({mmed:.4f}, rms {sm:.4f}); N={ok.sum()}",
                      warn=(lab == "numubar"))
    out.check(0.8 < slope < 1.2,
              f"cvwgt contains deut->MINERvA: slope of ln median(cvwgt/ccqe) vs "
              f"ln median(d2m) over Q2 bins = {slope:.3f} (1 expected)", warn=True)
    out.text("")


def main():
    args = common.parse_args(__doc__.splitlines()[0])
    by_det = common.files_by_detector(args)
    out = common.Output("qe_zexp", "qe_zexp_mva_to_lqcd validation "
                                   "(validation/qe_zexp/validate_qe_zexp.py)")
    pca, evals, np_evecs = eigen_pca()
    out.text(f"  PCA eigenvalues {np.round(evals, 6)}; Eigen 3.4.0 eigenvectors == -numpy: "
             f"{bool(np.allclose(EIGEN_EVECS, -np_evecs, atol=1e-7))}")
    xs = NievesQEXSec()
    rng = np.random.default_rng(1)
    for det, files in by_det.items():
        d = common.load(files, process, xs, pca)
        out.save(common.fig_domain(d, {"qe_zexp (tune)": d["w"]}, det, "qe_zexp_mva_to_lqcd"),
                 f"domain_{det}")
        out.save(common.fig_weight_hists({"tune": d["w"], "nusyst": d["w_ns"]}, det,
                                         "qe_zexp_mva_to_lqcd"), f"weight_hists_{det}")
        out.save(fig_weight_vs_q2(d, det, rng), f"weight_vs_q2_{det}")
        out.save(fig_spectra(d, det), f"spectra_{det}")
        out.save(fig_recoil(d, det), f"recoil_{det}")
        out.save(fig_knob(d, det), f"knob_orientation_{det}")
        fig, xc, mr, md = fig_cvwgt(d, det)
        out.save(fig, f"cvwgt_zexp_{det}")
        ok = np.isfinite(mr) & np.isfinite(md)
        slope = np.polyfit(np.log(md[ok]), np.log(mr[ok]), 1)[0] if ok.sum() > 2 else np.nan
        summarize(out, d, det, len(files), slope)
    return out.close()


if __name__ == "__main__":
    sys.exit(main())
