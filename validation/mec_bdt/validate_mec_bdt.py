#!/usr/bin/env python3
"""Validation plots for the mec_bdt calculator (AR23/SuSAv2 CC MEC ->
exclusive-Valencia 2p2h, Zihao Lin's hep_ml GBReweighter).

The weight is exp(BDT score) of 8 pre-FSI features [p_px, p_py, p_pz,
p2_px, p2_py, p2_pz, mu_py, mu_pz] in the training "reaction frame"
(neutrino +z, muon pT along -y), normalized per file to a cvwgt-weighted
mean of 1 over the reweighted events (fakedata/calculators/mec_bdt.py,
fakedata/bdt.py, data/mec_bdt_susav2_to_valencia.json). The calculator
builds the features from the sBruce genie_prefsi_* momenta, which are
stored in a nu/lepton frame (z = nu, y = lepton pT, x = y cross z,
cafpyana analysis_village/gump/loaddf.py), by flipping the y components.
On the CV MC sBruce files this script runs the real calculator and checks /
plots, per detector (cvwgt-weighted):

  domain            w != 1 by flavour x CC/NC x mode, for mec_bdt and for
                    the stored SuSAToVal q0-binned knob product at label +1
  weight_hists      weight distributions (mec_bdt per-file; knob +1)
  frame             sBruce lepton px / py in the stored frame (px == 0,
                    py >= 0), and transverse-momentum balance of the BDT
                    features (sum of y components ~ 0 after the flip) next to
                    the training-source CSV
  features          the 8 BDT inputs as fed to the BDT, per detector, over
                    the training source (AR23 SuSAv2 on C12, ICARUS-like BNB)
                    and target (Valencia) CSVs from PROfit/MEC-BDT-WGT/models
                    (area-normalized); y features also shown WITHOUT the flip
                    (dotted) to show which orientation matches the training
  ordering          |p_sub| / |p_lead| in sBruce and in the two training CSVs
                    (the source is sorted by KE; is the target?), and the BDT
                    raw weight vs that ratio
  xflip             ln(w(px -> -px) / w): the training frame is left-handed
                    (x_hat = y_hat rotated +90 deg about z) while sBruce is
                    right-handed; the x axes coincide (x_t = x_s = u x z for
                    lepton pT direction u), so no x flip is needed -- this
                    plot shows how much it would matter anyway
  weight_vs_kin     cvwgt-weighted mean weight (16-84% band) vs q0, q3, Enu,
                    W, p_mu, cos theta_mu, |p_lead|, |p_sub|/|p_lead|
  knob_compare      shape comparison on the BDT-domain events of nominal,
                    BDT-weighted and SuSAToVal knob-weighted (label +1 =
                    full SuSA->Valencia shift, area-normalized to nominal)
                    distributions of q0, q3, Enu, p_mu, cos theta_mu
  knob_q3q0         mean BDT weight and mean (normalized) knob weight in
                    (q3, q0)
  pairs             pre-FSI nucleon-pair content of CC numu / CC numubar /
                    NC MEC (pp / np / nn), i.e. which part of MEC is reweighted

The SuSAToVal knobs (nusystematics MECq0q3InterpWeighting, release
v1_05_10_sbn3, fcl SuSAv2ToValencia_MEC_q0binned): one dial per q0 bin
[0,.2,.4,.6,10] GeV, response 1 + d (w_map - 1) clamped to [0.1, 1000],
where w_map is the SuSAv2 -> Valencia (q3, q0) map blended in Enu (0.4 -
2.5 GeV; q0, q3 < 1.2 GeV) -- so d = +1 is the FULL shift. The tool selects
on the 2-nucleon cluster pdg (NN / NP) only, NOT on CC / flavour. Because
the response is clamped below at 0.1 but w_map is never below 0.1, an event
with stored w(+1) > 1.9 and w(-1) == 0.1 proves label +1 is d = +1.

Checks: frame (lep px == 0, py >= 0); feature port == the BDTReweight
transform_momentum_to_reaction_frame applied to the sBruce vectors; JSON
inference reproduces the training-source CSV weights; mask closure (w == 1
outside numu CC MEC pp, numubar / NC MEC untouched); per-file normalization
closure (before the driver clip); knob orientation; training-target
ordering; x-flip sensitivity; fraction of weights at the driver clip.

Writes validation/plots/mec_bdt/*.png, mec_bdt.pdf and summary.txt.

Usage: ./venv/bin/python validation/mec_bdt/validate_mec_bdt.py
           [--max-files N] [files...]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
from matplotlib.colors import TwoSlopeNorm                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

import common                                                # noqa: E402
from common import C, INK, INK2, GRID, NOM_COLOR, RW_COLOR   # noqa: E402

from fakedata.bdt import GBReweighterJSON                    # noqa: E402
from fakedata.calculator import WEIGHT_CLIP                  # noqa: E402
from fakedata.calculators.mec_bdt import (                   # noqa: E402
    BRANCHES, MODEL_PATH, MECBDTCalculator)
from fakedata.sbruce import MODE_MEC, valid                  # noqa: E402

TRAIN_DIR = "/Users/gputnam/Work/osc/PROfit/MEC-BDT-WGT/models"
TRAIN_SRC = os.path.join(TRAIN_DIR, "AR23SuSAv2_2p2h_pp_states.csv")
TRAIN_TGT = os.path.join(TRAIN_DIR, "newValencia_2p2h_pp_states.csv")
TRAIN_ROWS = 200000

FEATS = ["leading_proton_px", "leading_proton_py", "leading_proton_pz",
         "subleading_proton_px", "subleading_proton_py", "subleading_proton_pz",
         "leading_muon_py", "leading_muon_pz"]
FEAT_LABEL = ["p1 px", "p1 py", "p1 pz", "p2 px", "p2 py", "p2 pz",
              "mu py", "mu pz"]
KNOB = "multisigma_MECq0q3InterpWeighting_SBN_v3_SuSAToVal_MECResponse_q0bin{}"
Q0_EDGES = [0.0, 0.2, 0.4, 0.6, 10.0]

EXTRA = ["genie_q0", "genie_q3", "genie_Enu", "genie_W",
         "genie_prefsi_n_px", "genie_prefsi_n_py", "genie_prefsi_n_pz",
         "genie_prefsi_n2_px", "genie_prefsi_n2_py", "genie_prefsi_n2_pz"]


def reaction_frame(mu, p1, p2):
    """Port of BDTReweight/analysis.py:transform_momentum_to_reaction_frame
    (y_hat = -mu_T/|mu_T|, x_hat = (-y_hat_y, y_hat_x)), applied to any frame
    with the neutrino along +z. Returns the 8 BDT features."""
    yh = -mu[:, :2] / np.linalg.norm(mu[:, :2], axis=1, keepdims=True)
    xh = np.stack([-yh[:, 1], yh[:, 0]], axis=1)
    out = []
    for p in (p1, p2):
        out += [np.sum(p[:, :2] * xh, 1), np.sum(p[:, :2] * yh, 1), p[:, 2]]
    out += [-np.linalg.norm(mu[:, :2], axis=1), mu[:, 2]]
    return np.stack(out, axis=1)


def vec(a, s):
    return np.stack([a[f"genie_prefsi_{s}_p{c}"] for c in "xyz"],
                    axis=1).astype(np.float64)


def process(sb, model):
    a = sb.arrays(BRANCHES + EXTRA)
    n = sb.n_entries
    w = MECBDTCalculator().compute(sb)["fdwgt_mec_bdt"]

    lep, p1, p2 = vec(a, "lep"), vec(a, "p"), vec(a, "p2")
    hasl = valid(a["genie_prefsi_lep_px"], a["genie_prefsi_lep_py"],
                 a["genie_prefsi_lep_pz"])
    hasp = valid(*[a[f"genie_prefsi_p_p{c}"] for c in "xyz"])
    hasp2 = valid(*[a[f"genie_prefsi_p2_p{c}"] for c in "xyz"])
    hasn = valid(*[a[f"genie_prefsi_n_p{c}"] for c in "xyz"])
    hasn2 = valid(*[a[f"genie_prefsi_n2_p{c}"] for c in "xyz"])
    mec = a["genie_mode"] == MODE_MEC
    mask = mec & (a["true_isnc"] == 0) & (a["true_pdg"] == 14) & hasl & hasp & hasp2

    # features as the calculator builds them, and via the training transform
    f_calc = np.full((n, 8), np.nan)
    f_calc[mask] = np.column_stack([
        p1[mask, 0], -p1[mask, 1], p1[mask, 2], p2[mask, 0], -p2[mask, 1],
        p2[mask, 2], -lep[mask, 1], lep[mask, 2]])
    f_ref = reaction_frame(lep[mask], p1[mask], p2[mask])
    dfeat = np.max(np.abs(f_ref - f_calc[mask])) if mask.any() else 0.0

    raw = np.ones(n)
    raw_xf = np.ones(n)
    if mask.any():
        raw[mask] = model.predict_weights(f_calc[mask])
        fx = f_calc[mask].copy()
        fx[:, [0, 3]] *= -1
        raw_xf[mask] = model.predict_weights(fx)
    cv = a["cvwgt"].astype(np.float64)
    closure = (np.sum((cv * w)[mask]) / np.sum(cv[mask])) if mask.any() else 1.0
    scale = (np.sum(cv[mask]) / np.sum((cv * raw)[mask])) if mask.any() else 1.0

    # stored SuSAToVal knobs: product over the four q0-bin dials
    k = {}
    for s in (1.0, -1.0, 2.0):
        prod = np.ones(n)
        for b in range(4):
            prod *= np.nan_to_num(common.knob(sb, KNOB.format(b), s), nan=1.0)
        k[s] = prod

    return dict(genie_mode=a["genie_mode"], true_isnc=a["true_isnc"],
                true_pdg=a["true_pdg"], cvwgt=cv, w=w, raw=raw, raw_xf=raw_xf,
                mask=mask, mec=mec, hasl=hasl, hasp=hasp, hasp2=hasp2,
                hasn=hasn, hasn2=hasn2,
                lep_px=a["genie_prefsi_lep_px"], lep_py=a["genie_prefsi_lep_py"],
                lep_p=np.linalg.norm(lep, axis=1),
                cos_mu=lep[:, 2] / np.linalg.norm(lep, axis=1),
                p1=np.linalg.norm(p1, axis=1), p2=np.linalg.norm(p2, axis=1),
                q0=a["genie_q0"], q3=a["genie_q3"], enu=a["genie_Enu"],
                W=a["genie_W"],
                **{f"f{i}": f_calc[:, i] for i in range(8)},
                **{f"fnf{i}": (f_calc[:, i] * (-1 if i in (1, 4, 6) else 1))
                   for i in range(8)},
                k1=k[1.0], km1=k[-1.0], k2=k[2.0],
                dfeat=np.full(n, dfeat), closure=np.full(n, closure),
                scale=np.full(n, scale))


def load_csv(path, rows=TRAIN_ROWS):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        head = f.readline().strip().split(",")
    arr = np.loadtxt(path, delimiter=",", skiprows=1, max_rows=rows)
    return {h: arr[:, i] for i, h in enumerate(head)}


def csv_feats(c):
    return np.stack([c[k] for k in FEATS], axis=1)


def csv_pmag(c, which):
    return np.sqrt(sum(c[f"{which}_p{x}"] ** 2 for x in "xyz"))


# ---------------------------------------------------------------- plots

def fig_frame(d, det, src):
    h = d["hasl"]
    m = d["mask"]
    cv = d["cvwgt"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axs[0]
    ax.hist(d["lep_px"][h], np.linspace(-1e-6, 1e-6, 81), color=C[0])
    ax.set_xlabel("genie_prefsi_lep_px [GeV] (stored)")
    ax.set_title(f"lep px: max |px| = {np.max(np.abs(d['lep_px'][h])):.1e}",
                 fontsize=10, color=INK)
    ax = axs[1]
    ax.hist(d["lep_py"][h], np.linspace(-0.2, 1.5, 86), color=C[0])
    ax.set_xlabel("genie_prefsi_lep_py [GeV] (stored)")
    ax.set_title(f"lep py: fraction < 0 = {np.mean(d['lep_py'][h] < 0):.1e}",
                 fontsize=10, color=INK)
    ax = axs[2]
    bins = np.linspace(-1.2, 1.2, 61)
    sy = d["f1"][m] + d["f4"][m] + d["f6"][m]
    sx = d["f0"][m] + d["f3"][m]
    ax.hist(sy, bins, weights=cv[m], density=True, histtype="step", lw=2,
            color=C[0], label=f"sBruce sum py (flipped) mean {np.average(sy, weights=cv[m]):+.3f}")
    ax.hist(sx, bins, weights=cv[m], density=True, histtype="step", lw=2,
            color=C[1], label=f"sBruce sum px, mean {np.average(sx, weights=cv[m]):+.3f}")
    if src is not None:
        ty = src["leading_proton_py"] + src["subleading_proton_py"] + src["leading_muon_py"]
        ax.hist(ty, bins, density=True, histtype="step", lw=1.5, ls="--",
                color=INK2, label=f"training source sum py, mean {ty.mean():+.3f}")
    ax.set_xlabel("p1 + p2 (+ mu) transverse component, BDT frame [GeV]")
    ax.set_title("pre-FSI transverse balance (2p2h pair Fermi motion)",
                 fontsize=10, color=INK)
    ax.legend(fontsize=7)
    fig.suptitle(f"{det}: sBruce pre-FSI frame", color=INK)
    fig.tight_layout()
    return fig


def fig_features(d, det, src, tgt):
    m = d["mask"]
    cv = d["cvwgt"][m]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    rng = [(-0.8, 0.8), (-0.6, 1.6), (-0.8, 1.6), (-0.6, 0.6), (-0.6, 0.8),
           (-0.6, 0.8), (-1.6, 0.05), (-0.2, 2.2)]
    for i, ax in enumerate(axs.flat):
        bins = np.linspace(*rng[i], 61)
        ax.hist(d[f"f{i}"][m], bins, weights=cv, density=True, histtype="step",
                lw=2, color=C[0], label=f"sBruce {det} (as fed)")
        ax.hist(d[f"f{i}"][m], bins, weights=cv * d["w"][m], density=True,
                histtype="step", lw=1.2, color=C[2], label=f"sBruce {det} x BDT")
        if i in (1, 4, 6):
            ax.hist(d[f"fnf{i}"][m], bins, weights=cv, density=True,
                    histtype="step", lw=1.2, ls=":", color=C[0],
                    label="sBruce, NO y flip")
        if src is not None:
            ax.hist(src[FEATS[i]], bins, density=True, histtype="step", lw=1.5,
                    color=INK2, ls="--", label="train source (SuSAv2 C12)")
            ax.hist(src[FEATS[i]], bins, weights=src["weight"], density=True,
                    histtype="step", lw=1.2, color=C[4], ls="--",
                    label="train source x BDT")
        if tgt is not None:
            ax.hist(tgt[FEATS[i]], bins, weights=tgt["weight"], density=True,
                    histtype="step", lw=1.5, color=C[1], label="train target (Valencia)")
        ax.set_xlabel(FEAT_LABEL[i] + " [GeV]")
        ax.set_yticks([])
    axs[0, 0].legend(fontsize=6)
    fig.suptitle(f"{det}: BDT input features (area-normalized)", color=INK)
    fig.tight_layout()
    return fig


def fig_ordering(d, det, src, tgt):
    m = d["mask"]
    cv = d["cvwgt"][m]
    r = d["p2"][m] / d["p1"][m]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axs[0]
    bins = np.linspace(0, 3, 61)
    ax.hist(r, bins, weights=cv, density=True, histtype="step", lw=2, color=C[0],
            label=f"sBruce {det}")
    if src is not None:
        rs = csv_pmag(src, "subleading_proton") / csv_pmag(src, "leading_proton")
        ax.hist(rs, bins, density=True, histtype="step", lw=1.5, ls="--",
                color=INK2, label=f"train source: frac > 1 = {np.mean(rs > 1):.3f}")
    if tgt is not None:
        rt = csv_pmag(tgt, "subleading_proton") / csv_pmag(tgt, "leading_proton")
        ax.hist(rt, bins, weights=tgt["weight"], density=True, histtype="step",
                lw=1.5, color=C[1],
                label=f"train target: frac > 1 = {np.average(rt > 1, weights=tgt['weight']):.3f}")
    ax.axvline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("|p_subleading| / |p_leading|")
    ax.set_title("'leading' / 'subleading' ordering", fontsize=10, color=INK)
    ax.legend(fontsize=8)
    ax = axs[1]
    rb = np.linspace(0, 1, 21)
    prof(ax, r, d["raw"][m] * d["scale"][m], cv, rb, C[0], "BDT weight (normalized)")
    ax.set_xlabel("|p_subleading| / |p_leading| (sBruce)")
    ax.set_ylabel("weight")
    ax.set_yscale("log")
    ax.set_title("BDT weight vs momentum ratio", fontsize=10, color=INK)
    ax.legend(fontsize=8)
    fig.suptitle(f"{det}: proton ordering", color=INK)
    fig.tight_layout()
    return fig


def prof(ax, x, w, cv, bins, color, label):
    idx = np.digitize(x, bins) - 1
    mean = np.full(len(bins) - 1, np.nan)
    lo = np.full(len(bins) - 1, np.nan)
    hi = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        s = idx == i
        if s.sum() >= 5:
            mean[i] = np.average(w[s], weights=cv[s])
            lo[i], hi[i] = np.percentile(w[s], [16, 84])
    xc = 0.5 * (bins[1:] + bins[:-1])
    ax.fill_between(xc, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(xc, mean, "o-", color=color, ms=3, lw=1.5, label=label)
    ax.axhline(1, color=INK2, lw=0.8, ls=":")


KIN = [("q0", "q0 [GeV]", np.linspace(0, 1.2, 25)),
       ("q3", "q3 [GeV]", np.linspace(0, 1.6, 33)),
       ("enu", "E_nu [GeV]", np.linspace(0, 3, 31)),
       ("W", "W [GeV]", np.linspace(0.8, 2.0, 25)),
       ("lep_p", "p_mu (pre-FSI) [GeV]", np.linspace(0, 2.5, 26)),
       ("cos_mu", "cos theta_mu (nu frame)", np.linspace(-1, 1, 21)),
       ("p1", "|p_lead| [GeV]", np.linspace(0, 1.5, 31))]


def fig_weight_vs_kin(d, det):
    m = d["mask"]
    cv = d["cvwgt"][m]
    kn = d["k1"][m] * np.sum(cv) / np.sum(cv * d["k1"][m])
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, (key, xl, bins) in zip(axs.flat, KIN):
        prof(ax, d[key][m], d["w"][m], cv, bins, C[0], "mec_bdt")
        prof(ax, d[key][m], kn, cv, bins, C[1], "SuSAToVal knob +1 (norm.)")
        ax.set_xlabel(xl)
        ax.set_ylim(0, 3)
    ax = axs.flat[-1]
    ax.hist(d["q0"][m], KIN[0][2], weights=cv, color=GRID)
    ax.set_xlabel("q0 [GeV] (domain events)")
    for e in Q0_EDGES[1:-1]:
        ax.axvline(e, color=INK2, lw=0.8, ls=":")
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].set_ylabel("cvwgt-weighted mean weight (16-84%)")
    fig.suptitle(f"{det}: weight vs kinematics (numu CC MEC pp events)", color=INK)
    fig.tight_layout()
    return fig


def fig_knob_compare(d, det):
    m = d["mask"]
    cv = d["cvwgt"][m]
    kn = d["k1"][m] * np.sum(cv) / np.sum(cv * d["k1"][m])
    specs = [KIN[0], KIN[1], KIN[2], KIN[4], KIN[5]]
    fig, axs = plt.subplots(2, 5, figsize=(18, 6.5), sharex="col",
                            gridspec_kw=dict(height_ratios=[2.5, 1]))
    for c, (key, xl, bins) in enumerate(specs):
        x = d[key][m]
        h0, _ = np.histogram(x, bins, weights=cv)
        hb, _ = np.histogram(x, bins, weights=cv * d["w"][m])
        hk, _ = np.histogram(x, bins, weights=cv * kn)
        ax, axr = axs[0, c], axs[1, c]
        ax.stairs(h0, bins, color=NOM_COLOR, lw=2, label="nominal (SuSAv2)")
        ax.stairs(hb, bins, color=RW_COLOR, lw=2, label="mec_bdt")
        ax.stairs(hk, bins, color=C[1], lw=2, label="SuSAToVal knob +1, area-norm.")
        for h, col in ((hb, RW_COLOR), (hk, C[1])):
            r = np.divide(h, h0, out=np.full(len(h0), np.nan), where=h0 > 0)
            axr.stairs(r, bins, color=col, lw=1.5, baseline=None)
        axr.axhline(1, color=INK2, lw=0.8, ls=":")
        axr.set_ylim(0.4, 1.8)
        axr.set_xlabel(xl)
    axs[0, 0].legend(fontsize=7)
    axs[0, 0].set_ylabel("events")
    axs[1, 0].set_ylabel("/ nominal")
    fig.suptitle(f"{det}: mec_bdt vs nusystematics SuSAv2->Valencia (q3,q0) knob, "
                 "numu CC MEC pp (shapes)", color=INK)
    fig.tight_layout()
    return fig


def fig_knob_q3q0(d, det):
    m = d["mask"]
    cv = d["cvwgt"][m]
    kn = d["k1"][m] * np.sum(cv) / np.sum(cv * d["k1"][m])
    b3, b0 = np.linspace(0, 1.4, 15), np.linspace(0, 1.0, 11)
    h0, _, _ = np.histogram2d(d["q3"][m], d["q0"][m], [b3, b0], weights=cv)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    norm = TwoSlopeNorm(vmin=0.3, vcenter=1.0, vmax=3.0)
    for ax, (lab, w) in zip(axs, (("mec_bdt", d["w"][m]),
                                   ("SuSAToVal knob +1 (area-norm.)", kn))):
        h1, _, _ = np.histogram2d(d["q3"][m], d["q0"][m], [b3, b0], weights=cv * w)
        r = np.where(h0 > 0, h1 / np.where(h0 > 0, h0, 1), np.nan)
        im = ax.pcolormesh(b3, b0, r.T, cmap="RdBu_r", norm=norm)
        for i in range(len(b3) - 1):
            for j in range(len(b0) - 1):
                if np.isfinite(r[i, j]) and h0[i, j] > 0.002 * h0.sum():
                    ax.text(0.5 * (b3[i] + b3[i + 1]), 0.5 * (b0[j] + b0[j + 1]),
                            f"{r[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_xlabel("q3 [GeV]")
        ax.set_ylabel("q0 [GeV]")
        ax.set_title(lab, fontsize=10, color=INK)
        ax.grid(False)
    fig.colorbar(im, ax=axs, label="mean weight")
    fig.suptitle(f"{det}: mean weight in (q3, q0), numu CC MEC pp", color=INK)
    return fig


def pair_class(d, sel):
    pp = sel & d["hasp2"]
    nn = sel & d["hasn2"] & ~d["hasp2"]
    np_ = sel & d["hasp"] & d["hasn"] & ~d["hasp2"] & ~d["hasn2"]
    other = sel & ~(pp | nn | np_)
    return [("pp", pp), ("np", np_), ("nn", nn), ("other", other)]


def fig_pairs(d, det):
    rows = [("numu CC MEC", d["mec"] & (d["true_isnc"] == 0) & (d["true_pdg"] == 14)),
            ("numubar CC MEC", d["mec"] & (d["true_isnc"] == 0) & (d["true_pdg"] == -14)),
            ("NC MEC", d["mec"] & (d["true_isnc"] == 1))]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    cv = d["cvwgt"]
    for i, (lab, sel) in enumerate(rows):
        left = 0.0
        tot = cv[sel].sum()
        for j, (pl, m) in enumerate(pair_class(d, sel)):
            f = cv[m].sum() / tot if tot > 0 else 0
            ax.barh(i, f, left=left, color=C[j], label=pl if i == 0 else None)
            if f > 0.04:
                ax.text(left + f / 2, i, f"{f:.2f}", ha="center", va="center",
                        fontsize=8, color="white")
            left += f
    ax.set_yticks(range(len(rows)), [f"{lab} (N={s.sum()})" for lab, s in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("cvwgt fraction (pre-FSI nucleon pair)")
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title(f"{det}: MEC pre-FSI pair content (BDT applies to numu CC pp only)",
                 fontsize=10, color=INK, pad=24)
    fig.tight_layout()
    return fig


def fig_xflip(d, det, src):
    m = d["mask"]
    lr = np.log(d["raw_xf"][m] / d["raw"][m])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axs[0]
    ax.hist(lr, np.linspace(-1.5, 1.5, 121), weights=d["cvwgt"][m], color=C[0])
    ax.set_yscale("log")
    ax.set_xlabel("ln( w(p1x, p2x -> -p1x, -p2x) / w )")
    ax.set_title(f"x-flip sensitivity: rms {np.std(lr):.3f}, "
                 f"16/84% [{np.percentile(lr, 16):+.3f}, {np.percentile(lr, 84):+.3f}]",
                 fontsize=10, color=INK)
    ax = axs[1]
    bins = np.linspace(-0.8, 0.8, 61)
    ax.hist(d["f0"][m], bins, weights=d["cvwgt"][m], density=True, histtype="step",
            lw=2, color=C[0], label="sBruce p1 px")
    ax.hist(-d["f0"][m], bins, weights=d["cvwgt"][m], density=True,
            histtype="step", lw=1.2, ls="--", color=C[0], label="sBruce -p1 px")
    if src is not None:
        ax.hist(src["leading_proton_px"], bins, density=True, histtype="step",
                lw=1.5, color=INK2, label="train source p1 px")
        ax.hist(-src["leading_proton_px"], bins, density=True, histtype="step",
                lw=1.2, ls="--", color=INK2, label="train source -p1 px")
    ax.set_xlabel("p1 px [GeV] (BDT frame)")
    ax.legend(fontsize=8)
    ax.set_title("mirror symmetry in x", fontsize=10, color=INK)
    fig.suptitle(f"{det}: does the x orientation matter?", color=INK)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- summary

def summarize(out, d, det, nfiles, src, tgt, model_ok):
    cv = d["cvwgt"]
    m = d["mask"]
    w = d["w"]
    out.text(f"== {det}: {nfiles} files, {len(w)} slices")
    common.weight_summary(out, w, cv, "mec_bdt (per-file norm, before clip)")
    out.text(f"  per-file norm scales: {np.unique(np.round(d['scale'], 4))}")
    h = d["hasl"]
    out.check(bool(np.all(np.abs(d["lep_px"][h]) < 1e-6)),
              f"stored frame: |genie_prefsi_lep_px| < 1e-6 GeV (max {np.max(np.abs(d['lep_px'][h])):.1e})")
    out.check(bool(np.all(d["lep_py"][h] >= 0)),
              f"stored frame: genie_prefsi_lep_py >= 0 ({np.sum(d['lep_py'][h] < 0)} negative)")
    out.check(bool(np.max(d["dfeat"]) < 1e-6),
              f"calculator features == BDTReweight reaction-frame transform of the sBruce "
              f"vectors (max |diff| {np.max(d['dfeat']):.1e}; implies x_train == x_sBruce, "
              f"y_train == -y_sBruce)")
    if src is not None:
        out.check(model_ok[0] < 1e-6,
                  f"JSON BDT reproduces the training-source CSV 'weight' column "
                  f"(predict_weights output), max rel diff {model_ok[0]:.1e} on {model_ok[1]} rows")
        sy = d["f1"][m] + d["f4"][m] + d["f6"][m]
        ty = src["leading_proton_py"] + src["subleading_proton_py"] + src["leading_muon_py"]
        out.text(f"  mean sum-py (p1+p2+mu) BDT frame: sBruce {np.average(sy, weights=cv[m]):+.4f}, "
                 f"train source {ty.mean():+.4f} GeV; without the y flip sBruce would be "
                 f"{-np.average(sy, weights=cv[m]):+.4f}")
        for i, k in enumerate(FEATS):
            out.text(f"    feature {k:22s} mean sBruce {np.average(d[f'f{i}'][m], weights=cv[m]):+.3f}"
                     f"  train source {src[k].mean():+.3f}"
                     + (f"  target {np.average(tgt[k], weights=tgt['weight']):+.3f}" if tgt is not None else ""))
    # independent mask
    sel = (d["genie_mode"] == MODE_MEC) & (d["true_isnc"] == 0) & (d["true_pdg"] == 14) \
        & d["hasl"] & d["hasp"] & d["hasp2"]
    out.check(bool(np.all(w[~sel] == 1.0)), "w == 1 exactly outside numu CC MEC with lep/p/p2 filled")
    out.check(bool(np.mean(w[sel] != 1.0) > 0.999), f"w != 1 inside the domain ({np.mean(w[sel] != 1.0):.4f})")
    for lab, s in (("numubar CC MEC", d["mec"] & (d["true_isnc"] == 0) & (d["true_pdg"] == -14)),
                   ("NC MEC", d["mec"] & (d["true_isnc"] == 1)),
                   ("nue/nuebar MEC", d["mec"] & (np.abs(d["true_pdg"]) == 12))):
        out.check(bool(np.all(w[s] == 1.0)), f"{lab} excluded by design: w == 1 (N={s.sum()})")
    out.check(bool(np.max(np.abs(d["closure"] - 1)) < 1e-9),
              f"per-file closure: cvwgt-weighted mean w over masked events == 1 "
              f"(max |dev| {np.max(np.abs(d['closure'] - 1)):.1e}, before clip)")
    nclip = np.sum(w[m] > WEIGHT_CLIP[1])
    after = np.sum((cv * np.clip(w, *WEIGHT_CLIP))[m]) / np.sum(cv[m])
    out.check(abs(after - 1) < 0.01,
              f"{nclip} of {m.sum()} domain events above the driver clip {WEIGHT_CLIP[1]} "
              f"(max {w[m].max():.1f}); closure after clip {after:.4f}", warn=True)
    # per-file normalization robustness: one event must not dominate the sum
    for fi in np.unique(d["_file"]):
        s = m & (d["_file"] == fi)
        if not s.any():
            continue
        cr = (cv * d["raw"])[s]
        share = cr.max() / cr.sum()
        out.check(share < 0.05,
                  f"file {fi}: largest single-event share of the per-file norm sum "
                  f"{share:.2e} (raw w max {d['raw'][s].max():.3g}, scale {d['scale'][s][0]:.3g}, "
                  f"N={s.sum()}; median normalized w {np.median(w[s]):.3g})")
    numu_mec = d["mec"] & (d["true_isnc"] == 0) & (d["true_pdg"] == 14)
    for lab, s in pair_class(d, numu_mec):
        out.text(f"  numu CC MEC pair {lab:5s}: {cv[s].sum() / cv[numu_mec].sum():.3f} of cvwgt")
    extra = m & d["hasn"]
    out.check(extra.sum() == 0,
              f"domain events with a pre-FSI neutron too (more than 2 nucleons): {extra.sum()}",
              warn=True)
    nod = numu_mec & ~m
    out.text(f"  numu CC MEC NOT reweighted (no pp): {cv[nod].sum() / cv[numu_mec].sum():.3f} of cvwgt")
    # x-flip
    lr = np.log(d["raw_xf"][m] / d["raw"][m])
    out.check(np.std(lr) < 0.05,
              f"x-flip sensitivity: rms ln(w_xflip/w) = {np.std(lr):.3f}; "
              f"cvwgt-weighted mean ratio {np.average(d['raw_xf'][m] / d['raw'][m], weights=cv[m]):.4f}",
              warn=True)
    if tgt is not None:
        rt = csv_pmag(tgt, "subleading_proton") / csv_pmag(tgt, "leading_proton")
        rs = csv_pmag(src, "subleading_proton") / csv_pmag(src, "leading_proton")
        out.check(np.mean(rt > 1) < 0.01,
                  f"training target (Valencia) 'leading' proton has the larger |p|: "
                  f"|p_sub| > |p_lead| in {np.mean(rt > 1):.3f} of target rows "
                  f"(source: {np.mean(rs > 1):.3f}) -- the BDT learns an ordering "
                  f"mismatch, not physics", warn=True)
    # knob orientation and domain
    k1, km1 = d["k1"], d["km1"]
    act = np.abs(k1 - 1) > 1e-6
    n_pos = np.sum(act & (np.abs(km1 - 0.1) < 1e-6) & (k1 > 1.9 + 1e-4))
    n_neg = np.sum(act & (np.abs(k1 - 0.1) < 1e-6) & (km1 > 1.9 + 1e-4))
    out.check(n_pos > 0 and n_neg == 0,
              f"SuSAToVal knob orientation: label +1 = d +1 = full SuSAv2->Valencia shift "
              f"(w(-1) floored at 0.1 with w(+1) > 1.9: {n_pos}; reverse: {n_neg})")
    lin = act & (np.abs(k1 + km1 - 2) < 1e-4) & (k1 < 4)
    r2 = (d["k2"][lin] - 1) / (k1[lin] - 1)
    out.text(f"  knob linear in label: (w(+2)-1)/(w(+1)-1) median {np.median(r2):.4f} "
             f"(2 expected; < 2 where clamped)")
    for lab, s in (("numu CC MEC", numu_mec),
                   ("numubar CC MEC", d["mec"] & (d["true_isnc"] == 0) & (d["true_pdg"] == -14)),
                   ("NC MEC", d["mec"] & (d["true_isnc"] == 1)),
                   ("non-MEC", ~d["mec"])):
        out.text(f"  knob +1 active (w != 1): {lab:15s} {np.sum(act & s):6d} of {s.sum():6d}")
    out.check(np.sum(act & ~d["mec"]) == 0, "knob responds only to MEC")
    # knob vs BDT, coarse q0 bins
    cvm = cv[m]
    kn = k1[m] * np.sum(cvm) / np.sum(cvm * k1[m])
    out.text(f"  knob +1 cvwgt-weighted mean on BDT-domain events (rate change): "
             f"{np.average(k1[m], weights=cvm):.4f}; on all numu CC MEC "
             f"{np.average(k1[numu_mec], weights=cv[numu_mec]):.4f}")
    rows = []
    for lo, hi in zip(Q0_EDGES[:-1], Q0_EDGES[1:]):
        s = (d["q0"][m] >= lo) & (d["q0"][m] < hi)
        if s.any():
            rows.append((lo, hi, np.average(d["w"][m][s], weights=cvm[s]),
                         np.average(kn[s], weights=cvm[s]), cvm[s].sum() / cvm.sum()))
    for lo, hi, b, k, f in rows:
        out.text(f"    q0 [{lo:.1f},{hi:.1f}) frac {f:.3f}: mean BDT {b:.3f}  mean knob(norm) {k:.3f}")
    for key, xl, bins in (KIN[0], KIN[1], KIN[2], KIN[4], KIN[5]):
        x = d[key][m]
        h0, _ = np.histogram(x, bins, weights=cvm)
        hb, _ = np.histogram(x, bins, weights=cvm * d["w"][m])
        hk, _ = np.histogram(x, bins, weights=cvm * kn)
        ok = h0 > 0.01 * h0.max()
        rb, rk = hb[ok] / h0[ok], hk[ok] / h0[ok]
        corr = np.corrcoef(rb, rk)[0, 1] if ok.sum() > 2 else np.nan
        out.text(f"    shape ratio vs {key:6s}: BDT range [{rb.min():.2f},{rb.max():.2f}], "
                 f"knob range [{rk.min():.2f},{rk.max():.2f}], corr {corr:+.2f}")
    out.text("")


def main():
    args = common.parse_args(__doc__.splitlines()[0])
    by_det = common.files_by_detector(args)
    out = common.Output("mec_bdt", "mec_bdt validation (validation/mec_bdt/validate_mec_bdt.py)")
    model = GBReweighterJSON(MODEL_PATH)
    src = load_csv(TRAIN_SRC)
    tgt = load_csv(TRAIN_TGT)
    model_ok = (np.nan, 0)
    if src is not None:
        nchk = 50000
        wp = model.predict_weights(csv_feats(src)[:nchk])
        model_ok = (float(np.max(np.abs(wp / src["weight"][:nchk] - 1))), nchk)
    else:
        out.text(f"  (training CSVs not found under {TRAIN_DIR}: training overlays skipped)")
    for det, files in by_det.items():
        d = common.load(files, process, model)
        out.save(common.fig_domain(d, {"mec_bdt": d["w"], "SuSAToVal +1": d["k1"]},
                                   det, "mec_bdt"), f"domain_{det}")
        out.save(common.fig_weight_hists({"mec_bdt": d["w"], "SuSAToVal knob +1": d["k1"]},
                                         det, "mec_bdt", d["cvwgt"]), f"weight_hists_{det}")
        out.save(fig_frame(d, det, src), f"frame_{det}")
        out.save(fig_features(d, det, src, tgt), f"features_{det}")
        out.save(fig_ordering(d, det, src, tgt), f"ordering_{det}")
        out.save(fig_xflip(d, det, src), f"xflip_{det}")
        out.save(fig_weight_vs_kin(d, det), f"weight_vs_kin_{det}")
        out.save(fig_knob_compare(d, det), f"knob_compare_{det}")
        out.save(fig_knob_q3q0(d, det), f"knob_q3q0_{det}")
        out.save(fig_pairs(d, det), f"pairs_{det}")
        summarize(out, d, det, len(files), src, tgt, model_ok)
    return out.close()


if __name__ == "__main__":
    sys.exit(main())
