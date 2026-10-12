#!/usr/bin/env python3
"""Audit of what the sBruce input branches actually mean.

The calculators read ~60 SelectedEvents branches under assumptions written
in reweight.py's header and fakedata/sbruce.py (enum values, sentinels,
frames, units, "leading" definitions). This script tests each assumption on
the CV MC files against the data itself and, where possible, against an
independent witness -- GENIE's own reweight knobs in multisigmaTree, or a
recomputation from other branches. Per detector:

  knob_response     for every GENIE multisigma knob, the fraction of events
                    in each (flavour, CC/NC, genie_mode) category whose
                    weight at the knob's first non-zero sigma label differs
                    from 1. A knob responds to exactly its own process, so
                    this is GENIE's word on what genie_mode / true_isnc /
                    true_pdg mean (e.g. NormCCMEC must light up mode 10 CC
                    only).
  knob_orientation  the mean weight vs sigma label on each knob's own
                    process, for knobs whose physical direction is known
                    (+1 sigma on a normalization / MA / fate fraction raises
                    its own process). A curve falling with the label means
                    the stored labels are reversed.
  sentinels         per branch, the fraction of matched-neutrino and of
                    cosmic / unmatched (true_pdg == -1) slices at -999 / -1
                    / -128 / 0 -- what "unfilled" looks like per branch.
  energies          nu_E, true_E, genie_Enu agreement; nu_E0 vs true_E.
  prefsi_frame      lep px == 0 and py >= 0; |p_lep| (pre-FSI) vs true_mu_p
                    (post-FSI; a lepton has no FSI); genie_q0 / genie_q3
                    recomputed from Enu and genie_prefsi_lep_*; genie_W
                    recomputed as an on-shell nucleon at rest.
  beam_vs_detz      cos(theta_mu) to detector z (true_mu_dir_z) minus
                    cos(theta_mu) to the neutrino (pre-FSI frame), vs the
                    true vertex x: the post-FSI direction branches are in
                    detector coordinates, not along the neutrino.
  counts_vs_leading true_np / true_npi / true_npi0 vs whether the leading
                    true_p / true_cpi / true_pi0 branch is filled (the
                    counts drop primaries with G4ID < 0 or cryostat < 0).
  photons           true_g_p (leading final-state photon by energy) vs the
                    pre-FSI photon, with the 10 MeV veto line used by
                    spp_lowq2 / minerva_3dqelike, split by whether the event
                    has a pre-FSI photon or pi0 (i.e. whether the post-FSI
                    photon can be a de-excitation gamma).
  misc              baseline [m] per detector, cvwgt per mode, genie_pmiss
                    filled fraction per mode, duplicate truth matches.

Writes validation/plots/inputs/.

Usage: ./venv/bin/python validation/inputs/validate_inputs.py [--max-files N]
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

import common                                                # noqa: E402
from common import plt, INK, INK2, GRID, C, MODE_NAMES       # noqa: E402

M_MU, M_E = 0.1056583755, 0.51099895e-3
M_P, M_N = 0.93827208816, 0.93956542052

BR = ["genie_mode", "true_genie_mode", "true_pdg", "true_nu_pdg", "true_isnc",
      "true_iscc", "true_iscosmic", "true_E", "nu_E", "nu_E0", "genie_Enu",
      "genie_q0", "genie_q3", "genie_W", "genie_pmiss", "genie_emiss",
      "genie_prefsi_lep_px", "genie_prefsi_lep_py", "genie_prefsi_lep_pz",
      "genie_prefsi_p_px", "genie_prefsi_n_px", "genie_prefsi_cpi_px",
      "genie_prefsi_cpi_fsi",
      "genie_prefsi_pi0_px", "genie_prefsi_pi0_py", "genie_prefsi_pi0_pz",
      "genie_prefsi_g_px", "genie_prefsi_g_py", "genie_prefsi_g_pz",
      "true_mu_p", "true_mu_dir_x", "true_mu_dir_y", "true_mu_dir_z",
      "true_p_p", "true_p2_p", "true_cpi_p", "true_pi0_p", "true_g_p",
      "true_np", "true_nn", "true_npi", "true_npi0", "true_nmu",
      "true_vtx_x_y", "true_vtx_y_y", "true_vtx_z_y",
      "baseline", "cvwgt", "run", "evt", "tmatch_idx"]

V1 = "GENIEReWeight_SBN_v1_multisigma_{}"
V1_KNOBS = ["CoulombCCQE", "VecFFCCQEshape", "NormCCMEC", "NormNCMEC",
            "DecayAngMEC", "MaCCRES", "MaNCRES", "MvCCRES", "MvNCRES",
            "RDecBR1gamma", "RDecBR1eta", "Theta_Delta2Npi", "ThetaDelta2NRad",
            "AhtBY", "BhtBY", "CV1uBY", "CV2uBY", "NormCCCOH", "NormNCCOH",
            "MFP_pi", "FrCEx_pi", "FrInel_pi", "FrAbs_pi", "FrPiProd_pi",
            "MaNCEL", "EtaNCEL"]
V3_KNOBS = (["multisigma_ZExpPCAWeighter_SBN_v3_MvA_b1"]
            + [f"multisigma_MECq0q3InterpWeighting_SBN_v3_SuSAToVal_MECResponse_q0bin{i}"
               for i in range(2)]
            + ["multisigma_CCQETemplateReweight_SBN_v3_LFGToSF_q0bin1",
               "multisigma_QEInterference_SBN_v3_QEIntf_dial_0",
               "multisigma_GENIEReWeight_SBN_v3_FrG4LoE_N"])


def knob_name(k):
    return k if k.startswith("multisigma_") else V1.format(k)


def short(k):
    return (k.replace("multisigma_", "").replace("_SBN_v3", "")
            .replace("MECq0q3InterpWeighting_SuSAToVal_MECResponse_", "MEC_SuSAToVal_")
            .replace("CCQETemplateReweight_", "CCQETmpl_")
            .replace("GENIEReWeight_", ""))


# own process of the knobs with a known physical direction: (mask fn,
# expected sign of (w - 1) at TRUE +1 sigma)
def orientation_targets(d):
    cc, nc = d["true_isnc"] == 0, d["true_isnc"] == 1
    nu = d["true_pdg"] != -1
    m, fsi = d["genie_mode"], d["genie_prefsi_cpi_fsi"]
    return {
        "NormCCMEC": nu & cc & (m == 10), "NormNCMEC": nu & nc & (m == 10),
        "NormCCCOH": nu & cc & (m == 3), "NormNCCOH": nu & nc & (m == 3),
        "MaCCRES": nu & cc & (m == 1), "MaNCRES": nu & nc & (m == 1),
        "MaNCEL": nu & nc & (m == 0),
        "FrCEx_pi": fsi == 2, "FrInel_pi": fsi == 3, "FrAbs_pi": fsi == 4,
        "FrPiProd_pi": fsi == 7,
    }


def process(sb):
    d = sb.arrays([b for b in BR if sb.has_branch(b)])
    d = {k: np.asarray(v) for k, v in d.items()}
    labels = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
    for k in V1_KNOBS + V3_KNOBS:
        name = knob_name(k)
        if not sb.has_multisigma(name):
            continue
        s = sb._file["multisigmaTree"][name + "_sigma"].array(
            entry_stop=1, library="np")[0]
        s = np.asarray(s, dtype=float)
        first = s[s != 0][-1] if np.any(s > 0) else s[s != 0][0]
        first = 1.0 if 1.0 in s else first
        d[f"knob1:{k}"] = sb.multisigma_at_sigma(name, first)
        if len(s) == 7:
            for lab in labels:
                d[f"knob@{lab:+.0f}:{k}"] = sb.multisigma_at_sigma(name, lab)
    return d


# ---------------------------------------------------------------- figures

def fig_knob_response(d, det):
    cats = [(c, m) for c, m in common.categories(d) if m.sum() >= 5]
    knobs = [k for k in V1_KNOBS + V3_KNOBS if f"knob1:{k}" in d]
    mat = np.zeros((len(cats), len(knobs)))
    for i, (_, m) in enumerate(cats):
        for j, k in enumerate(knobs):
            w = d[f"knob1:{k}"][m]
            mat[i, j] = np.mean(np.isfinite(w) & (np.abs(w - 1) > 1e-6))
    fig, ax = plt.subplots(figsize=(15, 0.33 * len(cats) + 2.4))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(cats)):
        for j in range(len(knobs)):
            if mat[i, j] > 0.005:
                ax.text(j, i, f"{mat[i, j]:.2f}"[1:] if mat[i, j] < 0.995 else "1",
                        ha="center", va="center", fontsize=6,
                        color="white" if mat[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(knobs)), [short(k) for k in knobs], rotation=60,
                  ha="right", fontsize=7)
    ax.set_yticks(range(len(cats)), [f"{c} (N={m.sum()})" for c, m in cats],
                  fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="fraction of events with w != 1")
    ax.set_title(f"{det}: which (flavour, CC/NC, genie_mode) each GENIE knob "
                 "responds to", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def knob_orientation(d):
    """{knob: (labels, mean w on own process at each label, N)}"""
    out = {}
    labels = [-3, -2, -1, 0, 1, 2, 3]
    for k, m in orientation_targets(d).items():
        if f"knob@+1:{k}" not in d or not m.any():
            continue
        ys = []
        for lab in labels:
            w = d[f"knob@{lab:+d}:{k}"][m]
            ys.append(np.nanmean(w))
        out[k] = (labels, np.array(ys), int(m.sum()))
    return out


def fig_knob_orientation(orient, det):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (k, (x, y, n)) in enumerate(orient.items()):
        ax.plot(x, y, "o-", color=C[i % len(C)] if i < 6 else None, lw=1.5,
                ms=4, label=f"{k} (N={n})")
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("stored sigma label")
    ax.set_ylabel("mean weight on the knob's own process")
    ax.set_title(f"{det}: knob orientation -- every curve should RISE "
                 "(+1 sigma raises its own process)", fontsize=10, color=INK)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def fig_sentinels(d, det):
    br = ["true_pdg", "true_isnc", "true_iscc", "genie_mode", "true_genie_mode",
          "true_E", "genie_Enu", "true_np", "true_npi", "true_mu_p", "true_g_p",
          "genie_W", "genie_pmiss", "genie_prefsi_lep_pz", "cvwgt", "baseline",
          "tmatch_idx"]
    br = [b for b in br if b in d]
    vals = [(-999, "-999"), (-1, "-1"), (-128, "-128"), (0, "0"), (1, "1")]
    cos = d["true_pdg"] == -1
    fig, axs = plt.subplots(1, 2, figsize=(13, 0.35 * len(br) + 2), sharey=True)
    for ax, (lab, m) in zip(axs, [("matched neutrino", ~cos),
                                  ("cosmic / unmatched (true_pdg == -1)", cos)]):
        mat = np.array([[np.mean(d[b][m] == v) if m.any() else 0
                         for v, _ in vals] for b in br])
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(br)):
            for j in range(len(vals)):
                if mat[i, j] > 0:
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                            fontsize=7, color="white" if mat[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(vals)), [v for _, v in vals])
        ax.set_yticks(range(len(br)), [f"{b} ({d[b].dtype})" for b in br],
                      fontsize=8)
        ax.set_title(f"{lab}, N={m.sum()}", fontsize=10, color=INK)
        ax.grid(False)
    fig.suptitle(f"{det}: fraction of slices at each special value", color=INK)
    fig.tight_layout()
    return fig


def lep_mass(pdg):
    return np.where(np.abs(pdg) == 12, M_E, M_MU)


def derived(d):
    nu = d["true_pdg"] != -1
    cc = nu & (d["true_isnc"] == 0)
    lx, ly, lz = (d["genie_prefsi_lep_px"].astype(np.float64),
                  d["genie_prefsi_lep_py"].astype(np.float64),
                  d["genie_prefsi_lep_pz"].astype(np.float64))
    has_l = nu & (lz > -900)
    pl = np.sqrt(lx ** 2 + ly ** 2 + lz ** 2)
    E = d["genie_Enu"].astype(np.float64)
    ml = np.where(d["true_isnc"] == 1, 0.0, lep_mass(d["true_pdg"]))
    El = np.sqrt(pl ** 2 + ml ** 2)
    q0 = E - El
    q3 = np.sqrt(lx ** 2 + ly ** 2 + (E - lz) ** 2)
    Q2 = q3 ** 2 - q0 ** 2
    W2p = M_P ** 2 + 2 * M_P * q0 - Q2
    W2n = M_N ** 2 + 2 * M_N * q0 - Q2
    W2a = (0.5 * (M_P + M_N)) ** 2 + (M_P + M_N) * q0 - Q2
    Wc = [np.sqrt(np.clip(x, 0, None)) for x in (W2p, W2n, W2a)]
    return dict(nu=nu, cc=cc, has_l=has_l, pl=pl, q0=q0, q3=q3, Wc=Wc,
                cos_nu=np.divide(lz, pl, out=np.full_like(pl, np.nan),
                                 where=pl > 0))


def fig_prefsi(d, x, det):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    h = x["has_l"]
    ax = axs[0, 0]
    ax.hist(d["genie_prefsi_lep_px"][h], np.linspace(-1e-6, 1e-6, 81),
            histtype="step", color=C[0], label="lep px")
    ax.set_xlabel("genie_prefsi_lep_px [GeV]")
    ax.set_title(f"lep px (|px| max {np.max(np.abs(d['genie_prefsi_lep_px'][h])):.1e})",
                 fontsize=9)
    ax = axs[0, 1]
    ax.hist(d["genie_prefsi_lep_py"][h], np.linspace(-0.2, 1.5, 86),
            histtype="step", color=C[0])
    ax.set_xlabel("genie_prefsi_lep_py [GeV]")
    ax.set_title(f"lep py < 0: {np.sum(d['genie_prefsi_lep_py'][h] < 0)} events",
                 fontsize=9)
    ax = axs[0, 2]
    m = h & x["cc"] & (np.abs(d["true_pdg"]) == 14) & (d["true_mu_p"] > -900)
    r = x["pl"][m] - d["true_mu_p"][m]
    ax.hist(r, np.linspace(-1e-3, 1e-3, 81), histtype="step", color=C[0])
    ax.set_yscale("log")
    ax.set_xlabel("|p_lep| pre-FSI - true_mu_p [GeV]")
    ax.set_title(f"numu CC: |diff| > 1e-4 in {np.sum(np.abs(r) > 1e-4)} / {m.sum()}",
                 fontsize=9)
    for ax, key, lab in ((axs[1, 0], "q0", "genie_q0"), (axs[1, 1], "q3", "genie_q3")):
        m = h & (d[lab] > -900)
        r = x[key][m] - d[lab][m]
        ax.hist(np.clip(r, -1e-3, 1e-3), np.linspace(-1e-3, 1e-3, 81),
                histtype="step", color=C[0])
        ax.set_yscale("log")
        ax.set_xlabel(f"recomputed - {lab} [GeV] (clipped to +-1e-3)")
        ax.set_title(f"max |diff| {np.max(np.abs(r)):.1e}, N={m.sum()} "
                     "(NC: massless lepton)", fontsize=9)
    ax = axs[1, 2]
    m = h & (d["genie_W"] > -900)
    for lab, Wc, c in zip(("m_p", "m_n", "(m_p+m_n)/2"), x["Wc"], C):
        r = Wc[m] - d["genie_W"][m]
        ax.hist(np.clip(r, -0.01, 0.01), np.linspace(-0.01, 0.01, 81),
                histtype="step", color=c, label=f"M = {lab}")
    best = np.min(np.abs(np.stack([Wc[m] for Wc in x["Wc"]]) - d["genie_W"][m]), axis=0)
    ax.set_yscale("log")
    ax.set_xlabel("recomputed W - genie_W [GeV] (clipped to +-0.01)")
    ax.set_title(f"best of three |diff| > 1e-3: {np.sum(best > 1e-3)} / {m.sum()}",
                 fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle(f"{det}: pre-FSI frame and derived kinematics", color=INK)
    fig.tight_layout()
    return fig


def fig_beam(d, x, det):
    m = x["has_l"] & x["cc"] & (np.abs(d["true_pdg"]) == 14) & (d["true_mu_dir_z"] > -900)
    dc = d["true_mu_dir_z"][m] - x["cos_nu"][m]
    vx = d["true_vtx_x_y"][m] if "true_vtx_x_y" in d else np.zeros(m.sum())
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    axs[0].hist(dc, np.linspace(-0.06, 0.06, 121), histtype="step", color=C[0])
    p = np.percentile(dc, [1, 50, 99])
    axs[0].set_xlabel("cos theta_mu (detector z) - cos theta_mu (neutrino)")
    axs[0].set_title(f"1/50/99%: {p[0]:+.4f} / {p[1]:+.4f} / {p[2]:+.4f}",
                     fontsize=9)
    sel = np.random.default_rng(0).choice(len(dc), min(len(dc), 20000), replace=False)
    axs[1].scatter(vx[sel], dc[sel], s=2, alpha=0.3, color=C[0], lw=0)
    axs[1].set_ylim(-0.06, 0.06)
    axs[1].set_xlabel("true vertex x [cm]")
    axs[1].set_ylabel("cos difference")
    fig.suptitle(f"{det}: true_mu_dir is in detector coordinates, not along the "
                 "neutrino (numu CC)", color=INK)
    fig.tight_layout()
    return fig, p


def fig_counts(d, det):
    nu = d["true_pdg"] != -1
    pairs = [("true_np", "true_p_p"), ("true_npi", "true_cpi_p"),
             ("true_npi0", "true_pi0_p")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    res = {}
    for ax, (cnt, lead) in zip(axs, pairs):
        n = d[cnt][nu]
        f = d[lead][nu] > -900
        rows = [("count == 0", n == 0), ("count >= 1", n >= 1),
                ("count = -999", n < -900)]
        mat = np.array([[np.sum(r & ~f), np.sum(r & f)] for _, r in rows])
        ax.imshow(np.log10(mat + 1), cmap="Blues", aspect="auto")
        for i in range(3):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1], [f"{lead} unfilled", f"{lead} filled"], fontsize=8)
        ax.set_yticks(range(3), [f"{cnt} {r}" for r, _ in rows], fontsize=8)
        ax.grid(False)
        res[cnt] = (mat[0, 1], mat[1, 0])
    fig.suptitle(f"{det}: final-state counts vs leading-particle branches "
                 "(off-diagonal = inconsistent)", color=INK)
    fig.tight_layout()
    return fig, res


def fig_photons(d, det):
    nu = d["true_pdg"] != -1
    gpre = d["genie_prefsi_g_px"] > -900
    pi0pre = d["genie_prefsi_pi0_px"] > -900
    g = d["true_g_p"]
    hasg = nu & (g > -900)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bins = np.logspace(-4, 0, 81)
    cv = d["cvwgt"]
    for lab, m, c in (("no pre-FSI photon or pi0", hasg & ~gpre & ~pi0pre, C[0]),
                      ("pre-FSI pi0", hasg & pi0pre, C[1]),
                      ("pre-FSI photon", hasg & gpre & ~pi0pre, C[2])):
        ax.hist(g[m], bins, weights=cv[m], histtype="step", color=c, lw=1.8,
                label=f"{lab} (N={m.sum()})")
    ax.axvline(0.010, color=INK2, ls="--", lw=1)
    ax.text(0.0105, ax.get_ylim()[1] * 0.5, "10 MeV veto", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("true_g_p [GeV]")
    ax.set_ylabel("events (cvwgt)")
    ax.set_title(f"{det}: leading final-state photon -- events with no pre-FSI "
                 "photon/pi0 are de-excitation / FSI gammas", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    deex = nu & ~gpre & ~pi0pre & (g > 0.010)
    return fig, (int(deex.sum()), float(np.sum(cv[deex]) / np.sum(cv[nu])))


def fig_misc(d, det):
    nu = d["true_pdg"] != -1
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    b = d["baseline"][nu & (d["baseline"] > -900)]
    axs[0].hist(b, 80, histtype="step", color=common.DET_COLOR[det])
    axs[0].set_xlabel("baseline [m]")
    axs[0].set_title(f"baseline: {b.min():.0f} - {b.max():.0f} m", fontsize=9)
    for i, (mv, mn) in enumerate(MODE_NAMES.items()):
        m = nu & (d["genie_mode"] == mv)
        if m.any():
            axs[1].hist(d["cvwgt"][m], np.linspace(0, 3, 61), histtype="step",
                        color=C[i], label=f"{mn} (frac != 1: {np.mean(d['cvwgt'][m] != 1):.2f})")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("cvwgt")
    axs[1].legend(fontsize=7)
    names = list(MODE_NAMES.values())
    fr = [np.mean(d["genie_pmiss"][nu & (d["genie_mode"] == mv)] > -900)
          if np.any(nu & (d["genie_mode"] == mv)) else 0 for mv in MODE_NAMES]
    axs[2].bar(names, fr, color=C[0])
    axs[2].set_ylabel("fraction with genie_pmiss filled")
    axs[2].set_ylim(0, 1.05)
    fig.suptitle(f"{det}: baseline, cvwgt per mode, genie_pmiss coverage", color=INK)
    fig.tight_layout()
    return fig


def duplicates(d):
    nu = (d["true_pdg"] != -1) & (d["true_E"] > -900)
    key = np.stack([d["_file"][nu], d["run"][nu], d["evt"][nu],
                    d["true_E"][nu].view(np.int32)], axis=1)
    _, cnt = np.unique(key, axis=0, return_counts=True)
    return int(np.sum(cnt[cnt > 1] - 1)), int(nu.sum())


# ---------------------------------------------------------------- main

def main():
    args = common.parse_args(__doc__)
    out = common.Output("inputs", "sBruce input-branch audit "
                        "(validation/inputs/validate_inputs.py)")
    for det, files in common.files_by_detector(args).items():
        out.text(f"== {det}: {len(files)} files")
        d = common.load(files, process)
        x = derived(d)
        nu = x["nu"]
        out.text(f"  slices: {len(nu)}, matched neutrino {nu.sum()}, "
                 f"cosmic/unmatched {np.sum(~nu)}")

        out.save(fig_knob_response(d, det), f"knob_response_{det}")
        orient = knob_orientation(d)
        out.save(fig_knob_orientation(orient, det), f"knob_orientation_{det}")
        for k, (lab, y, n) in orient.items():
            slope = y[4] - y[2]                     # label +1 minus label -1
            out.check(slope > 0, f"knob {k}: mean w at label +1 / -1 on own "
                      f"process = {y[4]:.3f} / {y[2]:.3f} (N={n}) -- "
                      + ("labels as GENIE" if slope > 0 else
                         "LABELS REVERSED (stored +1 is GENIE -1 sigma)"),
                      warn=True)
        # which (mode, isnc) each process knob responds to
        expect = {"NormCCMEC": (10, 0), "NormNCMEC": (10, 1), "MaCCRES": (1, 0),
                  "MaNCRES": (1, 1), "NormCCCOH": (3, 0), "NormNCCOH": (3, 1),
                  "MaNCEL": (0, 1), "VecFFCCQEshape": (0, 0),
                  "CoulombCCQE": (0, 0)}
        for k, (mv, ncv) in expect.items():
            if f"knob1:{k}" not in d:
                continue
            w = d[f"knob1:{k}"]
            r = np.isfinite(w) & (np.abs(w - 1) > 1e-6)
            own = nu & (d["genie_mode"] == mv) & (d["true_isnc"] == ncv)
            out.check(not np.any(r & ~own),
                      f"knob {k} responds only to genie_mode=={mv}, "
                      f"true_isnc=={ncv}: {np.sum(r & own)} own, "
                      f"{np.sum(r & ~own)} other")

        out.save(fig_sentinels(d, det), f"sentinels_{det}")
        cos = ~nu
        out.check(not (np.all(d["true_isnc"][cos] == 0) and np.all(d["true_iscc"][cos] == 0)),
                  f"cosmic slices have true_isnc == 0 AND true_iscc == 0 "
                  f"({cos.sum()} slices): true_isnc == 0 alone does NOT mean CC; "
                  "a CC selection must also require true_pdg != -1 (or a "
                  "filled genie_mode)", warn=True)
        out.check(bool(np.all((d["true_isnc"][nu] + d["true_iscc"][nu]) == 1)),
                  "matched slices: true_isnc + true_iscc == 1")
        gm = d["genie_mode"]
        out.check(bool(np.all(np.where(gm > -900, gm, -1) == d["true_genie_mode"])),
                  "genie_mode == true_genie_mode (-999 <-> -1)")
        tp = d["true_nu_pdg"]
        out.check(bool(np.all(np.where(tp > -900, tp, -1) == d["true_pdg"])),
                  "true_pdg == true_nu_pdg (-1 <-> -999)")
        modes = set(np.unique(gm[gm > -900]).astype(int))
        out.check(modes <= set(MODE_NAMES), f"genie_mode values {sorted(modes)} "
                  "within the simb enum {0,1,2,3,10}")

        e = nu & (d["true_E"] > -900)
        out.check(bool(np.all(d["true_E"][e] == d["genie_Enu"][e]))
                  and bool(np.all(d["true_E"][e] == d["nu_E"][e])),
                  f"nu_E == true_E == genie_Enu exactly ({e.sum()} slices)")
        out.check(np.all(d["nu_E0"][e] == d["true_E"][e]), f"nu_E0 != true_E on {np.mean(d['nu_E0'][e] != d['true_E'][e]):.1%} "
                  "of slices (spill maximum, not per slice -- never use as the "
                  "event's energy)", warn=True)

        out.save(fig_prefsi(d, x, det), f"prefsi_frame_{det}")
        h = x["has_l"]
        out.check(bool(np.all(np.abs(d["genie_prefsi_lep_px"][h]) < 1e-6))
                  and bool(np.all(d["genie_prefsi_lep_py"][h] >= 0)),
                  "pre-FSI frame: lep px == 0 and py >= 0")
        for k in ("q0", "q3"):
            m = h & (d[f"genie_{k}"] > -900)
            r = np.abs(x[k][m] - d[f"genie_{k}"][m])
            nbad = np.sum(r > 1e-5)
            out.check(nbad <= 1e-4 * m.sum(),
                      f"genie_{k} == recomputed from Enu and genie_prefsi_lep: "
                      f"{nbad} of {m.sum()} differ by > 1e-5 GeV (max "
                      f"{r.max():.1e}; the few outliers are COH events)")
        m = h & (d["genie_W"] > -900)
        best = np.min(np.abs(np.stack([Wc[m] for Wc in x["Wc"]]) - d["genie_W"][m]), axis=0)
        out.check(np.mean(best < 1e-3) > 0.999,
                  f"genie_W == on-shell-nucleon-at-rest W (m_p, m_n or avg) for "
                  f"{np.mean(best < 1e-3):.4%} of {m.sum()}")

        fig, p = fig_beam(d, x, det)
        out.save(fig, f"beam_vs_detz_{det}")
        out.check(max(abs(p[0]), abs(p[2])) < 0.005,
                  f"cos theta_mu (det z) - cos theta_mu (nu): 1/50/99% = "
                  f"{p[0]:+.4f}/{p[1]:+.4f}/{p[2]:+.4f} -- calculators that "
                  "treat detector z as the beam axis (tki.py, minerva) inherit "
                  "this", warn=True)

        fig, res = fig_counts(d, det)
        out.save(fig, f"counts_vs_leading_{det}")
        for cnt, (a, b) in res.items():
            out.check(a == 0 and b == 0, f"{cnt}: {a} slices with count 0 but "
                      f"leading branch filled, {b} with count >= 1 but unfilled",
                      warn=True)

        fig, (nde, fde) = fig_photons(d, det)
        out.save(fig, f"photons_{det}")
        out.check(nde == 0, f"{nde} slices ({fde:.2%} of cvwgt) have true_g_p > "
                  "10 MeV with NO pre-FSI photon or pi0 (de-excitation / FSI "
                  "gammas) -- vetoed by spp_lowq2 / minerva signal definitions",
                  warn=True)

        out.save(fig_misc(d, det), f"misc_{det}")
        b = d["baseline"][nu & (d["baseline"] > -900)]
        out.text(f"  baseline [m]: {b.min():.1f} - {b.max():.1f}")
        ndup, nn = duplicates(d)
        out.check(ndup == 0, f"{ndup} of {nn} matched slices share their "
                  "neutrino with another selected slice (same file/run/evt/E)",
                  warn=True)
        for mv, mn in MODE_NAMES.items():
            m = nu & (gm == mv)
            if m.any():
                out.text(f"  mode {mn:3s}: N={m.sum():6d}  cvwgt != 1: "
                         f"{np.mean(d['cvwgt'][m] != 1):.3f}  genie_pmiss "
                         f"filled: {np.mean(d['genie_pmiss'][m] > -900):.3f}")
        out.text()
    return out.close()


if __name__ == "__main__":
    sys.exit(main())
