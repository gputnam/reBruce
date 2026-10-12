#!/usr/bin/env python3
"""Validation plots for the minerva_3dqelike calculator.

The calculator (fakedata/calculators/minerva_qelike.py, fakedata/tki.py,
data/minerva_3dqelike_bnb.csv) reweights numu CC QE-like events by the
MINERvA LE/ME 3D (p_z, p_T, SumT_p) data/AR23 ratio extrapolated to the BNB,
with p_z scaled by E_LE / E_BNB. Two branches: the 3D lookup, and _pzmarg
(per-file p_z-marginalized, no theta cut). On the CV MC sBruce files this
script runs the real calculator and, per detector (cvwgt-weighted), draws /
checks:

  domain           w != 1 per flavour x CC/NC x mode, both branches
  weights          weight distributions
  occupancy        cvwgt of the 3D signal per (p_T, SumT_p) cell, one panel
                   per scaled-p_z bin, plus the in/out-of-range budget
  closure          independent table lookup (CSV read here, signal rebuilt
                   here) == calculator, both branches; per-cell yield
                   closure of _pzmarg (sum cv*w3d == sum cv*w_marg over each
                   file's in-p_z-range no-theta signal), drawn per cell
  theta_frame      the muon angle measured from detector z (what the
                   calculator uses) vs from the true neutrino direction
                   (cos = genie_prefsi_lep_pz/|lep|, the per-event nu frame):
                   cos-difference per detector, and the resulting 3D-bin
                   migration / weight change when (p_z, p_T) are taken in the
                   neutrino frame (NUISANCE uses the neutrino direction)
  photon_veto      true_g_p for numu CC 0pi candidates (signal minus the
                   photon veto), split by whether GENIE has a pre-FSI photon
                   (genie_prefsi_g excludes de-excitation photons); how much
                   in-range signal the de-excitation photons veto
  spectra          nominal vs weighted muon p_z, p_T, SumT_p of numu CC

Plus text checks: true_np > 2 share of the in-range signal (SumT_p
under-counted); "no mesons" with counts carrying the G4ID/cryostat filter
vs the unfiltered leading-particle branches true_cpi_p / true_pi0_p; muon
|p| post-FSI vs pre-FSI lepton.

Writes validation/plots/minerva_3dqelike/*.png, minerva_3dqelike.pdf and
summary.txt.

Usage: ./venv/bin/python validation/minerva_3dqelike/validate_minerva.py
           [--max-files N] [files...]
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
from matplotlib.colors import LogNorm                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

import common                                                # noqa: E402
from common import C, INK, INK2                              # noqa: E402

import fakedata.calculators  # noqa: E402,F401  (registers calculators)
from fakedata import calculator                             # noqa: E402
from fakedata.sbruce import valid                            # noqa: E402

NAME = "minerva_3dqelike"
B3D = "fdwgt_minerva_3dqelike_bnb"
BMG = B3D + "_pzmarg"
CSV = os.path.join(HERE, "..", "..", "data", "minerva_3dqelike_bnb.csv")
M_P = 0.938272081
PZ_SCALE = 0.70 / 3.25          # E_BNB / E_LE (README)
COS20 = np.cos(np.radians(20.0))

BRANCHES = ["true_isnc", "true_pdg", "genie_mode", "cvwgt", "nu_E",
            "true_mu_p", "true_mu_dir_x", "true_mu_dir_y", "true_mu_dir_z",
            "true_p_p", "true_p2_p", "true_np", "true_npi", "true_npi0",
            "true_g_p", "true_cpi_p", "true_pi0_p",
            "genie_prefsi_lep_px", "genie_prefsi_lep_py", "genie_prefsi_lep_pz",
            "genie_prefsi_g_px", "genie_prefsi_g_py", "genie_prefsi_g_pz"]


def load_table():
    with open(CSV) as f:
        rows = list(csv.DictReader(x for x in f if not x.startswith("#")))
    shape = tuple(1 + max(int(r[k]) for r in rows) for k in ("ipz", "ipt", "itp"))
    w = np.ones(shape)
    exc = np.zeros(shape, bool)
    edges = [set(), set(), set()]
    for r in rows:
        i = (int(r["ipz"]), int(r["ipt"]), int(r["itp"]))
        w[i] = float(r["weight"])
        exc[i] = r["excluded"] == "True"
        for k, ax in enumerate(("pz", "pt", "tp")):
            edges[k] |= {float(r[f"{ax}_lo"]), float(r[f"{ax}_hi"])}
    return [np.array(sorted(e)) for e in edges], w, exc


def axis_idx(v, e):
    i = np.searchsorted(e, v, side="right") - 1
    return np.where((v >= e[0]) & (v < e[-1]) & np.isfinite(v), i, -1)


def lookup(edges, w, pz, pt, tp, sig):
    ii = [np.where(sig, axis_idx(x, e), -1) for x, e in zip((pz, pt, tp), edges)]
    inb = (ii[0] >= 0) & (ii[1] >= 0) & (ii[2] >= 0)
    out = np.ones(len(pz))
    out[inb] = w[ii[0][inb], ii[1][inb], ii[2][inb]]
    return out, ii, inb


def process(sb, table):
    edges, w, _ = table
    a = sb.arrays(BRANCHES)
    out = calculator.build({"type": "minerva_3dqelike"}).compute(sb)
    d = {k: np.asarray(v) for k, v in a.items()}
    d["w3d"], d["wmg"] = out[B3D], out[BMG]
    n = len(d["cvwgt"])
    p = a["true_mu_p"].astype(np.float64)
    dx, dy, dz = (a[f"true_mu_dir_{c}"].astype(np.float64) for c in "xyz")
    sumtp = np.zeros(n)
    for k in ("true_p_p", "true_p2_p"):
        pp = np.where(valid(a[k]), a[k].astype(np.float64), 0.0)
        sumtp += np.sqrt(pp ** 2 + M_P ** 2) - M_P
    pre = (a["true_isnc"] == 0) & (a["true_pdg"] == 14) & valid(a["true_mu_p"], a["true_mu_dir_z"])
    nomes = (a["true_npi"] == 0) & (a["true_npi0"] == 0)
    gveto = a["true_g_p"] > 0.010
    base = pre & nomes & ~gveto
    pz_det, pt_det = p * dz / PZ_SCALE, p * np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_s = pz_det / np.hypot(pz_det, pt_det)
    sig = base & (cos_s > COS20)
    w3_ref, ii, inb = lookup(edges, w, pz_det, pt_det, sumtp, sig)

    # p_z-marginalized, rebuilt here
    wnt, iint, inbnt = lookup(edges, w, pz_det, pt_det, sumtp, base)
    cv = a["cvwgt"].astype(np.float64)
    pop = np.zeros(w.shape)
    np.add.at(pop, (iint[0][inbnt], iint[1][inbnt], iint[2][inbnt]), cv[inbnt])
    den = pop.sum(0)
    w2 = np.where(den > 0, (pop * w).sum(0) / np.where(den > 0, den, 1), 1.0)
    in2 = base & (iint[1] >= 0) & (iint[2] >= 0)
    wmg_ref = np.ones(n)
    wmg_ref[in2] = w2[iint[1][in2], iint[2][in2]]
    # per-cell closure over the in-p_z-range no-theta signal, with the
    # CALCULATOR's marginalized weight
    y3 = np.zeros(w.shape[1:])
    ym = np.zeros(w.shape[1:])
    np.add.at(y3, (iint[1][inbnt], iint[2][inbnt]), (cv * wnt)[inbnt])
    np.add.at(ym, (iint[1][inbnt], iint[2][inbnt]), (cv * d["wmg"])[inbnt])

    # neutrino-frame muon kinematics
    lp = np.sqrt(sum(a[f"genie_prefsi_lep_{c}"].astype(np.float64) ** 2 for c in ("px", "py", "pz")))
    has_lep = valid(a["genie_prefsi_lep_px"], a["genie_prefsi_lep_pz"]) & (lp > 0)
    cos_nu = np.where(has_lep, a["genie_prefsi_lep_pz"] / np.where(has_lep, lp, 1), np.nan)
    pz_nu = p * cos_nu / PZ_SCALE
    pt_nu = p * np.sqrt(np.clip(1 - cos_nu ** 2, 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_snu = pz_nu / np.hypot(pz_nu, pt_nu)
    sig_nu = base & has_lep & (cos_snu > COS20)
    w3_nu, ii_nu, inb_nu = lookup(edges, w, pz_nu, pt_nu, sumtp, sig_nu)

    g = np.sqrt(sum(a[f"genie_prefsi_g_{c}"].astype(np.float64) ** 2 for c in ("px", "py", "pz")))
    has_g = valid(a["genie_prefsi_g_px"], a["genie_prefsi_g_pz"])
    pre_g = has_g & (g > 0.010)

    # where the photon-vetoed events would sit
    _, iig, inb_g = lookup(edges, w, pz_det, pt_det, sumtp, pre & nomes & (cos_s > COS20))

    for k in [k for k in d if k.startswith("genie_prefsi")]:
        del d[k]
    flat = lambda ii_: np.where(ii_[0] >= 0, (ii_[0] * 9 + ii_[1]) * 10 + ii_[2], -1)  # noqa: E731
    d.update(pz=pz_det, pt=pt_det, sumtp=sumtp, cos_det=dz, cos_nu=cos_nu,
             pz_nu=pz_nu, pt_nu=pt_nu, base=base, sig=sig, sig_nu=sig_nu, pre=pre,
             nomes=nomes, gveto=gveto, pre_g=pre_g, inb=inb, inb_nu=inb_nu,
             inb_g=inb_g, w3_ref=w3_ref, wmg_ref=wmg_ref, w3_nu=w3_nu,
             bin_det=np.where(inb, flat(ii), -1), bin_nu=np.where(inb_nu, flat(ii_nu), -1),
             ipz=ii[0], ipt=ii[1], itp=ii[2], lep_p=np.where(has_lep, lp, np.nan),
             cos_true_nu=np.where(has_lep, cos_nu, np.nan),
             mg_close=np.full(n, np.max(np.abs(y3 - ym) / np.maximum(np.abs(y3), 1e-12))),
             n_empty=np.full(n, int(np.sum(den <= 0))),
             )
    CELLS.append((y3, ym))
    return d


CELLS = []


# ---------------------------------------------------------------- plots

def fig_occupancy(d, det, edges):
    cv = d["cvwgt"]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    m = d["inb"]
    h = np.zeros((5, 9, 10))
    np.add.at(h, (d["ipz"][m], d["ipt"][m], d["itp"][m]), cv[m])
    vmax = max(h.max(), 1)
    for i in range(5):
        ax = axs.flat[i]
        im = ax.imshow(np.where(h[i] > 0, h[i], np.nan).T, origin="lower", aspect="auto",
                       norm=LogNorm(vmin=0.5, vmax=vmax), cmap="Blues")
        ax.set_xticks(range(9), [f"{e:.2f}" for e in edges[1][:-1]], fontsize=6, rotation=45)
        ax.set_yticks(range(10), [f"{e:.2f}" for e in edges[2][:-1]], fontsize=6)
        ax.set_xlabel("p_T low edge [GeV/c]", fontsize=8)
        ax.set_ylabel("SumT_p low edge [GeV]", fontsize=8)
        ax.grid(False)
        lo, hi = edges[0][i] * PZ_SCALE, edges[0][i + 1] * PZ_SCALE
        ax.set_title(f"scaled p_z bin {i}: true p_z {lo:.3f}-{hi:.3f} GeV/c, "
                     f"cv={h[i].sum():.0f}", fontsize=8, color=INK)
        fig.colorbar(im, ax=ax)
    ax = axs.flat[5]
    sig = d["sig"]
    tot = cv[sig].sum()
    under = [np.sum(cv[sig & (x < e[0])]) / tot for x, e in zip((d["pz"], d["pt"], d["sumtp"]), edges)]
    over = [np.sum(cv[sig & (x >= e[-1])]) / tot for x, e in zip((d["pz"], d["pt"], d["sumtp"]), edges)]
    xx = np.arange(3)
    ax.bar(xx - 0.2, under, 0.4, color=C[0], label="below first edge")
    ax.bar(xx + 0.2, over, 0.4, color=C[1], label="above last edge")
    ax.set_xticks(xx, ["p_z (scaled)", "p_T", "SumT_p"])
    ax.set_ylabel("fraction of 3D-signal cvwgt")
    ax.set_title(f"in all 3 ranges: {cv[d['inb']].sum() / tot:.3f}", fontsize=9, color=INK)
    ax.legend(fontsize=8)
    fig.suptitle(f"{det}: 3D signal occupancy (cvwgt)", color=INK)
    fig.tight_layout()
    return fig


def fig_closure(det, cells):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (y3, ym) in enumerate(cells):
        m = y3 > 0
        ax.scatter(np.flatnonzero(m.ravel()), (ym[m] / y3[m]).ravel() - 1, s=8,
                   color=C[i % len(C)], alpha=0.7)
    ax.set_xlabel("(p_T, SumT_p) cell index (ipt*10 + itp)")
    ax.set_ylabel("sum cv*w_marg / sum cv*w_3D - 1")
    ax.set_title(f"{det}: _pzmarg per-cell yield closure, one colour per file",
                 fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def fig_theta(d, det):
    cv = d["cvwgt"]
    m = d["pre"] & np.isfinite(d["cos_nu"])
    dc = (d["cos_det"] - d["cos_nu"])[m]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axs[0]
    lim = max(np.percentile(np.abs(dc), 99.5), 1e-4)
    ax.hist(np.clip(dc, -lim, lim), np.linspace(-lim, lim, 101), weights=cv[m],
            histtype="step", lw=1.8, color=common.DET_COLOR.get(det, C[0]))
    p1, p99 = np.percentile(dc, [1, 99])
    ax.set_xlabel("cos theta_mu (det z) - cos theta_mu (nu dir)")
    ax.set_title(f"numu CC: 1-99% [{p1:+.4f}, {p99:+.4f}]", fontsize=9, color=INK)
    ax = axs[1]
    for lab, x, y, col in (("p_z scaled", d["pz"], d["pz_nu"], C[0]),
                           ("p_T", d["pt"], d["pt_nu"], C[1])):
        mm = m & d["base"]
        r = (x - y)[mm]
        lim2 = max(np.percentile(np.abs(r), 99.5), 1e-4)
        ax.hist(np.clip(r, -lim2, lim2), np.linspace(-lim2, lim2, 101), weights=cv[mm],
                histtype="step", lw=1.8, color=col, label=lab)
    ax.set_xlabel("det frame - nu frame [GeV/c]")
    ax.set_title("QE-like candidates", fontsize=9, color=INK)
    ax.legend(fontsize=8)
    ax = axs[2]
    either = d["inb"] | d["inb_nu"]
    mig = either & (d["bin_det"] != d["bin_nu"])
    dw = (d["w3d"] - d["w3_nu"])[either]
    ax.hist(dw, np.linspace(-2, 2, 81), weights=cv[either], histtype="step", lw=1.8, color=C[2])
    ax.set_yscale("log")
    ax.set_xlabel("w_3D (det frame) - w_3D (nu frame)")
    ax.set_title(f"bin migrations: {mig.sum()} of {either.sum()} "
                 f"({100 * cv[mig].sum() / max(cv[either].sum(), 1e-9):.2f}% cvwgt)",
                 fontsize=9, color=INK)
    fig.suptitle(f"{det}: muon angle from detector z vs from the neutrino direction",
                 color=INK)
    fig.tight_layout()
    return fig, mig, either


def fig_photon(d, det):
    cand = d["pre"] & d["nomes"]
    gp = d["true_g_p"].astype(np.float64) * 1000.0
    bins = np.logspace(-1, 3, 81)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for lab, m, col in (("no pre-FSI photon > 10 MeV", cand & ~d["pre_g"], C[0]),
                        ("pre-FSI photon > 10 MeV", cand & d["pre_g"], C[1])):
        mm = m & (gp > 0)
        ax.hist(gp[mm], bins, weights=d["cvwgt"][mm], histtype="step", lw=1.8,
                color=col, label=f"{lab} (N={m.sum()})")
    ax.axvline(10, color=INK, ls="--", lw=1, label="veto 10 MeV")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true_g_p [MeV]")
    ax.set_ylabel("cvwgt")
    ax.set_title(f"{det}: numu CC 0pi candidates (signal minus photon veto)",
                 fontsize=10, color=INK)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_spectra(d, det):
    cv = d["cvwgt"]
    m = d["pre"]
    p = d["true_mu_p"].astype(np.float64)
    specs = [(p * d["true_mu_dir_z"], "muon p_z (det, unscaled) [GeV/c]", np.linspace(-0.5, 2.0, 51)),
             (d["pt"], "muon p_T [GeV/c]", np.linspace(0, 1.2, 49)),
             (d["sumtp"], "SumT_p [GeV]", np.linspace(0, 0.9, 46))]
    fig, axs = plt.subplots(2, 3, figsize=(15, 6.5), sharex="col",
                            gridspec_kw=dict(height_ratios=[2.5, 1]))
    for c, (x, xl, bins) in enumerate(specs):
        common.ratio_panel(axs[0, c], axs[1, c], x[m], bins, cv[m], d["w3d"][m],
                           [("3D", np.ones(m.sum(), bool), C[0])], xl)
        h0, _ = np.histogram(x[m], bins, weights=cv[m])
        h1, _ = np.histogram(x[m], bins, weights=(cv * d["wmg"])[m])
        axs[0, c].stairs(h1, bins, color=C[1], lw=1.5, label="weighted _pzmarg")
        axs[1, c].stairs(np.divide(h1, h0, out=np.full(len(h0), np.nan), where=h0 > 0),
                         bins, color=C[1], lw=1.5, label="pzmarg", baseline=None)
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=8)
    fig.suptitle(f"{det}: numu CC muon / proton spectra, nominal vs weighted", color=INK)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- main

def pc(m, cv, base):
    return f"{m.sum()} ({100 * cv[m].sum() / max(cv[base].sum(), 1e-12):.2f}% cvwgt)"


def run_det(out, det, files, table):
    CELLS.clear()
    d = common.load(files, process, table)
    cells = list(CELLS)
    cv = d["cvwgt"]
    out.text(f"== {det}: {len(files)} files, {len(cv)} slices")
    for b, w in ((B3D, d["w3d"]), (BMG, d["wmg"])):
        common.weight_summary(out, w, cv, b)
    numucc = d["pre"]
    out.text(f"  reweighted share of numu CC cvwgt: 3D "
             f"{100 * cv[numucc & (d['w3d'] != 1)].sum() / cv[numucc].sum():.1f}%, "
             f"pzmarg {100 * cv[numucc & (d['wmg'] != 1)].sum() / cv[numucc].sum():.1f}%")

    # events within float32 rounding of a bin edge may legitimately land in
    # the neighbouring bin (the calculator sums SumT_p in float32, this
    # script in float64): count them separately
    near = np.zeros(len(cv), bool)
    for x, e in zip((d["pz"], d["pt"], d["sumtp"]), table[0]):
        near |= np.min(np.abs(np.nan_to_num(x)[:, None] - e[None, :]), axis=1) < 1e-6
    for lab, wc, wr in (("3D", d["w3d"], d["w3_ref"]), ("pzmarg", d["wmg"], d["wmg_ref"])):
        bad = np.abs(wc - wr) > 1e-9
        out.check(not np.any(bad & ~near),
                  f"{lab}: calculator == independent lookup: {np.sum(bad & ~near)} "
                  f"mismatches away from edges, {np.sum(bad & near)} within 1e-6 of a "
                  f"bin edge (float32 vs float64), max |dw| {np.max(np.abs(wc - wr)):.1e}")
    out.check(d["mg_close"].max() < 1e-9, f"pzmarg: per-cell yield closure per file "
              f"(max rel {d['mg_close'].max():.1e})")
    out.text(f"    empty (p_T, SumT_p) cells per file: "
             f"{sorted(int(x) for x in set(d["n_empty"][np.r_[0, np.flatnonzero(np.diff(d["_file"])) + 1]]))} of 90")
    out.check(bool(np.all(d["w3d"][~d["sig"]] == 1.0)), "3D: w == 1 outside the signal")

    # -- domain leaks: anything but numu CC with w != 1
    leak = ~numucc & ((d["w3d"] != 1) | (d["wmg"] != 1))
    out.check(leak.sum() == 0, f"no weight outside numu CC ({leak.sum()} slices)")

    # -- signal audit
    inb = d["inb"]
    np3 = inb & (d["true_np"] > 2)
    out.text(f"  true_np > 2 in the 3D in-range signal: {pc(np3, cv, inb)} "
             "(SumT_p under-counted; documented)")
    cand = numucc & d["nomes"]
    gv = cand & d["gveto"]
    out.text(f"  photon veto on numu CC 0pi candidates: {pc(gv, cv, cand)}; without a "
             f"pre-FSI photon > 10 MeV (de-excitation/FSI origin): {pc(gv & ~d['pre_g'], cv, cand)}")
    gin = d["inb_g"] & d["gveto"] & ~d["pre_g"]
    out.text(f"    of those, would sit in a 3D bin: {pc(gin, cv, d['inb_g'])} of the "
             "in-bin no-photon-veto population")
    gp = d["true_g_p"]
    out.text(f"    candidates with 0 < true_g_p <= 10 MeV: {pc(cand & (gp > 0) & (gp <= 0.010), cv, cand)}")
    m1 = d["sig"] & valid(d["true_cpi_p"])
    m2 = d["sig"] & valid(d["true_pi0_p"])
    out.text(f"  'no mesons': 3D signal with npi==0 but true_cpi_p filled: {m1.sum()}, "
             f"npi0==0 but true_pi0_p filled: {m2.sum()} (of {d['sig'].sum()})")
    out.check(m1.sum() + m2.sum() <= 1e-3 * d["sig"].sum(),
              "meson leaks via the G4ID-filtered counts < 0.1% of signal", warn=True)
    ok = numucc & np.isfinite(d["lep_p"])
    dl = np.abs(d["lep_p"][ok] - d["true_mu_p"][ok])
    out.text(f"  |pre-FSI lepton| vs true_mu_p: 99% |dp| = {np.percentile(dl, 99):.1e} GeV, "
             f"max {dl.max():.1e}")

    # -- theta / frames
    ft, mig, either = fig_theta(d, det)
    m = numucc & np.isfinite(d["cos_nu"])
    dc = (d["cos_det"] - d["cos_nu"])[m]
    out.text(f"  cos theta(det z) - cos theta(nu): 1-99% [{np.percentile(dc, 1):+.4f}, "
             f"{np.percentile(dc, 99):+.4f}], median {np.median(dc):+.5f}")
    y_det = np.sum((cv * d["w3d"])[either])
    y_nu = np.sum((cv * d["w3_nu"])[either])
    out.text(f"  nu-frame (p_z, p_T): 3D bin migrations {pc(mig, cv, either)} of the "
             f"union in-bin; signal det {d['sig'].sum()} vs nu {d['sig_nu'].sum()}; "
             f"reweighted yield change (nu/det - 1) {y_nu / y_det - 1:+.4f}")
    out.check(cv[mig].sum() / max(cv[either].sum(), 1e-9) < 0.02,
              "detector-z vs neutrino-direction bin migration < 2% of cvwgt", warn=True)
    with np.errstate(invalid="ignore"):
        th_true = np.degrees(np.arccos(np.clip(d["cos_nu"], -1, 1)))
    out.text(f"  3D signal with unscaled theta(mu, nu) > 20 deg (NUISANCE cut, here "
             f"applied on scaled kinematics): {pc(inb & (th_true > 20), cv, inb)}")

    figs = {"domain": common.fig_domain(d, {"3D": d["w3d"], "pzmarg": d["wmg"]},
                                        det, "minerva_3dqelike"),
            "weights": common.fig_weight_hists({"3D": d["w3d"], "pzmarg": d["wmg"]},
                                               det, "minerva_3dqelike"),
            "occupancy": fig_occupancy(d, det, table[0]),
            "closure": fig_closure(det, cells),
            "theta_frame": ft,
            "photon_veto": fig_photon(d, det),
            "spectra": fig_spectra(d, det)}
    for k, f in figs.items():
        out.save(f, f"{k}_{det}")
    out.text("")


def main():
    args = common.parse_args(__doc__.splitlines()[0])
    out = common.Output(NAME, "minerva_3dqelike validation "
                              "(validation/minerva_3dqelike/validate_minerva.py)")
    table = load_table()
    for det, files in common.files_by_detector(args).items():
        run_det(out, det, files, table)
    return out.close()


if __name__ == "__main__":
    sys.exit(main())
