#!/usr/bin/env python3
"""Validation plots for the pi_fsi_ha2025 calculator (hA2018 -> hA2025).

The weight is w = frac_hA2025 / frac_hA2018 for the FSI fate (cex / abs /
inel / pipro) of the event's leading pre-FSI charged pion, at its pre-FSI
kinetic energy on Ar40 (fakedata/calculators/pi_fsi.py, fakedata/ha2025.py,
data/ha_pion_fsi_weights_A40.csv). On the CV MC sBruce files this script
runs the real calculator and checks / plots, per detector (cvwgt-weighted):

  weight_vs_ke      event weights vs pre-FSI KE on top of the table curve
                    (closure: every event on the curve, flat beyond the
                    [1, 999] MeV clamp)
  fsi_code_vs_ke    composition of genie_prefsi_cpi_fsi codes vs KE
  fate_closure      four-fate fractions vs KE: MC unweighted vs weighted,
                    over the table hA2018 / hA2025 fractions (weighted
                    counts are divided by the UNWEIGHTED four-fate total,
                    so they show the reweighted rate, not a renormalized
                    shape)
  weight_hist       weight distribution per fate
  knob_matrix       per raw genie_prefsi_cpi_fsi code (code 4 split by
                    surviving pi+- count), the fraction of events whose GENIE
                    FrCEx/FrAbs/FrInel/FrPiProd_pi +1 sigma multisigma
                    weight is > 1. A fate knob moves the events of its own
                    fate one way and (by renormalization) every other fate
                    the other way; in these files +1 sigma LOWERS the own
                    fate, so each fate shows up as the single ~0 cell in
                    its row and non-interacting pions as a row near 0
                    everywhere (weights == 1). An independent check, by
                    GENIE's own reweighter, of which code is which fate.

History: this check found that the calculator originally read code 3 as
elastic (w = 1) and split code 4 into abs / inel by true_npi; the codes are
GENIE v3 INukeFateHA_t (2 CEx, 3 Inelas, 4 Abs, 7 PiProd, no elastic fate),
and both halves of code 4 respond to FrAbs. Fixed in pi_fsi.py.

Hard checks (assert): calculator weights match an independent recomputation
(fate from the raw branches, np.interp on the CSV read here) to
< 1e-4 (KE from float32 vs float64 momenta); w == 1
exactly for events with no reweightable fate.

Writes validate-pionFSI/plots/*.png, plots/pion_fsi_validation.pdf and
validate-pionFSI/summary.txt.

Usage: ./venv/bin/python validate-pionFSI/validate_pion_fsi.py
           [--max-files N] [files...]
       (default: every CV MC file in the sbn-rewgted-20 sBruce directory)
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from fakedata.calculators.pi_fsi import M_PION, PiFSIhA2025  # noqa: E402
from fakedata.ha2025 import CSV_PATH, FATES                  # noqa: E402
from fakedata.sbruce import MODE_DIS, MODE_RES, SBruceFile, valid  # noqa: E402

DEFAULT_DIR = "/Users/gputnam/Work/osc/sbn-rewgted-20-sBruce/sbn-rewgted-20"
DEFAULT_GLOBS = ["SBNDMCCV_*_sbruce.root",
                 "ICARUSRun*_SpringMCOverlay_rewgt_*_sbruce.root"]
PLOT_DIR = os.path.join(HERE, "plots")
SUMMARY = os.path.join(HERE, "summary.txt")

FSI_NOINT = 1         # kIHAFtNoInteraction

BRANCHES = ["genie_prefsi_cpi_px", "genie_prefsi_cpi_py", "genie_prefsi_cpi_pz",
            "genie_prefsi_cpi_fsi", "true_npi", "true_npi0", "true_cpi_p",
            "genie_mode", "true_isnc", "nu_E", "cvwgt"]
KNOBS = ["FrCEx_pi", "FrAbs_pi", "FrInel_pi", "FrPiProd_pi"]   # FATES order
KNOB_FMT = "GENIEReWeight_SBN_v1_multisigma_{}"

# reference palette (dataviz skill, light mode): fixed slot per fate
FATE_COLOR = {"cex": "#2a78d6", "abs": "#eb6834", "inel": "#1baf7a",
              "pipro": "#eda100"}
FATE_LABEL = {"cex": "charge exchange", "abs": "absorption",
              "inel": "inelastic", "pipro": "pion production"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NOM_COLOR, RW_COLOR = "#52514e", "#2a78d6"
MODE_COLORS = {"all CC": "#0b0b0b", "RES": "#2a78d6", "DIS": "#eb6834"}

KE_BINS = np.arange(0.0, 1025.0, 25.0)

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 110,
})


def load_table():
    """The runtime CSV, read independently of fakedata.ha2025."""
    tab = {f: [] for f in FATES}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(x for x in f if not x.startswith("#")):
            tab[r["fate"]].append((float(r["KE"]), float(r["frac2018"]),
                                   float(r["frac2025"]), float(r["weight"])))
    return {f: np.array(sorted(v)).T for f, v in tab.items()}   # ke,f18,f25,w


def detector(path):
    return "SBND" if os.path.basename(path).startswith("SBND") else "ICARUS"


def process(path, table):
    sb = SBruceFile(path)
    n = sb.n_entries
    a = sb.arrays(BRANCHES)
    w = PiFSIhA2025().compute(sb)["fdwgt_pi_fsi_ha2025"]

    has = valid(a["genie_prefsi_cpi_px"], a["genie_prefsi_cpi_py"],
                a["genie_prefsi_cpi_pz"])
    p = np.sqrt(a["genie_prefsi_cpi_px"].astype(np.float64) ** 2
                + a["genie_prefsi_cpi_py"].astype(np.float64) ** 2
                + a["genie_prefsi_cpi_pz"].astype(np.float64) ** 2)
    ke = np.where(has, 1000.0 * (np.sqrt(p ** 2 + M_PION ** 2) - M_PION), np.nan)

    fsi = a["genie_prefsi_cpi_fsi"]
    npi = a["true_npi"]
    fate = np.full(n, -1)
    # GENIE v3 INukeFateHA_t: 2 CEx, 3 Inelas, 4 Abs, 7 PiProd
    fate[has & (fsi == 2)] = 0
    fate[has & (fsi == 4)] = 1
    fate[has & (fsi == 3)] = 2
    fate[has & (fsi == 7)] = 3

    # independent recomputation of the weight
    w_ref = np.ones(n)
    for i, f in enumerate(FATES):
        sel = fate == i
        kt, wt = table[f][0], table[f][3]
        w_ref[sel] = np.interp(np.clip(ke[sel], kt[0], kt[-1]), kt, wt)
    # the calculator forms KE from the float32 momenta, this from float64:
    # agreement to float32 precision (x the steepest table slope), not bit-exact
    dw = np.max(np.abs(w - w_ref))
    assert dw < 1e-4, f"{path}: calculator != reference, max |dw| = {dw}"
    assert np.all(w[fate < 0] == 1.0), f"{path}: w != 1 off-domain"

    # GENIE FSI knobs: does +1 sigma raise this event's weight?
    up = np.zeros((n, len(KNOBS)), dtype=bool)
    knob_ok = True
    for j, k in enumerate(KNOBS):
        name = KNOB_FMT.format(k)
        if not sb.has_multisigma(name):
            knob_ok = False
            continue
        ws = sb.multisigma_at_sigma(name, 1.0)
        up[:, j] = np.isfinite(ws) & (ws > 1.0 + 1e-6)

    sb.close()
    return dict(ke=ke, fsi=fsi, fate=fate, w=w, has=has, npi=npi,
                npi0=a["true_npi0"], cpi_p=a["true_cpi_p"],
                mode=a["genie_mode"], isnc=a["true_isnc"], nu_E=a["nu_E"],
                cvwgt=a["cvwgt"], up=up, dw=np.full(n, dw),
                knob_ok=np.full(n, knob_ok))


def concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ---------------------------------------------------------------- plots

def fig_weight_vs_ke(d, det, table, rng):
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for i, (f, ax) in enumerate(zip(FATES, axs.flat)):
        sel = np.flatnonzero(d["fate"] == i)
        kt, wt = table[f][0], table[f][3]
        # extend the curve flat past the clamp so the expectation is visible
        ax.plot(np.r_[0.0, kt, 1300.0], np.r_[wt[0], wt, wt[-1]],
                color=INK2, lw=2, label="table (clamped)", zorder=2)
        if len(sel):
            show = rng.choice(sel, min(len(sel), 20000), replace=False)
            ax.scatter(d["ke"][show], d["w"][show], s=4, color=FATE_COLOR[f],
                       alpha=0.5, lw=0, label="events", zorder=3)
            dev = np.max(np.abs(d["w"][sel] - np.interp(
                np.clip(d["ke"][sel], kt[0], kt[-1]), kt, wt)))
        else:
            dev = 0.0
        ax.axhline(1.0, color=GRID, lw=1, zorder=1)
        # the table has narrow spikes where frac2018 -> 0 (just above the
        # 1e-6 guard); cap the axis at the events and note the peak
        top = max(2.0, 1.3 * (d["w"][sel].max() if len(sel) else 1.0))
        if wt.max() > top:
            ax.set_ylim(0, top)
            kp = kt[np.argmax(wt)]
            ax.annotate(f"table peak w={wt.max():.0f}\nat KE={kp:.0f} MeV",
                        xy=(kp, top), xytext=(kp + 60, 0.8 * top),
                        fontsize=8, color=INK2,
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
        ax.set_title(f"{FATE_LABEL[f]}  (N={len(sel)}, max |w-curve|={dev:.1e})",
                     fontsize=10, color=INK)
        ax.set_ylabel("weight")
        ax.set_xlim(0, 1300)
        ax.legend(loc="best", fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("pre-FSI charged-pion KE [MeV]")
    fig.suptitle(f"{det}: event weights vs table curve", color=INK)
    fig.tight_layout()
    return fig


def fig_fsi_codes(d, det):
    cats = [("no interaction (code 1)", (d["fsi"] == FSI_NOINT), "#c3c2b7"),
            ("cex (code 2)", d["fate"] == 0, FATE_COLOR["cex"]),
            ("inel (code 3)", d["fate"] == 2, FATE_COLOR["inel"]),
            ("abs (code 4)", d["fate"] == 1, FATE_COLOR["abs"]),
            ("pipro (code 7)", d["fate"] == 3, FATE_COLOR["pipro"])]
    has = d["has"]
    other = has.copy()
    for _, m, _ in cats:
        other &= ~m
    cats.append(("other code", other, "#4a3aa7"))
    tot, _ = np.histogram(d["ke"][has], KE_BINS, weights=d["cvwgt"][has])
    tot = np.where(tot > 0, tot, np.nan)
    x = 0.5 * (KE_BINS[1:] + KE_BINS[:-1])
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    fr = []
    for name, m, c in cats:
        m = m & has
        h, _ = np.histogram(d["ke"][m], KE_BINS, weights=d["cvwgt"][m])
        fr.append(h / tot)
    ax.stackplot(x, np.nan_to_num(fr), colors=[c for *_, c in cats],
                 labels=[n for n, *_ in cats], edgecolor="white", lw=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of events with a pre-FSI pi+-")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    axn.step(KE_BINS[:-1], np.nan_to_num(tot), where="post", color=INK2, lw=1.5)
    axn.set_ylabel("events / 25 MeV")
    axn.set_xlabel("pre-FSI charged-pion KE [MeV]")
    axn.set_xlim(0, 1000)
    fig.suptitle(f"{det}: leading pre-FSI pi+- FSI code vs KE", color=INK)
    fig.tight_layout()
    return fig


def fig_fate_closure(d, det, table):
    in4 = d["fate"] >= 0
    tot, _ = np.histogram(d["ke"][in4], KE_BINS, weights=d["cvwgt"][in4])
    x = 0.5 * (KE_BINS[1:] + KE_BINS[:-1])
    ok = tot > 0
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for i, (f, ax) in enumerate(zip(FATES, axs.flat)):
        m = d["fate"] == i
        cw = d["cvwgt"][m]
        h0, _ = np.histogram(d["ke"][m], KE_BINS, weights=cw)
        h1, _ = np.histogram(d["ke"][m], KE_BINS, weights=cw * d["w"][m])
        e0, _ = np.histogram(d["ke"][m], KE_BINS, weights=cw ** 2)
        e1, _ = np.histogram(d["ke"][m], KE_BINS, weights=(cw * d["w"][m]) ** 2)
        kt, f18, f25 = table[f][0], table[f][1], table[f][2]
        ax.plot(kt, f18, color=INK2, ls="--", lw=1.5, label="table hA2018")
        ax.plot(kt, f25, color=INK, lw=1.5, label="table hA2025")
        ax.errorbar(x[ok], h0[ok] / tot[ok], np.sqrt(e0[ok]) / tot[ok],
                    fmt="o", mfc="white", mec=FATE_COLOR[f], ecolor=FATE_COLOR[f],
                    ms=5, lw=1, label="MC unweighted")
        ax.errorbar(x[ok], h1[ok] / tot[ok], np.sqrt(e1[ok]) / tot[ok],
                    fmt="o", color=FATE_COLOR[f], ms=5, lw=1,
                    label="MC weighted")
        ax.set_title(FATE_LABEL[f], fontsize=10, color=INK)
        ax.set_ylabel("fraction of 4-fate events")
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 1000)
    axs[0, 0].legend(fontsize=8, loc="best")
    for ax in axs[1]:
        ax.set_xlabel("pre-FSI charged-pion KE [MeV]")
    fig.suptitle(f"{det}: fate fractions vs KE (denominator: unweighted "
                 "cex+abs+inel+pipro)", color=INK)
    fig.tight_layout()
    return fig


def fig_weight_hist(d, det):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(3.0, np.max(d["w"]) * 1.02), 121)
    for i, f in enumerate(FATES):
        m = d["fate"] == i
        if np.any(m):
            ax.hist(d["w"][m], bins, histtype="step", lw=2, color=FATE_COLOR[f],
                    label=f"{FATE_LABEL[f]} (N={m.sum()})")
    n1 = np.count_nonzero(d["fate"] < 0)
    ax.set_yscale("log")
    ax.set_xlabel("pi_fsi_ha2025 weight")
    ax.set_ylabel("events")
    ax.set_title(f"{det}: weights per fate  ({n1} other events at w = 1 exactly,"
                 " not shown)", fontsize=10, color=INK)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_knob_matrix(d, det):
    if not np.all(d["knob_ok"]):
        return None
    has, fsi, npi = d["has"], d["fsi"], d["npi"]
    rows = [("code 1 (no int.)", has & (fsi == FSI_NOINT)),
            ("code 2 -> cex", d["fate"] == 0),
            ("code 3 -> inel", d["fate"] == 2),
            ("code 4, 0 pi+- -> abs", (d["fate"] == 1) & (npi == 0)),
            ("code 4, >=1 pi+- -> abs", (d["fate"] == 1) & (npi > 0)),
            ("code 7 -> pipro", d["fate"] == 3)]
    mat = np.zeros((len(rows), 4))
    nf = np.zeros(len(rows), dtype=int)
    for i, (_, m) in enumerate(rows):
        nf[i] = m.sum()
        if nf[i]:
            mat[i] = d["up"][m].mean(axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(rows)):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] > 0.6 else INK, fontsize=10)
    ax.set_xticks(range(4), [k.replace("_pi", "") for k in KNOBS])
    ax.set_yticks(range(len(rows)),
                  [f"{r} (N={nf[i]})" for i, (r, _) in enumerate(rows)])
    ax.set_xlabel("GENIE knob whose +1 sigma weight > 1")
    ax.set_ylabel("genie_prefsi_cpi_fsi -> pi_fsi_ha2025 fate")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="fraction of events")
    ax.set_title(f"{det}: FSI code vs GENIE fate knobs (own fate = the ~0 "
                 "cell in a row)", color=INK, fontsize=10)
    fig.tight_layout()
    return fig


def fig_observables(d, det):
    cc = (d["isnc"] == 0)
    modes = {"all CC": cc, "RES": cc & (d["mode"] == MODE_RES),
             "DIS": cc & (d["mode"] == MODE_DIS)}
    specs = [("cpi_p", "post-FSI leading pi+- momentum [GeV]",
              np.linspace(0, 1.5, 31), d["cpi_p"] > 0),
             ("npi", "true N pi+-", np.arange(-0.5, 5.5, 1.0), d["npi"] >= 0),
             ("npi0", "true N pi0", np.arange(-0.5, 5.5, 1.0), d["npi0"] >= 0),
             ("nu_E", "true E_nu [GeV]", np.linspace(0, 3, 31), d["nu_E"] > 0)]
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.5), sharex="col",
                            gridspec_kw=dict(height_ratios=[2.5, 1]))
    for c, (key, xl, bins, filled) in enumerate(specs):
        ax, axr = axs[0, c], axs[1, c]
        m = modes["all CC"] & filled
        cw = d["cvwgt"][m]
        h0, _ = np.histogram(d[key][m], bins, weights=cw)
        h1, _ = np.histogram(d[key][m], bins, weights=cw * d["w"][m])
        ax.stairs(h0, bins, color=NOM_COLOR, lw=2, label="nominal (hA2018)")
        ax.stairs(h1, bins, color=RW_COLOR, lw=2, label="weighted (hA2025)")
        ax.set_ylabel("CC events")
        for name, mm in modes.items():
            mm = mm & filled
            cw = d["cvwgt"][mm]
            g0, _ = np.histogram(d[key][mm], bins, weights=cw)
            g1, _ = np.histogram(d[key][mm], bins, weights=cw * d["w"][mm])
            r = np.divide(g1, g0, out=np.full(len(g0), np.nan), where=g0 > 0)
            axr.stairs(r, bins, color=MODE_COLORS[name], lw=1.5, label=name,
                       baseline=None)
        axr.axhline(1.0, color=INK2, lw=0.8, ls=":")
        axr.set_ylabel("weighted / nominal")
        axr.set_xlabel(xl)
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=8)
    fig.suptitle(f"{det}: effect on CC truth distributions", color=INK)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- summary

def summarize(d, det, nfiles):
    n = len(d["w"])
    cw = d["cvwgt"]
    lines = [f"== {det}: {nfiles} files, {n} events",
             f"  with pre-FSI pi+-:        {d['has'].sum():8d} "
             f"({100 * d['has'].mean():.1f}%)",
             f"  reweighted (4 fates):     {(d['fate'] >= 0).sum():8d} "
             f"({100 * (d['fate'] >= 0).mean():.1f}%)"]
    for i, f in enumerate(FATES):
        m = d["fate"] == i
        if m.any():
            lines.append(f"    {f:6s} N={m.sum():7d}  mean w={d['w'][m].mean():.4f}"
                         f"  min={d['w'][m].min():.4f}  max={d['w'][m].max():.4f}")
    cc = d["isnc"] == 0
    lines += [f"  mean w (all):             {d['w'].mean():.4f}",
              f"  cvwgt-weighted mean w:    {np.sum(cw * d['w']) / np.sum(cw):.4f}",
              f"  CC rate change:           "
              f"{np.sum((cw * d['w'])[cc]) / np.sum(cw[cc]) - 1:+.4f}",
              f"  w range:                  [{d['w'].min():.4f}, {d['w'].max():.4f}]",
              f"  checks: calculator vs independent recomputation max |dw| = "
              f"{d['dw'].max():.1e} (< 1e-4), w == 1 off-domain: PASS"]
    if not np.all(d["knob_ok"]):
        lines.append("  (GENIE Fr*_pi multisigma knobs missing in some files: "
                     "knob matrix skipped)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--max-files", type=int, default=None,
                    help="per detector")
    args = ap.parse_args()
    files = args.files or sorted(
        f for g in DEFAULT_GLOBS for f in glob.glob(os.path.join(DEFAULT_DIR, g)))
    by_det = {}
    for f in files:
        by_det.setdefault(detector(f), []).append(f)
    if args.max_files:
        by_det = {k: v[:args.max_files] for k, v in by_det.items()}

    table = load_table()
    rng = np.random.default_rng(1)
    os.makedirs(PLOT_DIR, exist_ok=True)
    summaries = []
    with PdfPages(os.path.join(PLOT_DIR, "pion_fsi_validation.pdf")) as pdf:
        for det in sorted(by_det, reverse=True):          # SBND first
            parts = []
            for f in by_det[det]:
                print(f"{os.path.basename(f)}")
                parts.append(process(f, table))
            d = concat(parts)
            figs = {"weight_vs_ke": fig_weight_vs_ke(d, det, table, rng),
                    "fsi_code_vs_ke": fig_fsi_codes(d, det),
                    "fate_closure": fig_fate_closure(d, det, table),
                    "weight_hist": fig_weight_hist(d, det),
                    "knob_matrix": fig_knob_matrix(d, det),
                    "observables": fig_observables(d, det)}
            for name, fig in figs.items():
                if fig is None:
                    continue
                fig.savefig(os.path.join(PLOT_DIR, f"{name}_{det}.png"))
                pdf.savefig(fig)
                plt.close(fig)
            s = summarize(d, det, len(by_det[det]))
            print(s)
            summaries.append(s)
    with open(SUMMARY, "w") as f:
        f.write("pi_fsi_ha2025 validation (validate-pionFSI/validate_pion_fsi.py)\n")
        f.write("\n\n".join(summaries) + "\n")
    print(f"wrote {PLOT_DIR}/ and {SUMMARY}")


if __name__ == "__main__":
    main()
