"""Shared helpers for the validation/<name>/ scripts.

File discovery (the sbn-rewgted-20 CV MC sBruce files, split by detector),
the plot style, a per-validation output directory under validation/plots/
(gitignored), a Report collecting PASS / WARN / FAIL lines (a physics check
never aborts the script: a failed check is a finding to report), and the
"domain matrix" every calculator validation draws.

Facts about the sBruce inputs these scripts rely on are established, with
plots, by validation/inputs/validate_inputs.py.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fakedata.calculator import WEIGHT_CLIP                  # noqa: E402
from fakedata.sbruce import SBruceFile                       # noqa: E402

DEFAULT_DIR = "/Users/gputnam/Work/osc/sbn-rewgted-20-sBruce/sbn-rewgted-20"
DEFAULT_GLOBS = ["SBNDMCCV_*_sbruce.root",
                 "ICARUSRun*_SpringMCOverlay_rewgt_*_sbruce.root"]
PLOT_ROOT = os.path.join(HERE, "plots")

# reference palette (dataviz skill, light mode)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NOM_COLOR, RW_COLOR = "#52514e", "#2a78d6"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#c3c2b7"]
DET_COLOR = {"SBND": "#2a78d6", "ICARUS": "#eb6834"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 110,
})

# genie_mode is the CAF / simb enum (NOT GENIE's ScatteringType_t)
MODE_NAMES = {0: "QE", 1: "RES", 2: "DIS", 3: "COH", 10: "MEC"}


def detector(path):
    return "SBND" if os.path.basename(path).startswith("SBND") else "ICARUS"


def parse_args(description=None, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("files", nargs="*",
                    help="sBruce files (default: every CV MC file in "
                         "the sbn-rewgted-20 directory)")
    ap.add_argument("--max-files", type=int, default=None,
                    help="per detector")
    if extra:
        extra(ap)
    return ap.parse_args()


def files_by_detector(args):
    files = args.files or sorted(
        f for g in DEFAULT_GLOBS for f in glob.glob(os.path.join(DEFAULT_DIR, g)))
    by_det = {}
    for f in files:
        by_det.setdefault(detector(f), []).append(f)
    if args.max_files:
        by_det = {k: v[:args.max_files] for k, v in by_det.items()}
    # SBND first
    return {k: by_det[k] for k in sorted(by_det, reverse=True)}


def load(files, func, *fargs):
    """Run func(SBruceFile, *fargs) -> dict of per-event arrays on every file
    and concatenate. Scalars are broadcast per file under '_file'."""
    parts = []
    for i, f in enumerate(files):
        print(f"  {os.path.basename(f)}", flush=True)
        with SBruceFile(f) as sb:
            d = func(sb, *fargs)
        n = len(next(iter(d.values())))
        d["_file"] = np.full(n, i)
        parts.append(d)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def knob(sb, name, sigma):
    """Raw multisigmaTree weight at the stored sigma LABEL (no reorientation;
    see validation/inputs for which knobs have reversed labels)."""
    return sb.multisigma_at_sigma(name, float(sigma))


class Output:
    """validation/plots/<name>/: PNGs, one PDF of all figures, summary.txt."""

    def __init__(self, name, title):
        self.name = name
        self.dir = os.path.join(PLOT_ROOT, name)
        os.makedirs(self.dir, exist_ok=True)
        self.pdf = PdfPages(os.path.join(self.dir, f"{name}.pdf"))
        self.lines = [title, ""]
        self.n_fail = 0
        self.n_warn = 0

    def save(self, fig, stem):
        if fig is None:
            return
        fig.savefig(os.path.join(self.dir, f"{stem}.png"))
        self.pdf.savefig(fig)
        plt.close(fig)

    def text(self, s=""):
        print(s, flush=True)
        self.lines.append(s)

    def check(self, ok, what, warn=False):
        tag = "PASS" if ok else ("WARN" if warn else "FAIL")
        if not ok:
            if warn:
                self.n_warn += 1
            else:
                self.n_fail += 1
        self.text(f"  [{tag}] {what}")
        return ok

    def close(self):
        self.pdf.close()
        self.text("")
        self.text(f"{self.n_fail} FAIL, {self.n_warn} WARN")
        with open(os.path.join(self.dir, "summary.txt"), "w") as f:
            f.write("\n".join(self.lines) + "\n")
        print(f"wrote {self.dir}/")
        return 1 if self.n_fail else 0


# ----------------------------------------------------------------- categories

def categories(d):
    """Mutually exclusive (label, mask) event categories for domain matrices.

    Needs d['true_pdg'], d['true_isnc'], d['genie_mode']. Cosmic / unmatched
    slices are true_pdg == -1 (their true_isnc is 0, not a sentinel)."""
    pdg, isnc, mode = d["true_pdg"], d["true_isnc"], d["genie_mode"]
    cos = pdg == -1
    cats = [("cosmic/unmatched", cos)]
    flav = {"numu": 14, "numubar": -14, "nue": 12, "nuebar": -12}
    for cc, ccm in (("CC", isnc == 0), ("NC", isnc == 1)):
        for fl, p in flav.items():
            for m, mn in MODE_NAMES.items():
                sel = (~cos) & ccm & (pdg == p) & (mode == m)
                if sel.any():
                    cats.append((f"{fl} {cc} {mn}", sel))
    return cats


def fig_domain(d, weights, det, title):
    """Fraction of events (and of cvwgt) with w != 1 per category, one column
    per weight branch. weights: {label: array}."""
    cats = categories(d)
    names = list(weights)
    mat = np.zeros((len(cats), len(names)))
    for i, (_, m) in enumerate(cats):
        for j, k in enumerate(names):
            if m.any():
                mat[i, j] = np.mean(weights[k][m] != 1.0)
    fig, ax = plt.subplots(figsize=(2.2 + 1.3 * len(names), 0.32 * len(cats) + 1.6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(cats)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if mat[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(cats)),
                  [f"{c} (N={m.sum()})" for c, m in cats], fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="fraction with w != 1")
    ax.set_title(f"{det}: {title} -- domain", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def fig_weight_hists(weights, det, title, cvwgt=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    allw = np.concatenate([w[w != 1.0] for w in weights.values()] + [[1.0]])
    hi = max(2.0, min(np.max(allw) * 1.02, WEIGHT_CLIP[1] * 1.02))
    bins = np.linspace(0, hi, 121)
    for i, (k, w) in enumerate(weights.items()):
        m = w != 1.0
        ax.hist(w[m], bins, histtype="step", lw=1.8, color=C[i % len(C)],
                label=f"{k} (N={m.sum()} of {len(w)} != 1)")
    ax.set_yscale("log")
    ax.set_xlabel("weight (events with w != 1)")
    ax.set_ylabel("events")
    ax.set_title(f"{det}: {title} -- weights", fontsize=10, color=INK)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def weight_summary(out, w, cv, label):
    m = w != 1.0
    out.text(f"  {label}: N(w!=1)={m.sum()} ({100 * m.mean():.1f}%), "
             f"cvwgt-weighted mean w={np.sum(cv * w) / np.sum(cv):.4f}, "
             f"range [{w.min():.4f}, {w.max():.4f}], "
             f"at clip {np.sum((w <= WEIGHT_CLIP[0]) | (w >= WEIGHT_CLIP[1]))}")
    out.check(bool(np.all(np.isfinite(w)) and np.all(w >= 0)),
              f"{label}: weights finite and >= 0")


def ratio_panel(ax, axr, x, bins, cv, w, masks, xlabel, ylabel="events"):
    """Nominal vs weighted histogram on ax, weighted/nominal for each
    (label, mask, color) on axr."""
    h0, _ = np.histogram(x, bins, weights=cv)
    h1, _ = np.histogram(x, bins, weights=cv * w)
    ax.stairs(h0, bins, color=NOM_COLOR, lw=2, label="nominal")
    ax.stairs(h1, bins, color=RW_COLOR, lw=2, label="weighted")
    ax.set_ylabel(ylabel)
    for lab, m, col in masks:
        g0, _ = np.histogram(x[m], bins, weights=cv[m])
        g1, _ = np.histogram(x[m], bins, weights=(cv * w)[m])
        r = np.divide(g1, g0, out=np.full(len(g0), np.nan), where=g0 > 0)
        axr.stairs(r, bins, color=col, lw=1.5, label=lab, baseline=None)
    axr.axhline(1.0, color=INK2, lw=0.8, ls=":")
    axr.set_ylabel("weighted / nominal")
    axr.set_xlabel(xlabel)
