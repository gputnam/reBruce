#!/usr/bin/env python3
"""Validate the ccqe_xsec_corr calculator event-by-event on sBruce files.

For every file, the calculator's weights (CCQEXSecCorr.compute on the real
SBruceFile, before the driver's WEIGHT_CLIP) are compared in BOTH modes with
an independent reference: the nusystematics CCQEXSecCorr logic transcribed
to C++ and compiled by ROOT --

  - selection from the raw branches: CC (true_isnc == 0) QE
    (genie_mode == 0), true_pdg in {+-12, +-14}, filled Enu/q0/q3;
  - CCQEXSecCorr::GetEventResponse's E_nu window, lower_bound bracketing,
    5 MeV snapping and linear blend (verbatim);
  - MECq0q3ResponseCalc::GetCentralWeight with UseNearestBin + EdgeClamp
    (verbatim), i.e. ROOT's own TAxis::FindBin / TH2::GetBinContent on the
    TH2Ds of data/ccqe_xsec_corr_sbn2_maps.root read by ROOT itself.

  match_tool: true   against exactly that;
  default            against that plus reBruce's two departures, written
                     out again here: an event without its pre-FSI recoil
                     nucleon (p for nu, n for nu-bar: hyperon production
                     etc.) gets 1; a map covers (q3, q0) only inside its
                     axes and in a non-empty bin -- neither blended map
                     covers: 1, one does: its weight alone.

Floating-point contraction is switched off in the reference, as in a
gcc x86-64 build of nusystematics, so the two must agree EXACTLY.

Also checked per file:
  - weight == 1 exactly for every event outside the domain;
  - genie_q0 / genie_q3 equal E_nu - E_lep and |p_nu - p_lep| recomputed
    from genie_prefsi_lep_* with the flavor's charged-lepton mass (the
    tool's definition), to < 1e-5 GeV (float32 storage);
  - default mode: no CC QE event at weight 0;
  - summary: CC QE events, flavor mix, events set to 1 for no recoil
    nucleon (`hyp`) and for no map coverage (`unc`), tool-mode zeros,
    default-mode weight percentiles, events above the driver clip (10),
    cvwgt-weighted mean weight.

Needs PyROOT:  export PYTHONPATH=$(root-config --libdir)
Usage: ./venv/bin/python validation/validate_ccqe_xsec_corr.py [files...]
       (default: every *.root in the sbn-rewgted-20 sBruce directory)
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fakedata.calculator import WEIGHT_CLIP                  # noqa: E402
from fakedata.calculators.ccqe_corr import CCQEXSecCorr      # noqa: E402
from fakedata.ccqe_corr import ENERGY_GRID, FLAVORS, MAPS_PATH  # noqa: E402
from fakedata.sbruce import SBruceFile                       # noqa: E402

DEFAULT_DIR = "/Users/gputnam/Work/osc/sbn-rewgted-20-sBruce/sbn-rewgted-20"
LEPTON_MASS = {12: 0.51099895e-3, 14: 0.1056583755}
EXTRA = ["cvwgt", "genie_prefsi_lep_px", "genie_prefsi_lep_py",
         "genie_prefsi_lep_pz", "genie_prefsi_p_px", "genie_prefsi_n_px"]

REFERENCE_CPP = r"""
#include "TFile.h"
#include "TH2D.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

namespace ccqe_ref {
#pragma clang fp contract(off)

// MECq0q3ResponseCalc::GetCentralWeight, UseNearestBin = EdgeClamp = true
double central_weight(const TH2D* h, double q0, double q3,
                      double wmin, double wmax) {
  const double xMin = h->GetXaxis()->GetXmin();
  const double xMax = h->GetXaxis()->GetXmax();
  const double yMin = h->GetYaxis()->GetXmin();
  const double yMax = h->GetYaxis()->GetXmax();
  auto in_x = (q3 >= xMin && q3 <= xMax);
  auto in_y = (q0 >= yMin && q0 <= yMax);
  int ix = h->GetXaxis()->FindBin(q3);
  int iy = h->GetYaxis()->FindBin(q0);
  if (!in_x || !in_y) {
    ix = std::clamp(ix, 1, h->GetNbinsX());
    iy = std::clamp(iy, 1, h->GetNbinsY());
  }
  double w = h->GetBinContent(ix, iy);
  return std::max(wmin, std::min(wmax, w));
}

// reBruce coverage: inside both axes and a non-empty bin
bool covers(const TH2D* h, double q0, double q3) {
  const TAxis* xa = h->GetXaxis();
  const TAxis* ya = h->GetYaxis();
  if (q3 < xa->GetXmin() || q3 > xa->GetXmax()) return false;
  if (q0 < ya->GetXmin() || q0 > ya->GetXmax()) return false;
  return h->GetBinContent(xa->FindBin(q3), ya->FindBin(q0)) != 0.0;
}

struct Tool {
  Tool() {}
  Tool(const Tool&) = delete;
  Tool& operator=(const Tool&) = delete;
  std::vector<double> fEgrid;
  std::map<int, std::vector<std::unique_ptr<TH2D>>> fMaps;
  double fWmin = 0.0, fWmax = 30.0;
  double fEnuMin = 0.20, fEnuMax = 3.00, fEnuSnapTol = 5e-3;

  void add(int pdg, TH2D* h) {
    auto c = static_cast<TH2D*>(h->Clone());
    c->SetDirectory(nullptr);
    fMaps[pdg].emplace_back(c);
  }

  // CCQEXSecCorr::GetEventResponse, from the flavor lookup on
  double weight(int nu_pdg, double ev_Enu, double ev_q0, double ev_q3,
                bool coverage) {
    auto it = fMaps.find(nu_pdg);
    if (it == fMaps.end()) return 1.0;
    if (ev_Enu < fEnuMin - 1e-6 || ev_Enu > fEnuMax + 1e-6) return 1.0;
    const auto &calcs = it->second;
    auto it_hi = std::lower_bound(fEgrid.begin(), fEgrid.end(), ev_Enu);
    size_t ih = (it_hi == fEgrid.end()) ? fEgrid.size() - 1
                                        : std::distance(fEgrid.begin(), it_hi);
    size_t il = (ih == 0) ? 0 : ih - 1;
    const double Elo = fEgrid[il];
    const double Ehi = fEgrid[ih];
    double t = 0.0;
    if (std::fabs(ev_Enu - Elo) <= fEnuSnapTol) {
      ih = il; t = 0.0;
    } else if (std::fabs(ev_Enu - Ehi) <= fEnuSnapTol) {
      il = ih; t = 0.0;
    } else {
      t = (ih == il || Ehi <= Elo)
              ? 0.0
              : std::clamp((ev_Enu - Elo) / (Ehi - Elo), 0.0, 1.0);
    }
    const double w_lo = central_weight(calcs[il].get(), ev_q0, ev_q3, fWmin, fWmax);
    const double w_hi = central_weight(calcs[ih].get(), ev_q0, ev_q3, fWmin, fWmax);
    if (coverage) {
      const bool c_lo = covers(calcs[il].get(), ev_q0, ev_q3);
      const bool c_hi = covers(calcs[ih].get(), ev_q0, ev_q3);
      if (!c_lo && !c_hi) return 1.0;
      if (!c_hi) return std::clamp(w_lo, fWmin, fWmax);
      if (!c_lo) return std::clamp(w_hi, fWmin, fWmax);
    }
    double w_blend = (1.0 - t) * w_lo + t * w_hi;
    return std::clamp(w_blend, fWmin, fWmax);
  }

  // one call per file: in_domain decided by the caller's raw selection
  void weights(int n, const int* pdg, const double* enu, const double* q0,
               const double* q3, const bool* in_domain, bool coverage,
               double* out) {
    for (int i = 0; i < n; ++i)
      out[i] = in_domain[i] ? weight(pdg[i], enu[i], q0[i], q3[i], coverage)
                            : 1.0;
  }
};
}  // namespace ccqe_ref
"""


def build_reference():
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gInterpreter.Declare(REFERENCE_CPP)
    tool = ROOT.ccqe_ref.Tool()
    for e in ENERGY_GRID:
        tool.fEgrid.push_back(e)
    f = ROOT.TFile.Open(MAPS_PATH)
    for pdg, name in FLAVORS.items():
        for e in ENERGY_GRID:
            h = f.Get(f"{name}/E{e:.2f}")
            if not h:
                raise RuntimeError(f"ROOT cannot read {name}/E{e:.2f}")
            tool.add(pdg, h)
    f.Close()
    return tool


def reference_weights(tool, a, match_tool):
    """Reference weights with the selection made here from raw branches.
    Returns (weights, CC QE domain, events dropped for no recoil nucleon)."""
    pdg = np.ascontiguousarray(a["true_pdg"], dtype=np.int32)
    enu, q0, q3 = (np.ascontiguousarray(a[b], dtype=np.float64)
                   for b in ("genie_Enu", "genie_q0", "genie_q3"))
    dom = np.ascontiguousarray(
        (a["genie_mode"] == 0) & (a["true_isnc"] == 0)
        & np.isin(pdg, [12, -12, 14, -14])
        & (enu > -900) & (q0 > -900) & (q3 > -900))
    recoil = np.where(pdg > 0, a["genie_prefsi_p_px"] > -900,
                      a["genie_prefsi_n_px"] > -900)
    no_recoil = dom & ~recoil
    use = dom if match_tool else np.ascontiguousarray(dom & recoil)
    out = np.empty(len(pdg), dtype=np.float64)
    tool.weights(len(pdg), pdg, enu, q0, q3, use, not match_tool, out)
    return out, dom, no_recoil


def check_q(a, dom):
    """Max |genie_q0/q3 - recomputed from the pre-FSI lepton| over dom."""
    pdg = np.abs(a["true_pdg"][dom])
    m = np.where(pdg == 12, LEPTON_MASS[12], LEPTON_MASS[14])
    px, py, pz = (a[f"genie_prefsi_lep_{c}"][dom].astype(np.float64)
                  for c in ("px", "py", "pz"))
    enu = a["genie_Enu"][dom].astype(np.float64)
    ok = (px > -900) & (py > -900) & (pz > -900)
    q0 = enu - np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m ** 2)
    q3 = np.sqrt(px ** 2 + py ** 2 + (enu - pz) ** 2)
    d0 = np.abs(q0 - a["genie_q0"][dom])[ok]
    d3 = np.abs(q3 - a["genie_q3"][dom])[ok]
    return (max(d0.max(initial=0), d3.max(initial=0)), int((~ok).sum()))


def main(paths):
    calc = CCQEXSecCorr()
    calc_tool = CCQEXSecCorr(match_tool=True)
    tool = build_reference()
    n_fail = 0
    tot = dict(skipped=0, ev=0, dom=0, hyp=0, unc=0, zero_tool=0, zero=0,
               over=0, mism=0, mism_tool=0)
    print(f"{'file':46s} {'events':>7} {'CCQE':>6} {'nue':>4} {'numub':>5} "
          f"{'mism':>4} {'mtool':>5} {'dq[GeV]':>8} {'hyp':>4} {'unc':>4} "
          f"{'0tool':>5} {'w=0':>4} {'>10':>4} "
          f"{'p1':>6} {'p50':>6} {'p99':>6} {'<w>cv':>6}  status")
    for p in paths:
        with SBruceFile(p) as sb:
            missing = [b for b in calc.branches_needed() + EXTRA
                       if not sb.has_branch(b)]
            if missing:
                print(f"{os.path.basename(p):46s} {sb.n_entries:>7}  skipped: "
                      f"no {', '.join(missing[:3])}"
                      f"{' ...' if len(missing) > 3 else ''} (no GENIE truth)")
                tot["skipped"] += 1
                continue
            a = sb.arrays(calc.branches_needed() + EXTRA)
            w = calc.compute(sb)[calc.branch]
            wt = calc_tool.compute(sb)[calc_tool.branch]
        ref, dom, no_recoil = reference_weights(tool, a, match_tool=False)
        ref_t, _, _ = reference_weights(tool, a, match_tool=True)
        mism = int(np.sum(w != ref))
        mism_t = int(np.sum(wt != ref_t))
        outside_ok = bool(np.all(w[~dom] == 1.0) and np.all(wt[~dom] == 1.0))
        dq, no_lep = check_q(a, dom) if dom.any() else (0.0, 0)
        wd = w[dom]
        # set to 1 for lack of coverage: in the domain with a recoil nucleon
        # and 1 here, but not 1 in the tool
        unc = dom & ~no_recoil & (w == 1.0) & (wt != 1.0)
        ok = (mism == 0 and mism_t == 0 and outside_ok and dq < 1e-5
              and np.all(np.isfinite(w)) and not np.any(wd == 0))
        n_fail += not ok
        pdg = a["true_pdg"][dom]
        cv = a["cvwgt"][dom].astype(np.float64)
        pct = (np.percentile(wd, [1, 50, 99]) if dom.any()
               else [np.nan] * 3)
        mean = np.average(wd, weights=cv) if dom.any() and cv.sum() else np.nan
        print(f"{os.path.basename(p):46s} {len(w):>7} {dom.sum():>6} "
              f"{np.sum(np.abs(pdg) == 12):>4} {np.sum(pdg == -14):>5} "
              f"{mism:>4} {mism_t:>5} {dq:8.1e} {no_recoil.sum():>4} "
              f"{unc.sum():>4} {np.sum(wt[dom] == 0):>5} {np.sum(wd == 0):>4} "
              f"{np.sum(w > WEIGHT_CLIP[1]):>4} "
              f"{pct[0]:6.3f} {pct[1]:6.3f} {pct[2]:6.3f} {mean:6.3f}  "
              + ("OK" if ok else "FAIL")
              + ("" if outside_ok else " (weight != 1 outside domain)")
              + (f" ({no_lep} without lepton)" if no_lep else ""))
        tot["ev"] += len(w)
        tot["dom"] += int(dom.sum())
        tot["hyp"] += int(no_recoil.sum())
        tot["unc"] += int(unc.sum())
        tot["zero_tool"] += int(np.sum(wt[dom] == 0))
        tot["zero"] += int(np.sum(wd == 0))
        tot["over"] += int(np.sum(w > WEIGHT_CLIP[1]))
        tot["mism"] += mism
        tot["mism_tool"] += mism_t
    print(f"\n{len(paths) - tot['skipped']} files checked "
          f"({tot['skipped']} skipped), {tot['ev']} events, {tot['dom']} CC QE "
          f"(nu_e/nu_mu).\n"
          f"  default:    {tot['mism']} mismatches vs the C++ reference; "
          f"{tot['hyp']} set to 1 for no recoil nucleon, {tot['unc']} for no "
          f"map coverage; {tot['zero']} at 0; {tot['over']} above the driver "
          f"clip {WEIGHT_CLIP[1]}\n"
          f"  match_tool: {tot['mism_tool']} mismatches vs the C++ reference; "
          f"{tot['zero_tool']} at 0\n"
          f"{n_fail} failures")
    return 1 if n_fail or tot["skipped"] == len(paths) else 0


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(glob.glob(os.path.join(DEFAULT_DIR, "*.root")))
    sys.exit(main(args))
