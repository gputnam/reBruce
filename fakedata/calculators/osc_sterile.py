"""3+1 sterile-neutrino nu_mu disappearance at IceCube-preferred points.

Two calculators: osc_sterile_ic2024 (below: the survival probability per
event at a fixed 110 m) and osc_sterile_ic2024_unfolded (the SBND
reco-energy offset unfolded to a true-energy weight; see the comment block
above SterileIceCube2024Unfolded).

    w = 1 - sin^2(2 theta_24) * sin^2(1.267 * dm2_41[eV^2] * L[km] / E[GeV])

applied to every event whose true neutrino is nu_mu or nu_mu-bar, CC AND NC.
With theta_14 = theta_34 = 0 (IceCube's assumption), sin^2(2 theta_mumu) =
sin^2(2 theta_24) and every nu_mu that disappears goes to nu_s, so NC events
are depleted by the same factor as CC. nu_e / nu_e-bar (no U_e4 in this
model), slices without a truth neutrino (true_pdg == -1) and events without
a filled true_E get exactly 1.0.

L is FIXED at the SBND baseline, 110 m, for SBND and ICARUS files alike --
the stored `baseline` branch is deliberately not read. E is the true
neutrino energy (true_E, identical to genie_Enu on schema-20 files). The
two-flavour short-baseline formula is used; there is no averaging over the
decay-pipe length.

Source
------
IceCube Collaboration, "A search for an eV-scale sterile neutrino using
improved high-energy nu_mu event reconstruction in IceCube",
Phys. Rev. Lett. 133, 201804 (2024), arXiv:2405.08070 (companion PRD
arXiv:2405.08077): 10.7 years of 0.5-100 TeV up-going nu_mu, best fit
sin^2(2 theta_24) = 0.16, dm2_41 = 3.5 eV^2 (p = 3.1% vs no sterile), and a
CLOSED 95% CL (Wilks) contour.

No data release carries the contour, so it was digitized from the vector
paths of the paper's Fig. 1 (`results.pdf` in the arXiv source; the 95% CL
contour is the blue path with dash pattern [11.1 4.8]). Axis calibration
from the tick marks, in the figure's local frame: sin^2(2 theta_24) =
0.01 / 0.1 / 1 at x = 169.07 / 301.92 / 434.77 pt, dm2 = 0.1 / 1 / 10 eV^2
at y = 144.50 / 230.83 / 317.15 pt (log axes). The best-fit star lands at
(0.159, ~3.6 eV^2), reproducing the quoted best fit. The closed 95% region
spans dm2 = 2.07 - 12.1 eV^2 and sin^2(2 theta_24) = 0.052 - 0.348.

Points (the highest mixing allowed at each dm2; at the two dm2 extremes the
contour's tip is the only allowed mixing):

  label     dm2_41 [eV^2]   sin^2(2 theta_24)
  dm2lo      2.07            0.116     95% CL contour, minimum dm2
  bestfit    3.5             0.16      IceCube best fit
  dm2hi     12.1             0.187     95% CL contour, maximum dm2
"""

import numpy as np

from ..calculator import Calculator, register
from ..sbruce import valid

BRANCH = "fdwgt_osc_ic2024"

# label -> (dm2_41 [eV^2], sin^2(2 theta_24)); see the module docstring
POINTS = {
    "dm2lo": (2.07, 0.116),
    "bestfit": (3.5, 0.16),
    "dm2hi": (12.1, 0.187),
}

BASELINE_KM = 0.110     # SBND, used for every file regardless of detector
OSC_CONST = 1.267       # dm2[eV^2] L[km] / E[GeV] -> phase [rad]


def numu_survival(e_gev, dm2, s22, l_km=BASELINE_KM):
    """Two-flavour P(nu_mu -> nu_mu) at energy e_gev [GeV]."""
    return 1.0 - s22 * np.sin(OSC_CONST * dm2 * l_km / e_gev) ** 2


@register("osc_sterile_ic2024")
class SterileIceCube2024(Calculator):
    def __init__(self, branch=BRANCH, points=None, baseline_km=BASELINE_KM,
                 flavors=(14, -14), pdg_branch="true_pdg",
                 energy_branch="true_E"):
        self.branch = branch
        if points is None:
            points = POINTS
        elif not isinstance(points, dict):          # list of labels
            points = {p: POINTS[p] for p in points}
        self.points = {k: (float(v[0]), float(v[1])) for k, v in points.items()}
        self.baseline_km = float(baseline_km)
        self.flavors = [int(f) for f in flavors]
        self.pdg_branch = pdg_branch
        self.energy_branch = energy_branch

    def branches_needed(self):
        return [self.pdg_branch, self.energy_branch]

    def compute(self, sbruce):
        a = sbruce.arrays(self.branches_needed())
        n = sbruce.n_entries
        e = np.asarray(a[self.energy_branch], dtype=np.float64)
        mask = (np.isin(a[self.pdg_branch], self.flavors) & valid(e)
                & np.isfinite(e) & (e > 0))
        out = {}
        for label, (dm2, s22) in self.points.items():
            w = self.ones(n)
            w[mask] = numu_survival(e[mask], dm2, s22, self.baseline_km)
            name = f"{self.branch}_{label}"
            self.report_coverage(name, mask, n)
            out[name] = w
        return out


# ---------------------------------------------------------------------------
# osc_sterile_ic2024_unfolded: the SBND reco-energy oscillation offset,
# unfolded to a true-energy weight.
#
# For each point the 3+1 oscillation is applied to the SBND CV nu_mu +
# nu_mu-bar (CC and NC) sample at each event's own `baseline` and true_E, and
# histogrammed in reco energy (nu_E_calo, the GUMP signal-box variable; GUMP
# bins plus open under/overflow). The oscillated reco spectrum is unfolded to
# true energy by D'Agostini iteration (fakedata.unfold; prior = MC truth,
# UNFOLD_N_ITER iterations) on the cvwgt-weighted 2D migration matrix
# (nu_E_calo x true_E) of the same events, from the full SBND CV set. The
# weight in each true-energy bin is unfolded / null-truth: the true-energy-only
# distortion that reproduces the SBND reco signature of the oscillation. With
# no oscillation the prior is a fixed point, so the weight is exactly 1.
#
# The spectra need the whole SBND production while reweight.py sees one file
# at a time, so the table is built offline by
# scripts/build_osc_ic2024_unfolded.py and hard-coded below (--write
# regenerates it, --check verifies it). At run time only true_pdg and true_E
# are read: the weight is a piecewise-constant function of true E (last bin
# open-ended) applied to nu_mu / nu_mu-bar in SBND and ICARUS files alike;
# everything else gets exactly 1.0.
# ---------------------------------------------------------------------------

UNFOLDED_BRANCH = "fdwgt_osc_ic2024_unf"

# BEGIN UNFOLD TABLE -- generated by scripts/build_osc_ic2024_unfolded.py;
# do not edit by hand (regenerate with --write).
#   input:  13 files SBNDMCCV_*_sbruce.root
#           in /Users/gputnam/Work/osc/sbn-rewgted-20-sBruce/sbn-rewgted-20
#   events: 188245 entries, 180457 nu_mu/nu_mu-bar used, sum(cvwgt) = 188908.2
#   reco:   nu_E_calo, edges [0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.25, 1.5, np.inf] GeV
#   D'Agostini iterations: 50
UNFOLD_N_ITER = 50
UNFOLD_TRUE_EDGES = np.array([0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.25, 1.5, 2, 3, np.inf])
UNFOLD_WEIGHTS = {
    "dm2lo": np.array([0.928979, 0.939084, 0.955219, 0.962935, 0.969181, 0.974405, 0.978488, 0.981570, 0.983826, 0.985893, 0.987760, 0.988839, 0.989642, 0.991027, 0.993917, 0.995954, 0.987128, 0.981986, 0.981891]),
    "bestfit": np.array([0.856707, 0.853537, 0.869822, 0.886585, 0.900583, 0.914256, 0.925821, 0.935139, 0.942087, 0.949007, 0.954997, 0.958809, 0.961660, 0.966361, 0.976182, 0.984115, 0.958540, 0.943568, 0.943481]),
    "dm2hi": np.array([0.906961, 0.868519, 0.888844, 0.953843, 0.973377, 0.968517, 0.942629, 0.912701, 0.887803, 0.863836, 0.846332, 0.834497, 0.827383, 0.822344, 0.823033, 0.848567, 0.872745, 0.887747, 0.900155]),
}
# END UNFOLD TABLE


def unfolded_weight(e_gev, label):
    """Tabulated true-energy weight for one point (e_gev > 0)."""
    idx = np.searchsorted(UNFOLD_TRUE_EDGES, e_gev, side="right") - 1
    idx = np.clip(idx, 0, len(UNFOLD_TRUE_EDGES) - 2)
    return UNFOLD_WEIGHTS[label][idx]


@register("osc_sterile_ic2024_unfolded")
class SterileIceCube2024Unfolded(Calculator):
    def __init__(self, branch=UNFOLDED_BRANCH, points=None,
                 flavors=(14, -14), pdg_branch="true_pdg",
                 energy_branch="true_E"):
        self.branch = branch
        if points is None:
            points = list(UNFOLD_WEIGHTS)
        unknown = [p for p in points if p not in UNFOLD_WEIGHTS]
        if unknown:
            raise KeyError(f"no unfolded table for points {unknown}; "
                           f"known: {sorted(UNFOLD_WEIGHTS)}")
        self.points = list(points)
        self.flavors = [int(f) for f in flavors]
        self.pdg_branch = pdg_branch
        self.energy_branch = energy_branch

    def branches_needed(self):
        return [self.pdg_branch, self.energy_branch]

    def compute(self, sbruce):
        a = sbruce.arrays(self.branches_needed())
        n = sbruce.n_entries
        e = np.asarray(a[self.energy_branch], dtype=np.float64)
        mask = (np.isin(a[self.pdg_branch], self.flavors) & valid(e)
                & np.isfinite(e) & (e > 0))
        out = {}
        for label in self.points:
            w = self.ones(n)
            w[mask] = unfolded_weight(e[mask], label)
            name = f"{self.branch}_{label}"
            self.report_coverage(name, mask, n)
            out[name] = w
        return out
