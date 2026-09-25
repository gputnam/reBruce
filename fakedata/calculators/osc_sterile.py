"""3+1 sterile-neutrino nu_mu disappearance at IceCube-preferred points.

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
