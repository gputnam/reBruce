"""BNB flux variations: horn focusing and hadron production.

Both weights are looked up in maps binned in the neutrino's flux ancestry
(see fakedata/fluxmap.py and data/README.md for the map files):

  flux_horn_current       horn current 174 kA (CV) -> 171.5 kA
      w = weight2[parent](|p_parent|, theta_parent)
      Flavour-independent by construction, but the maps are validated for
      nu_mu only (per-bin closure 0.1% on SBND/ICARUS nu_mu, ~1% rms on
      nu_mu-bar; nu_e / nu_e-bar fail on statistics, their parents are ~0.3%
      of the decays in the flux files). Default: applied to nu_mu and
      nu_mu-bar, weight 1 for nu_e / nu_e-bar (`flavors` option).
      The raw maps reach ~100 in low-statistics mu+- cells; the driver's
      WEIGHT_CLIP bounds those.

  flux_hadron_production  Sanford-Wang pi+ production, universe 387
      w = <flavor>/<parent>/weight3(E_nu, |p_parent|, theta_parent)
      (`map` option: 'weight' drops theta, 'weight_E' keeps E_nu only)
      A single throw of the pi+ production uncertainty, not a +1 sigma
      shift: the nu_mu / pi+ weights average ~0.8, so this is a large
      normalization change as well as a shape change.

Parents with a map: pi+-, K+-, K0L, mu+-. Any other parent, a neutrino
without truth (cosmic / unmatched slice), and a flavour with no map get
weight 1.

Variables needed for a meaningful weight
----------------------------------------
Per matched true neutrino, all of it flux ancestry GENIE copies from the
flux entry into the CAF (rec.mc.nu.*, or rec.slc.truth.* for a slice):

  CAF field                     sBruce branch (default option value)   used by
  rec.mc.nu.parent_pdg          true_parent_pdg        (int)           both
  rec.mc.nu.parent_dcy_mom.x    true_parent_dcy_mom_x  [GeV/c]         both
  rec.mc.nu.parent_dcy_mom.y    true_parent_dcy_mom_y  [GeV/c]         both
  rec.mc.nu.parent_dcy_mom.z    true_parent_dcy_mom_z  [GeV/c]         both
  rec.mc.nu.initpdg (or .pdg)   true_pdg               (int, -1 none)  both
  rec.mc.nu.E                   true_E                 [GeV]           hadprod

parent_dcy_mom is the parent's momentum AT THE DECAY POINT in beam
coordinates. true_pdg stands in for initpdg (identical without
oscillations in the MC). Nothing detector-dependent is read, so the same
maps serve SBND and ICARUS.

sBruce schema 20 does NOT carry the four parent branches (see
MISSING_INFO.md); every branch name above is a constructor option so a
future schema can be mapped in the config. On a schema-20 file these
calculators only run under `reweight.py --test-shim`, which fills the parent
branches with defaults (fakedata/shim.py) -- that output is a plumbing
test, not a flux variation.
"""

import numpy as np

from ..calculator import Calculator, register
from ..fluxmap import HadronProductionMaps, HornCurrentMaps
from ..sbruce import valid

DEFAULT_BRANCHES = {
    "pdg_branch": "true_pdg",
    "energy_branch": "true_E",
    "parent_pdg_branch": "true_parent_pdg",
    "parent_px_branch": "true_parent_dcy_mom_x",
    "parent_py_branch": "true_parent_dcy_mom_y",
    "parent_pz_branch": "true_parent_dcy_mom_z",
}


class _FluxCalculator(Calculator):
    """Shared input handling: branch-name options and the truth mask."""

    uses_energy = True

    def __init__(self, branch, **branch_names):
        unknown = set(branch_names) - set(DEFAULT_BRANCHES)
        if unknown:
            raise TypeError(f"unknown option(s) {sorted(unknown)}")
        self.branch = branch
        self.names = {**DEFAULT_BRANCHES, **branch_names}

    def branches_needed(self):
        keys = ["pdg_branch", "energy_branch", "parent_pdg_branch",
                "parent_px_branch", "parent_py_branch", "parent_pz_branch"]
        if not self.uses_energy:
            keys.remove("energy_branch")
        return [self.names[k] for k in keys]

    def _load(self, sbruce):
        """(arrays keyed by option name, mask of events with filled truth)."""
        a = sbruce.arrays(self.branches_needed())
        a = {k: a[b] for k, b in self.names.items() if b in a}
        floats = [a["parent_px_branch"], a["parent_py_branch"],
                  a["parent_pz_branch"]]
        if self.uses_energy:
            floats.append(a["energy_branch"])
        mask = valid(*floats)
        for x in floats:
            mask &= np.isfinite(x)
        return a, mask


@register("flux_horn_current")
class FluxHornCurrent(_FluxCalculator):
    uses_energy = False

    def __init__(self, branch="fdwgt_flux_horn_171p5kA", table=None,
                 flavors=(14, -14), **branch_names):
        super().__init__(branch, **branch_names)
        self.flavors = [int(f) for f in flavors]
        self.maps = HornCurrentMaps(table)

    def compute(self, sbruce):
        a, mask = self._load(sbruce)
        n = sbruce.n_entries
        mask &= np.isin(a["pdg_branch"], self.flavors)
        weights = self.ones(n)
        weights[mask] = self.maps.weight(
            a["parent_pdg_branch"][mask], a["parent_px_branch"][mask],
            a["parent_py_branch"][mask], a["parent_pz_branch"][mask])
        self.report_coverage(self.branch, mask & (weights != 1.0), n)
        return {self.branch: weights}


@register("flux_hadron_production")
class FluxHadronProduction(_FluxCalculator):
    def __init__(self, branch="fdwgt_flux_hadprod_sw_piplus_u387",
                 table=None, map="weight3", **branch_names):
        super().__init__(branch, **branch_names)
        self.maps = HadronProductionMaps(table, map=map)

    def compute(self, sbruce):
        a, mask = self._load(sbruce)
        n = sbruce.n_entries
        weights = self.ones(n)
        weights[mask] = self.maps.weight(
            a["pdg_branch"][mask], a["parent_pdg_branch"][mask],
            a["energy_branch"][mask], a["parent_px_branch"][mask],
            a["parent_py_branch"][mask], a["parent_pz_branch"][mask])
        self.report_coverage(self.branch, mask & (weights != 1.0), n)
        return {self.branch: weights}
