"""BNB flux weight maps binned in the neutrino's flux ancestry.

A flux variation changes the population of decaying parent hadrons, so its
weight is looked up from the parent: its PDG code and its momentum at the
decay point (|p| and the angle theta to the beam axis), plus -- for the
hadron-production maps -- the neutrino flavour and energy. Two map files
(data/, provenance in data/README.md):

  flux_horn_171p5kA_weight_maps.root   horn current 174 -> 171.5 kA
      <parent>/weight2              TH2D  x = |p| [GeV/c], y = theta [rad]
  flux_hadprod_sw_piplus_u0387.root    Sanford-Wang pi+ production universe 387
      <flavor>/<parent>/weight3     TH3D  x = E_nu [GeV], y = |p|, z = theta
      <flavor>/<parent>/weight      TH2D  the same without theta
      <flavor>/weight_E             TH1D  E_nu only

Every lookup clamps the bin index into the histogram. The horn maps were
written by uproot and have ZERO under/overflow bins, so an out-of-range
parent must take the edge bin; the hadron-production maps store the edge
values in their flow bins, so clamping reproduces them exactly. One rule
serves both.

theta = atan2(hypot(px, py), pz) stays defined for a parent decaying at rest
(theta = 0; the horn maps isolate those in a [0, 1e-6) GeV/c momentum bin).
"""

import os

import numpy as np
import uproot

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HORN_PATH = os.path.join(DATA_DIR, "flux_horn_171p5kA_weight_maps.root")
HADPROD_PATH = os.path.join(DATA_DIR, "flux_hadprod_sw_piplus_u0387.root")

# parent pdg -> directory name in both map files; any other parent: weight 1
PARENTS = {211: "piplus", -211: "piminus", 321: "kplus", -321: "kminus",
           130: "k0l", -13: "muplus", 13: "muminus"}
# neutrino pdg -> directory name in the hadron-production file
FLAVORS = {14: "numu", -14: "numubar", 12: "nue", -12: "nuebar"}

HADPROD_MAPS = ("weight3", "weight", "weight_E")


class BinnedMap:
    """One histogram's inner bins, with a clamped vectorized lookup."""

    def __init__(self, values, edges):
        self.values = np.asarray(values, dtype=np.float64)
        self.edges = [np.asarray(e, dtype=np.float64) for e in edges]
        assert self.values.shape == tuple(len(e) - 1 for e in self.edges)

    @classmethod
    def from_root(cls, f, key):
        h = f[key]
        return cls(h.values(flow=False), [ax.edges() for ax in h.axes])

    def bin_indices(self, *coords):
        """Per-axis bin index, clamped into [0, nbins-1] (ROOT FindFixBin
        convention: a bin is [lo, hi))."""
        assert len(coords) == len(self.edges)
        return tuple(
            np.clip(np.searchsorted(e, np.asarray(x, dtype=np.float64),
                                    side="right") - 1, 0, len(e) - 2)
            for e, x in zip(self.edges, coords))

    def __call__(self, *coords):
        return self.values[self.bin_indices(*coords)]


def parent_p_theta(px, py, pz):
    """|p| and theta = atan2(pT, pz) of the parent momentum at decay
    (in double precision, as the source C++ does, from float32 branches)."""
    px, py, pz = (np.asarray(x, dtype=np.float64) for x in (px, py, pz))
    pt2 = px * px + py * py
    return np.sqrt(pt2 + pz * pz), np.arctan2(np.sqrt(pt2), pz)


class HornCurrentMaps:
    """Horn-current weight maps: w(parent, |p|, theta), flavour-independent."""

    def __init__(self, path=None):
        path = path or HORN_PATH
        with uproot.open(path) as f:
            if "zbins" in f and f["zbins"].axes[0].edges().size > 2:
                raise ValueError(f"{path}: 3D (v_z-slab) horn maps are not "
                                 "supported; expected a single slab")
            self.maps = {pdg: BinnedMap.from_root(f, f"{name}/weight2")
                         for pdg, name in PARENTS.items()}
            self.provenance = str(f["provenance"])

    def weight(self, parent_pdg, px, py, pz):
        """(N,) weights; 1 for a parent with no map."""
        parent_pdg = np.asarray(parent_pdg)
        out = np.ones(len(parent_pdg), dtype=np.float64)
        p, theta = parent_p_theta(px, py, pz)
        for pdg, m in self.maps.items():
            sel = parent_pdg == pdg
            if np.any(sel):
                out[sel] = m(p[sel], theta[sel])
        return out


class HadronProductionMaps:
    """Sanford-Wang hadron-production universe maps: w(flavor, parent,
    E_nu, |p|, theta) for map='weight3' (default), w(flavor, parent, E_nu,
    |p|) for 'weight', or w(flavor, E_nu) for 'weight_E'."""

    def __init__(self, path=None, map="weight3"):
        if map not in HADPROD_MAPS:
            raise ValueError(f"unknown hadron-production map '{map}'; "
                             f"choose from {HADPROD_MAPS}")
        self.map = map
        path = path or HADPROD_PATH
        with uproot.open(path) as f:
            if map == "weight_E":
                self.maps = {nu: BinnedMap.from_root(f, f"{fl}/weight_E")
                             for nu, fl in FLAVORS.items()}
            else:
                self.maps = {(nu, par): BinnedMap.from_root(f, f"{fl}/{pn}/{map}")
                             for nu, fl in FLAVORS.items()
                             for par, pn in PARENTS.items()}
            self.provenance = str(f["README"])

    def weight(self, nu_pdg, parent_pdg, E, px, py, pz):
        """(N,) weights; 1 for a flavour or parent with no map."""
        nu_pdg = np.asarray(nu_pdg)
        parent_pdg = np.asarray(parent_pdg)
        E = np.asarray(E, dtype=np.float64)
        out = np.ones(len(nu_pdg), dtype=np.float64)
        p, theta = parent_p_theta(px, py, pz)
        for key, m in self.maps.items():
            if self.map == "weight_E":
                # no parent axis: applies to every parent, as in the source
                # sw_caf_weight.h WeightEnergy()
                sel = nu_pdg == key
                if np.any(sel):
                    out[sel] = m(E[sel])
                continue
            sel = (nu_pdg == key[0]) & (parent_pdg == key[1])
            if not np.any(sel):
                continue
            if self.map == "weight3":
                out[sel] = m(E[sel], p[sel], theta[sel])
            else:
                out[sel] = m(E[sel], p[sel])
        return out
