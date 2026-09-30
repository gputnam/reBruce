"""CCQE cross-section correction maps (GENIE v3_04_02a -> v3_06_02, "sbn2").

A python port of the nusystematics CCQEXSecCorr tool

    src/nusystematics/systproviders/CCQEXSecCorr_tool.cc
    src/nusystematics/responsecalculators/MECq0q3ResponseCalc.hh
    fcl/CCQEXSecCorr.ToolConfig.fcl                     (the settings)

with its 4 flavors x 57 energies of (q3, q0) weight maps vendored in
data/ccqe_xsec_corr_sbn2_maps.root (built by
scripts/build_ccqe_xsec_corr_maps.py; provenance in data/README.md):

    <flavor>/E<energy:.2f>   TH2D  x = q3 [GeV], y = q0 [GeV], 500 x 500 on
                                   [0, 3.5]^2; zero under/overflow bins

Per event (flavor, E_nu, q0, q3), exactly as the tool with its fcl settings:
  1. E_nu outside [enu_min, enu_max] (+- 1e-6): weight 1.
  2. Bracket E_nu on the energy grid with lower_bound; snap to a grid point
     within snap_tol, else blend the two maps linearly in E_nu.
  3. Each map is read nearest-bin (ROOT TAxis::FindBin); a (q3, q0) outside
     the axis range takes the edge bin (EdgeClamp). The bin content is
     clamped to w_limits, and so is the blended weight.

Coverage (coverage=True, the default). Bins the source MC never populated
hold 0, and the tool reads them as weight 0 (or half the weight, if only one
of the two blended maps is empty); a (q3, q0) off the axes takes the edge
bin. Both are outside the maps' phase-space coverage, so by default a map
COVERS an event only when (q3, q0) is inside its axes and in a non-empty
bin:
  - both blended maps cover it:  the tool's blend;
  - only one does:               that map's weight alone;
  - neither does:                weight 1.
(A populated source bin with an empty target bin would also read 0; the
maps do not tell the two apart, so such a bin counts as uncovered too.)
coverage=False reproduces the tool exactly, zeros and edge bins included --
down to a coordinate exactly at an axis maximum, which ROOT's FindBin puts
in the (zero) overflow bin while the tool counts it as in range.

Maps are read lazily, one at a time, for the grid points some event needs:
all 228 held at once in float64 would take ~460 MB.
"""

import os

import numpy as np
import uproot

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MAPS_PATH = os.path.join(DATA_DIR, "ccqe_xsec_corr_sbn2_maps.root")

# neutrino pdg -> directory name in the map file
FLAVORS = {12: "nue", -12: "nuebar", 14: "numu", -14: "numubar"}

# fcl/CCQEXSecCorr.ToolConfig.fcl
ENERGY_GRID = tuple(round(0.20 + 0.05 * i, 2) for i in range(57))
ENU_MIN, ENU_MAX = 0.20, 3.00
ENU_SNAP_TOL = 5e-3
W_LIMITS = (0.0, 30.0)
ENU_GUARD = 1e-6  # hard-coded in CCQEXSecCorr::GetEventResponse


class Axis:
    """A fixed-width ROOT TAxis: (nbins, xmin, xmax)."""

    def __init__(self, nbins, xmin, xmax):
        self.nbins, self.xmin, self.xmax = int(nbins), float(xmin), float(xmax)

    def find_bin(self, x):
        """ROOT TAxis::FindBin (fixed bins): 0 underflow, nbins+1 overflow."""
        x = np.asarray(x, dtype=np.float64)
        inner = 1 + (self.nbins * (x - self.xmin)
                     / (self.xmax - self.xmin)).astype(np.int64)
        return np.where(x < self.xmin, 0,
                        np.where(x < self.xmax, inner, self.nbins + 1))

    def contains(self, x):
        return (x >= self.xmin) & (x <= self.xmax)


class WeightMap:
    """One h_weights_map: bin contents INCLUDING flow bins, indexed
    [ix, iy] with ix = q3 bin, iy = q0 bin (ROOT numbering)."""

    def __init__(self, values, xaxis, yaxis):
        self.values = np.asarray(values, dtype=np.float64)
        self.xaxis, self.yaxis = xaxis, yaxis
        assert self.values.shape == (xaxis.nbins + 2, yaxis.nbins + 2)

    @classmethod
    def from_root(cls, h):
        axes = [Axis(a.member("fNbins"), a.member("fXmin"), a.member("fXmax"))
                for a in (h.member("fXaxis"), h.member("fYaxis"))]
        return cls(h.values(flow=True), *axes)

    def central_weight(self, q0, q3, w_limits=W_LIMITS):
        """MECq0q3ResponseCalc::GetCentralWeight, UseNearestBin + EdgeClamp."""
        return self.lookup(q0, q3, w_limits)[0]

    def lookup(self, q0, q3, w_limits=W_LIMITS):
        """(central weight, covered): covered = (q3, q0) inside the axes and
        in a non-empty bin."""
        ix = self.xaxis.find_bin(q3)
        iy = self.yaxis.find_bin(q0)
        inside = self.xaxis.contains(q3) & self.yaxis.contains(q0)
        ix = np.where(inside, ix, np.clip(ix, 1, self.xaxis.nbins))
        iy = np.where(inside, iy, np.clip(iy, 1, self.yaxis.nbins))
        raw = self.values[ix, iy]
        return np.clip(raw, *w_limits), inside & (raw != 0)


class CCQEXSecCorrMaps:
    """The per-flavor, per-energy map set and the tool's E_nu interpolation.

    `loader(pdg, i)` returns the WeightMap for flavor pdg at ENERGY_GRID
    index i; by default it reads the vendored ROOT file. Tests pass their own.
    coverage: see the module docstring (False = the tool exactly).
    """

    def __init__(self, path=None, loader=None, energy_grid=ENERGY_GRID,
                 enu_min=ENU_MIN, enu_max=ENU_MAX, snap_tol=ENU_SNAP_TOL,
                 w_limits=W_LIMITS, coverage=True):
        self.path = path or MAPS_PATH
        self.coverage = bool(coverage)
        self.grid = np.asarray(energy_grid, dtype=np.float64)
        self.enu_min, self.enu_max = float(enu_min), float(enu_max)
        self.snap_tol = float(snap_tol)
        self.w_limits = (min(w_limits), max(w_limits))
        self._loader = loader
        if loader is None:
            with uproot.open(self.path) as f:
                self.provenance = str(f["provenance"])
                missing = [f"{fl}/E{e:.2f}" for fl in FLAVORS.values()
                           for e in self.grid if f"{fl}/E{e:.2f}" not in f]
            if missing:
                raise ValueError(f"{self.path}: missing maps {missing[:5]}"
                                 f"{' ...' if len(missing) > 5 else ''}")

    def _brackets(self, enu):
        """(il, ih, t) per event: CCQEXSecCorr::GetEventResponse's lower_bound
        bracketing, grid snapping and blend fraction."""
        g = self.grid
        ih = np.searchsorted(g, enu, side="left")
        ih = np.where(ih == len(g), len(g) - 1, ih)
        il = np.maximum(ih - 1, 0)
        elo, ehi = g[il], g[ih]
        snap_lo = np.abs(enu - elo) <= self.snap_tol
        snap_hi = ~snap_lo & (np.abs(enu - ehi) <= self.snap_tol)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where((ih == il) | (ehi <= elo), 0.0,
                         np.clip((enu - elo) / (ehi - elo), 0.0, 1.0))
        t = np.where(snap_lo | snap_hi, 0.0, t)
        ih = np.where(snap_lo, il, ih)
        il = np.where(snap_hi, ih, il)
        return il, ih, t

    def _map(self, f, pdg, i):
        if self._loader is not None:
            return self._loader(pdg, i)
        return WeightMap.from_root(f[f"{FLAVORS[pdg]}/E{self.grid[i]:.2f}"])

    def weight(self, nu_pdg, enu, q0, q3):
        """(N,) weights; 1 for a flavor with no maps or E_nu off the grid."""
        return self.weight_and_coverage(nu_pdg, enu, q0, q3)[0]

    def weight_and_coverage(self, nu_pdg, enu, q0, q3):
        """((N,) weights, (N,) uncovered). uncovered: the event is in a
        flavor / E_nu window with maps, but neither of its maps covers it
        (weight 1; with coverage=False it keeps the tool's weight)."""
        nu_pdg = np.asarray(nu_pdg)
        enu, q0, q3 = (np.asarray(x, dtype=np.float64) for x in (enu, q0, q3))
        out = np.ones(len(enu), dtype=np.float64)
        uncovered = np.zeros(len(enu), dtype=bool)
        sel = (np.isin(nu_pdg, list(FLAVORS))
               & (enu >= self.enu_min - ENU_GUARD)
               & (enu <= self.enu_max + ENU_GUARD))
        if not np.any(sel):
            return out, uncovered
        idx = np.flatnonzero(sel)
        il, ih, t = self._brackets(enu[idx])
        w_lo = np.empty(len(idx))
        w_hi = np.empty(len(idx))
        c_lo = np.empty(len(idx), dtype=bool)
        c_hi = np.empty(len(idx), dtype=bool)
        f = uproot.open(self.path) if self._loader is None else None
        try:
            for pdg in np.unique(nu_pdg[idx]):
                is_pdg = nu_pdg[idx] == pdg
                for i in np.unique(np.concatenate([il[is_pdg], ih[is_pdg]])):
                    m = self._map(f, int(pdg), int(i))
                    for w, c, ib in ((w_lo, c_lo, il), (w_hi, c_hi, ih)):
                        s = is_pdg & (ib == i)
                        if np.any(s):
                            w[s], c[s] = m.lookup(q0[idx][s], q3[idx][s],
                                                  self.w_limits)
        finally:
            if f is not None:
                f.close()
        if self.coverage:
            # one uncovered map: the other alone; neither: weight 1 (below)
            t = np.where(c_lo & ~c_hi, 0.0, np.where(c_hi & ~c_lo, 1.0, t))
        w = np.clip((1.0 - t) * w_lo + t * w_hi, *self.w_limits)
        cov = c_lo | c_hi
        out[idx] = np.where(cov, w, 1.0) if self.coverage else w
        uncovered[idx] = ~cov
        return out, uncovered
