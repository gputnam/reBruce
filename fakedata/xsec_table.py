"""Loading of the per-bin xsec-measurement weight tables (data/*.csv).

Each CSV row carries a measurement bin (edges), sigma_data, sigma_ar23, and
weight = sigma_data/sigma_ar23. Conventions (inherited from
fake-data-studies): weight = 1.0 outside the measured range and in bins with
no AR23 MC; no weight caps.

Tables:
  Table1D:      contiguous 1D bins for one observable
  Table2DSlice: the CC1mu1p DeltaAlphaT-in-DeltaPT-slices double differential
                (x = DeltaPT slice, y = DeltaAlphaT bins)
  TableBins2D:  irregular non-overlapping 2D bins (T2K NC1pi (p, costh))
  TableGrid3D:  regular 3D grid (MINERvA QE-like (p_z, p_T, SumT_p))
"""

import csv
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _read_csv(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(x for x in f if not x.startswith("#"))]
    return rows


class Table1D:
    def __init__(self, edges, weights):
        self.edges = np.asarray(edges, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        assert len(self.edges) == len(self.weights) + 1

    @property
    def n_bins(self):
        return len(self.weights)

    def bin_index(self, x):
        """Measured-bin index for each x; -1 if outside the measured range."""
        idx = np.digitize(x, self.edges) - 1
        idx = np.where((x >= self.edges[0]) & (x < self.edges[-1]), idx, -1)
        return np.clip(idx, -1, self.n_bins - 1)

    def weight(self, x):
        idx = self.bin_index(x)
        return np.where(idx >= 0, self.weights[np.maximum(idx, 0)], 1.0)


class TableBins2D:
    """Irregular, non-overlapping rectangular 2D bins."""

    def __init__(self, bins, weights):
        # bins: list of (x_lo, x_hi, y_lo, y_hi)
        self.bins = np.asarray(bins, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)

    @property
    def n_bins(self):
        return len(self.weights)

    def bin_index(self, x, y):
        idx = np.full(len(x), -1, dtype=np.int64)
        for i, (xlo, xhi, ylo, yhi) in enumerate(self.bins):
            sel = (x >= xlo) & (x < xhi) & (y >= ylo) & (y < yhi)
            idx[sel] = i
        return idx

    def weight(self, x, y):
        idx = self.bin_index(x, y)
        return np.where(idx >= 0, self.weights[np.maximum(idx, 0)], 1.0)


class TableGrid3D:
    """Regular (separable) 3D grid of bins.

    Used for the MINERvA triple-differential QE-like measurement, whose
    (p_z, p_T, SumT_p) binning is a full outer product of three edge arrays
    -- so the bin lookup is one digitize per axis rather than the linear
    scan TableBins2D needs for irregular bins.

    weights is stored flattened in C order (x slowest, z fastest), matching
    the (ipz, ipt, itp) row order of the source CSV, so callers can index
    `table.weights[bin_index(...)]` directly.
    """

    def __init__(self, x_edges, y_edges, z_edges, weights, excluded=None):
        self.x_edges = np.asarray(x_edges, dtype=np.float64)
        self.y_edges = np.asarray(y_edges, dtype=np.float64)
        self.z_edges = np.asarray(z_edges, dtype=np.float64)
        self.shape = (len(self.x_edges) - 1, len(self.y_edges) - 1,
                      len(self.z_edges) - 1)
        self.weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        assert self.weights.size == self.shape[0] * self.shape[1] * self.shape[2]
        # bins the measurement does not constrain (unmeasured / no MC); they
        # carry weight 1.0 by convention
        self.excluded = (np.zeros(self.weights.size, dtype=bool) if excluded is None
                         else np.asarray(excluded, dtype=bool).reshape(-1))
        assert self.excluded.size == self.weights.size

    @property
    def n_bins(self):
        return self.weights.size

    @staticmethod
    def _axis_index(v, edges):
        """Bin index along one axis; -1 outside [edges[0], edges[-1])."""
        idx = np.digitize(v, edges) - 1
        return np.where((v >= edges[0]) & (v < edges[-1]), idx, -1)

    def axis_indices(self, x, y, z):
        """Per-axis bin indices, each -1 outside that axis' range."""
        return (self._axis_index(x, self.x_edges),
                self._axis_index(y, self.y_edges),
                self._axis_index(z, self.z_edges))

    def bin_index(self, x, y, z):
        """Flat bin index for each (x, y, z); -1 if outside on any axis."""
        ix = self._axis_index(x, self.x_edges)
        iy = self._axis_index(y, self.y_edges)
        iz = self._axis_index(z, self.z_edges)
        inside = (ix >= 0) & (iy >= 0) & (iz >= 0)
        flat = (np.maximum(ix, 0) * self.shape[1]
                + np.maximum(iy, 0)) * self.shape[2] + np.maximum(iz, 0)
        return np.where(inside, flat, -1)

    def weight(self, x, y, z):
        idx = self.bin_index(x, y, z)
        return np.where(idx >= 0, self.weights[np.maximum(idx, 0)], 1.0)


def load_ub_table(csv_name, obs):
    """Load one observable from a MicroBooNE-format CSV.

    Handles both the cc1p0pi format (obs,ix,iy,x_lo,x_hi,y_lo,y_hi,...) and
    the cc2p0pi / ccpi format (obs,bin,x_lo,x_hi,...). The cc1p0pi 2D block
    "DeltaAlphaT_DeltaPT" is returned as a TableBins2D with
    x = DeltaPT, y = DeltaAlphaT.
    """
    rows = [r for r in _read_csv(os.path.join(DATA_DIR, csv_name))
            if r["obs"] == obs]
    if not rows:
        raise KeyError(f"observable {obs} not in {csv_name}")

    if "iy" in rows[0] and any(int(r["iy"]) >= 0 for r in rows):
        # 2D slice block: x_lo/x_hi = DeltaPT slice, y_lo/y_hi = DeltaAlphaT
        bins = [(float(r["x_lo"]), float(r["x_hi"]),
                 float(r["y_lo"]), float(r["y_hi"])) for r in rows]
        weights = [float(r["weight"]) for r in rows]
        return TableBins2D(bins, weights)

    rows = sorted(rows, key=lambda r: float(r["x_lo"]))
    edges = [float(r["x_lo"]) for r in rows] + [float(rows[-1]["x_hi"])]
    for i, r in enumerate(rows[1:], 1):
        assert abs(float(r["x_lo"]) - float(rows[i - 1]["x_hi"])) < 1e-9, \
            f"non-contiguous bins for {obs} in {csv_name}"
    return Table1D(edges, [float(r["weight"]) for r in rows])


def ub_observables(csv_name):
    """List the observables available in a MicroBooNE-format CSV."""
    seen = []
    for r in _read_csv(os.path.join(DATA_DIR, csv_name)):
        if r["obs"] not in seen:
            seen.append(r["obs"])
    return seen


def load_t2k_table():
    """T2K NC1pi (p_pi, cos_theta_pi) bins."""
    rows = _read_csv(os.path.join(DATA_DIR, "t2k_nc1pi_xsec.csv"))
    bins = [(float(r["p_lo"]), float(r["p_hi"]),
             float(r["costh_lo"]), float(r["costh_hi"])) for r in rows]
    weights = [float(r["weight"]) for r in rows]
    return TableBins2D(bins, weights)


def load_minerva_3dqelike_table():
    """MINERvA LE/ME 3D QE-like (p_z, p_T, SumT_p) BNB-extrapolated weights.

    See data/minerva_3dqelike_bnb.csv (built by
    scripts/build_minerva_3dqelike_table.py) for provenance.
    """
    rows = _read_csv(os.path.join(DATA_DIR, "minerva_3dqelike_bnb.csv"))

    def edges(lo, hi):
        return sorted({float(r[lo]) for r in rows} | {float(r[hi]) for r in rows})

    x, y, z = (edges("pz_lo", "pz_hi"), edges("pt_lo", "pt_hi"),
               edges("tp_lo", "tp_hi"))
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
    assert len(rows) == nx * ny * nz, \
        f"minerva_3dqelike table is not a dense {nx}x{ny}x{nz} grid"
    idx = [(int(r["ipz"]), int(r["ipt"]), int(r["itp"])) for r in rows]
    assert idx == sorted(idx) and idx[-1] == (nx - 1, ny - 1, nz - 1), \
        "minerva_3dqelike rows are not in (ipz, ipt, itp) order"
    return TableGrid3D(x, y, z, [float(r["weight"]) for r in rows],
                       excluded=[r["excluded"] == "True" for r in rows])
