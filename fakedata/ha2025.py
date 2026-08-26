"""hA2018 -> hA2025 pion FSI fate-fraction reweight.

Applies the weight

    w = frac_hA2025(fate; KE, A=40) / frac_hA2018(fate; KE, A=40)

for the four INTRANUKE pi+ FSI fates cex / abs / inel / pipro, from the
table data/ha_pion_fsi_weights_A40.csv. The table is ROOT's own
TGraph2D::Interpolate evaluation of the hA_TGraphs_2D/*.root graphs on a
0.5-MeV KE grid at A = 40, produced by scripts/extract_ha_tgraphs.py with
the reference hA2025Reweighter implementation (ROOT's Delaunay
triangulation is not reproducible outside ROOT, so the information is
extracted as ROOT evaluates it; between grid points the curve is piecewise
linear and np.interp reproduces it to the table's granularity).

Semantics inherited from the reference / GENIE FracADep:
KE clamped to [1, 999] MeV; weight = 1 where the hA2018 fraction is below
1e-6 (unmeasured / roundoff region; the guard is applied at extraction).
"""

import csv
import os

import numpy as np

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                        "ha_pion_fsi_weights_A40.csv")

FATES = ("cex", "abs", "inel", "pipro")


class HA2025Reweighter:
    def __init__(self, csv_path=None):
        rows = {f: [] for f in FATES}
        with open(csv_path or CSV_PATH) as f:
            for r in csv.DictReader(x for x in f if not x.startswith("#")):
                rows[r["fate"]].append((float(r["KE"]), float(r["weight"])))
        self.ke = {}
        self.w = {}
        for fate in FATES:
            pts = sorted(rows[fate])
            self.ke[fate] = np.array([p[0] for p in pts])
            self.w[fate] = np.array([p[1] for p in pts])

    def weight(self, ke_mev, fate_index):
        """Per-event weights.

        ke_mev:     (N,) pion kinetic energy [MeV] (clamped to the table)
        fate_index: (N,) int index into FATES (cex=0, abs=1, inel=2,
                    pipro=3); any other value -> weight 1
        """
        ke = np.asarray(ke_mev, dtype=np.float64)
        out = np.ones(len(ke), dtype=np.float64)
        for i, fate in enumerate(FATES):
            sel = fate_index == i
            if np.any(sel):
                kf = np.clip(ke[sel], self.ke[fate][0], self.ke[fate][-1])
                out[sel] = np.interp(kf, self.ke[fate], self.w[fate])
        return out
