"""Pion FSI reweight: INTRANUKE hA2018 -> hA2025 fate fractions.

Weight = frac_hA2025 / frac_hA2018 for the FSI fate of the event's leading
pre-FSI charged pion, evaluated at the pion's pre-FSI kinetic energy on
Ar40 (see fakedata/ha2025.py and data/ha_pion_fsi_weights_A40.csv).

Fate from the sBruce genie_prefsi_cpi_fsi code (INTRANUKE rescatter code;
see cafpyana gump/loaddf.py):
    2 (CEX)    -> cex
    4 (INELAS) -> "abs" when no charged pion survives to the final state
                  (true_npi == 0; a pion absorbed on the nucleus is coded
                  INELAS with no pion daughters), else "inel"
    7 (PIPROD) -> pipro
    -1/1/3 (none / no-interaction / elastic) and anything else -> weight 1
    (the hA2018 -> hA2025 map is defined for the four fates above only)

Caveats (see MISSING_INFO.md): only the leading charged pion is stored, so
events with several FSI-scattered pions are reweighted by the leading one
only; the abs-vs-inel split uses the event-level true_npi count as a proxy
for "this pion had no pion daughters".
"""

import numpy as np

from ..calculator import Calculator, register
from ..ha2025 import HA2025Reweighter
from ..sbruce import valid

M_PION = 0.13957018  # GeV

FSI_CEX = 2
FSI_INELAS = 4
FSI_PIPROD = 7

BRANCHES = [
    "genie_prefsi_cpi_px", "genie_prefsi_cpi_py", "genie_prefsi_cpi_pz",
    "genie_prefsi_cpi_fsi",
    "true_npi",
]


@register("pi_fsi_ha2025")
class PiFSIhA2025(Calculator):
    def __init__(self, branch="fdwgt_pi_fsi_ha2025", table=None):
        self.branch = branch
        self.rw = HA2025Reweighter(table)

    def branches_needed(self):
        return list(BRANCHES)

    def compute(self, sbruce):
        a = sbruce.arrays(self.branches_needed())
        n = sbruce.n_entries

        has_cpi = valid(a["genie_prefsi_cpi_px"], a["genie_prefsi_cpi_py"],
                        a["genie_prefsi_cpi_pz"])
        fsi = a["genie_prefsi_cpi_fsi"]

        # map FSI code -> FATES index (cex=0, abs=1, inel=2, pipro=3); -1 = no reweight
        fate = np.full(n, -1, dtype=np.int64)
        fate[has_cpi & (fsi == FSI_CEX)] = 0
        inelas = has_cpi & (fsi == FSI_INELAS)
        fate[inelas & (a["true_npi"] == 0)] = 1   # absorbed
        fate[inelas & (a["true_npi"] > 0)] = 2    # inelastic, pion survives
        fate[has_cpi & (fsi == FSI_PIPROD)] = 3

        mask = fate >= 0
        self.report_coverage(self.branch, mask, n)

        weights = self.ones(n)
        if not np.any(mask):
            return {self.branch: weights}

        p2 = (a["genie_prefsi_cpi_px"][mask] ** 2
              + a["genie_prefsi_cpi_py"][mask] ** 2
              + a["genie_prefsi_cpi_pz"][mask] ** 2)
        ke_mev = 1000.0 * (np.sqrt(p2 + M_PION ** 2) - M_PION)
        weights[mask] = self.rw.weight(ke_mev, fate[mask])
        return {self.branch: weights}
