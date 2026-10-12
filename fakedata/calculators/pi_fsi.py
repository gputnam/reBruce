"""Pion FSI reweight: INTRANUKE hA2018 -> hA2025 fate fractions.

Weight = frac_hA2025 / frac_hA2018 for the FSI fate of the event's leading
pre-FSI charged pion, evaluated at the pion's pre-FSI kinetic energy on
Ar40 (see fakedata/ha2025.py and data/ha_pion_fsi_weights_A40.csv).

Fate from the sBruce genie_prefsi_cpi_fsi code, the GENIE v3 INTRANUKE hA
fate (genie::INukeFateHA_t, which has no elastic fate for hA2018):
    2 (kIHAFtCEx)     -> cex
    3 (kIHAFtInelas)  -> inel
    4 (kIHAFtAbs)     -> abs
    7 (kIHAFtPiProd)  -> pipro
    -1/1 (none / kIHAFtNoInteraction) and anything else -> weight 1
    (the hA2018 -> hA2025 map is defined for the four fates above only)
The mapping is verified against GENIE's own FrCEx/FrInel/FrAbs/FrPiProd_pi
multisigma responses in validate-pionFSI/.

Caveat (see MISSING_INFO.md): only the leading charged pion is stored, so
events with several FSI-scattered pions are reweighted by the leading one
only.
"""

import numpy as np

from ..calculator import Calculator, register
from ..ha2025 import HA2025Reweighter
from ..sbruce import valid

M_PION = 0.13957018  # GeV

FSI_CEX = 2
FSI_INELAS = 3
FSI_ABS = 4
FSI_PIPROD = 7

BRANCHES = [
    "genie_prefsi_cpi_px", "genie_prefsi_cpi_py", "genie_prefsi_cpi_pz",
    "genie_prefsi_cpi_fsi",
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
        fate[has_cpi & (fsi == FSI_ABS)] = 1
        fate[has_cpi & (fsi == FSI_INELAS)] = 2
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
