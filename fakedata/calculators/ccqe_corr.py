"""CCQE template bugfix: GENIE v3_04_02a -> v3_06_02 CCQE ("sbn2").

A port of the nusystematics CCQEXSecCorr tool (a correction dial: one
weight, no variations), with its fcl/CCQEXSecCorr.ToolConfig.fcl settings:

    w = map[flavor](E_nu; q3, q0)

per-flavor (q3, q0) weight maps on an E_nu grid 0.20..3.00 GeV in 0.05 GeV
steps, blended linearly in E_nu, nearest-bin in (q3, q0), weights clamped
to [0, 30]. The lookup and its provenance are in fakedata/ccqe_corr.py; the
maps in data/ccqe_xsec_corr_sbn2_maps.root.

Domain: CC QE (genie_mode == 0, true_isnc == 0) nu_e, nu_e-bar, nu_mu and
nu_mu-bar (`flavors` option) with filled genie_Enu / genie_q0 / genie_q3,
and 0.20 <= E_nu <= 3.00 GeV. Weight 1 elsewhere.

Two departures from the tool, both toward weight 1 (default
`match_tool: false`):

  - Nucleon QE only. GENIE files QE hyperon production (nu-bar p -> l+
    Lambda / Sigma) and the like under QE too, and the tool weights them
    with maps built for nucleon QE; their q0 is far above the QE peak, in
    empty bins, so most get weight 0. Here an event must have its pre-FSI
    recoil nucleon -- a proton (genie_prefsi_p) for a neutrino, a neutron
    (genie_prefsi_n, sBruce schema >= 20) for an antineutrino -- or it gets
    weight 1. On sbn-rewgted-20: 82 of 370 nu_mu-bar and 19 of 215631 nu_mu
    CC QE events.
  - Map coverage. A (q3, q0) in an empty map bin or off the map axes is
    outside the phase space the maps cover: weight 1 rather than the tool's
    0 / edge-bin value (details in fakedata/ccqe_corr.py).

`match_tool: true` drops both and reproduces the tool exactly (what
validation/validate_ccqe_xsec_corr.py compares against the C++).

genie_q0 / genie_q3 are the lab-frame energy and 3-momentum transfer from
the GENIE record, the tool's q0 = E_nu - E_lep and q3 = |p_nu - p_lep|
(checked against genie_prefsi_lep_* to < 3e-7 GeV on sBruce schema 20).
The driver's global WEIGHT_CLIP (0, 10) is tighter than the tool's 30.
"""

import numpy as np

from ..calculator import Calculator, register
from ..ccqe_corr import FLAVORS, CCQEXSecCorrMaps
from ..sbruce import MODE_QE, valid

BRANCHES = ["genie_mode", "true_isnc", "true_pdg",
            "genie_Enu", "genie_q0", "genie_q3"]
# pre-FSI recoil nucleon per probe: nu n -> l- p, nu-bar p -> l+ n
RECOIL_BRANCHES = ["genie_prefsi_p_px", "genie_prefsi_n_px"]


@register("ccqe_xsec_corr")
class CCQEXSecCorr(Calculator):
    def __init__(self, branch="fdwgt_ccqe_xsec_corr", table=None,
                 flavors=tuple(FLAVORS), match_tool=False):
        self.branch = branch
        self.flavors = [int(f) for f in flavors]
        unknown = set(self.flavors) - set(FLAVORS)
        if unknown:
            raise ValueError(f"no CCQE correction maps for flavor(s) "
                             f"{sorted(unknown)}; have {sorted(FLAVORS)}")
        self.match_tool = bool(match_tool)
        self.maps = CCQEXSecCorrMaps(table, coverage=not self.match_tool)

    def branches_needed(self):
        return list(BRANCHES) + ([] if self.match_tool else RECOIL_BRANCHES)

    def compute(self, sbruce):
        a = sbruce.arrays(self.branches_needed())
        n = sbruce.n_entries
        mask = ((a["genie_mode"] == MODE_QE)
                & (a["true_isnc"] == 0)
                & np.isin(a["true_pdg"], self.flavors)
                & valid(a["genie_Enu"], a["genie_q0"], a["genie_q3"]))
        if not self.match_tool:
            recoil = np.where(a["true_pdg"] > 0, a["genie_prefsi_p_px"],
                              a["genie_prefsi_n_px"])
            nucleon = valid(recoil)
            self.report_coverage(f"{self.branch} no pre-FSI recoil nucleon "
                                 "(weight 1)", mask & ~nucleon, n)
            mask &= nucleon
        weights = self.ones(n)
        weights[mask], uncov_m = self.maps.weight_and_coverage(
            a["true_pdg"][mask], a["genie_Enu"][mask],
            a["genie_q0"][mask], a["genie_q3"][mask])
        self.report_coverage(self.branch, mask & (weights != 1.0), n)
        uncov = np.zeros(n, dtype=bool)
        uncov[mask] = uncov_m
        self.report_coverage(
            f"{self.branch} outside map coverage"
            + (" (tool weight kept)" if self.match_tool else " (weight 1)"),
            uncov, n)
        return {self.branch: weights}
