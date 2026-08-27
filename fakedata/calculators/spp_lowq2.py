"""Single-pion-production low-Q2 enhancement x MINERvA Tpi suppression.

Port of the SPP central-value correction from the ICARUS NuMI numu
cross-section analysis (J. Kim), sbnana/SBNAna/Cuts/NuMIXSecSysts.cxx:

    if( !IsSPP(nu) ) return 1.;                     # L14-L81
    Q2RW  = GetSPPQ2Reweight(kTruth_Q2(nu))         # L94-L118
    TpiRW = GetSPPTpiMINERvAFittedReweight(Tpi)     # L241-L265
    CVCorr = Q2RW * TpiRW                           # kTruth_NuMISPPCVCorrection

The branch holds CVCorr itself (AR23 -> corrected model), not the 1/CVCorr
"back to nominal" orientation the ISyst::Shift methods in that file use.

Two branches are produced, differing only in where the single-pion signal
definition and the leading-pion kinetic energy come from:

  _postfsi  final-state (post-FSI) truth: true_npi / true_npi0 / true_g_p
            and true_cpi_p -- the direct analogue of the reference, whose
            IsSPP loops over the G4 primary list.
  _prefsi   initial-state (pre-FSI) GENIE record: genie_prefsi_cpi /
            genie_prefsi_pi0 / genie_prefsi_g.

Q2 = genie_q3^2 - genie_q0^2 is common to both; events without it (or
outside the signal definition) get exactly 1.0.

Approximations (see MISSING_INFO.md):
  - no pion charge is stored, so the reference's "exactly one pi+" becomes
    "exactly one charged pion of either sign";
  - no kaon/eta information is stored, so the reference's n_mesons == 1
    reduces to the pi0 veto (the existing repository convention);
  - sBruce keeps only the *leading* pre-FSI charged pion and no pre-FSI
    multiplicities, so the _prefsi definition cannot veto events with more
    than one pre-FSI charged pion;
  - the reference's TargetA == 1 (hydrogen) veto is dropped: the SBN target
    is argon.
The photon and pi0 vetoes are exact: the stored particle is the leading one
by energy, so "the leading photon is above 10 MeV" is equivalent to "some
photon is above 10 MeV".

Following the reference, no CC / numu requirement is imposed -- IsSPP is a
final-state definition only.
"""

import numpy as np

from ..bba07 import M_PION
from ..calculator import Calculator, register
from ..landau import landau
from ..sbruce import valid

BRANCH = "wgt_jaesung_lowq2_pi_enhancement"
POSTFSI_SUFFIX = "_postfsi"
PREFSI_SUFFIX = "_prefsi"

# photon veto threshold, GeV (reference: prim.genE * 1000. > 10.0 MeV)
PHOTON_VETO_E = 0.010

# GetSPPQ2Reweight: 12 edges / 13 values, with the Q2 >= 3 -> X = 2.5 clamp
# folded in (X = 2.5 lands in [2.0, 3.0), whose value equals the final else).
_Q2_EDGES = np.array([0.025, 0.050, 0.100, 0.200, 0.300, 0.400,
                      0.500, 0.700, 1.000, 1.300, 2.000, 3.000])
_Q2_VALUES = np.array([1.253255, 1.589738, 1.733869, 1.651728, 1.659705,
                       1.584229, 1.703793, 1.475510, 1.456727, 1.252215,
                       1.048199, 1.650489, 1.650489])

# GetSPPTpiMINERvAFittedReweight: Landau fit below the cutoff, template above
_TPI_LANDAU_CUTOFF = 0.225          # GeV
_TPI_LANDAU_NORM = 6.70797696
_TPI_LANDAU_MPV = 0.12235454
_TPI_LANDAU_WIDTH = 0.05731087
_TPI_EDGES = np.array([0.250, 0.275, 0.300, 0.325, 0.350,
                       0.400, 0.500, 0.700, 1.000, 2.000])
_TPI_VALUES = np.array([0.755932, 0.638574, 0.493987, 0.391947, 0.323265,
                        0.452765, 0.594541, 0.768459, 0.658024, 0.873622,
                        0.873622])


def spp_q2_reweight(q2):
    """GetSPPQ2Reweight(Q2 [GeV^2]) -- the low-Q2 SPP enhancement template."""
    q2 = np.asarray(q2, dtype=np.float64)
    return _Q2_VALUES[np.searchsorted(_Q2_EDGES, q2, side="right")]


def spp_tpi_reweight(tpi):
    """GetSPPTpiMINERvAFittedReweight(Tpi [GeV]) -- the MINERvA pion-KE suppression."""
    tpi = np.asarray(tpi, dtype=np.float64)
    fitted = _TPI_LANDAU_NORM * landau(tpi, _TPI_LANDAU_MPV, _TPI_LANDAU_WIDTH)
    template = _TPI_VALUES[np.searchsorted(_TPI_EDGES, tpi, side="right")]
    return np.where(tpi < _TPI_LANDAU_CUTOFF, fitted, template)


def _kinetic_energy(p):
    """Pion kinetic energy [GeV] from its momentum magnitude [GeV]."""
    return np.sqrt(p ** 2 + M_PION ** 2) - M_PION


def _mag(a, prefix):
    """|p| of a genie_prefsi_<prefix> particle (nu/lepton frame; |p| is invariant)."""
    return np.sqrt(a[prefix + "_px"] ** 2 + a[prefix + "_py"] ** 2
                   + a[prefix + "_pz"] ** 2)


def _present(a, prefix):
    return valid(a[prefix + "_px"], a[prefix + "_py"], a[prefix + "_pz"])


Q2_BRANCHES = ["genie_q0", "genie_q3"]

POSTFSI_BRANCHES = ["true_npi", "true_npi0", "true_g_p", "true_cpi_p"]

PREFSI_BRANCHES = [
    "genie_prefsi_cpi_px", "genie_prefsi_cpi_py", "genie_prefsi_cpi_pz",
    "genie_prefsi_pi0_px", "genie_prefsi_pi0_py", "genie_prefsi_pi0_pz",
    "genie_prefsi_g_px", "genie_prefsi_g_py", "genie_prefsi_g_pz",
]


@register("jaesung_lowq2_pi_enhancement")
class SPPLowQ2PiEnhancement(Calculator):
    def __init__(self, branch=BRANCH):
        self.branch = branch

    def compute(self, sbruce):
        a = sbruce.arrays(Q2_BRANCHES + POSTFSI_BRANCHES + PREFSI_BRANCHES)
        n = sbruce.n_entries

        has_q2 = valid(a["genie_q0"], a["genie_q3"])
        q2 = np.where(has_q2, a["genie_q3"] ** 2 - a["genie_q0"] ** 2, 0.0)
        q2_rw = spp_q2_reweight(q2)

        out = {}
        for suffix, (mask, tpi) in (
                (POSTFSI_SUFFIX, self._postfsi(a)),
                (PREFSI_SUFFIX, self._prefsi(a))):
            name = self.branch + suffix
            mask = mask & has_q2
            self.report_coverage(name, mask, n)
            w = self.ones(n)
            w[mask] = q2_rw[mask] * spp_tpi_reweight(tpi[mask])
            out[name] = w
        return out

    @staticmethod
    def _postfsi(a):
        """IsSPP on the final state, and the leading post-FSI pion's KE."""
        mask = (
            (a["true_npi"] == 1)                       # exactly one charged pion
            & (a["true_npi0"] == 0)                    # meson veto -> pi0 veto
            & ~(a["true_g_p"] > PHOTON_VETO_E)         # no photon above 10 MeV
            & valid(a["true_cpi_p"])
        )
        return mask, _kinetic_energy(np.where(mask, a["true_cpi_p"], 0.0))

    @staticmethod
    def _prefsi(a):
        """IsSPP on the GENIE pre-FSI record, and that pion's KE."""
        has_cpi = _present(a, "genie_prefsi_cpi")
        has_pi0 = _present(a, "genie_prefsi_pi0")
        has_g = _present(a, "genie_prefsi_g")
        photon = np.where(has_g, _mag(a, "genie_prefsi_g"), 0.0)
        mask = has_cpi & ~has_pi0 & ~(photon > PHOTON_VETO_E)
        p = np.where(has_cpi, _mag(a, "genie_prefsi_cpi"), 0.0)
        return mask, _kinetic_energy(p)
