"""TESTING ONLY: fill branches an sBruce file lacks with default values.

DefaultBranchShim wraps an SBruceFile (or anything with its interface) and
serves a missing SelectedEvents branch from a default function of the
branches the file does have. A branch the file carries is never replaced.

The one default set here, FLUX_ANCESTRY_DEFAULTS, supplies the neutrino
parent branches the flux calculators read (fakedata/calculators/flux.py),
which sBruce schema 20 does not export. The values are NOT physics -- one
parent species per flavour, travelling along the beam with the momentum a
forward pi -> mu nu decay would need to produce the neutrino's energy:

    true_parent_pdg        nu_mu 211, nu_mu-bar -211, nu_e 321,
                           nu_e-bar 130; -1 (no truth -> weight 1) otherwise
    true_parent_dcy_mom_x  0
    true_parent_dcy_mom_y  0
    true_parent_dcy_mom_z  true_E / 0.427, -999 without truth
                           (E_nu = 0.427 p_pi for a forward pi -> mu nu)

That puts every truth-matched event in a populated, energy-correlated cell
of the maps, so the full lookup path runs on real files. Weights produced
this way test the plumbing only; they are not a flux variation. Used by
`reweight.py --test-shim` and the unit tests.
"""

import numpy as np

from .sbruce import SENTINEL, valid

# E_nu / p_pi for a pi -> mu nu decay with the neutrino along the pion
# direction, in the relativistic limit: 1 - m_mu^2 / m_pi^2
PION_FORWARD_ENU_FRACTION = 0.427

SHIM_PARENT_BY_FLAVOR = {14: 211, -14: -211, 12: 321, -12: 130}


def _shim_parent_pdg(get):
    pdg = get("true_pdg")
    out = np.full(len(pdg), -1, dtype=np.int32)
    for nu, parent in SHIM_PARENT_BY_FLAVOR.items():
        out[pdg == nu] = parent
    return out


def _shim_zero(get):
    return np.zeros(len(get("true_E")), dtype=np.float32)


def _shim_pz(get):
    E = get("true_E")
    return np.where(valid(E), E / PION_FORWARD_ENU_FRACTION,
                    SENTINEL).astype(np.float32)


# shimmed branch -> (real branches it is built from, builder(get))
FLUX_ANCESTRY_DEFAULTS = {
    "true_parent_pdg": (["true_pdg"], _shim_parent_pdg),
    "true_parent_dcy_mom_x": (["true_E"], _shim_zero),
    "true_parent_dcy_mom_y": (["true_E"], _shim_zero),
    "true_parent_dcy_mom_z": (["true_E"], _shim_pz),
}


class DefaultBranchShim:
    """SBruceFile wrapper: missing branches come from `defaults`."""

    def __init__(self, sbruce, defaults=None):
        self._sb = sbruce
        self.path = getattr(sbruce, "path", None)
        self.n_entries = sbruce.n_entries
        defaults = FLUX_ANCESTRY_DEFAULTS if defaults is None else defaults
        # only branches the file lacks, and whose inputs it has
        self.shimmed = {b: d for b, d in defaults.items()
                        if not sbruce.has_branch(b)
                        and all(sbruce.has_branch(s) for s in d[0])}
        self._cache = {}

    def has_branch(self, name):
        return name in self.shimmed or self._sb.has_branch(name)

    def arrays(self, branches):
        real = [b for b in branches if b not in self.shimmed]
        out = self._sb.arrays(real) if real else {}
        for b in branches:
            if b in self.shimmed and b not in self._cache:
                _, build = self.shimmed[b]
                self._cache[b] = build(lambda s: self._sb.arrays([s])[s])
            if b in self.shimmed:
                out[b] = self._cache[b]
        return {b: out[b] for b in branches}

    def array(self, branch):
        return self.arrays([branch])[branch]

    def __getattr__(self, name):
        # anything else (multisigma access, close, ...) goes to the file
        return getattr(self._sb, name)
