"""sBruce file access: branch loading, sentinel handling, derived kinematics.

sBruce conventions (schema 19):
  - All SelectedEvents branches are flat scalars (one value per entry).
  - Missing/unfilled float values are -999 (NOT NaN); Char_t sentinels are -128,
    integer sentinels -1.
  - genie_mode (float): 0=QE, 1=RES, 2=DIS, 3=COH, 10=MEC, -999=no GENIE truth.
  - The genie_prefsi_* particle momenta are stored in a nu/lepton frame:
    z_hat = neutrino direction, y_hat = lepton transverse direction
    (so genie_prefsi_lep_px == 0 and genie_prefsi_lep_py >= 0).
  - multisigmaTree / multisimTree are friend trees aligned by entry index.
"""

import numpy as np
import uproot

SENTINEL = -999.0
# values > VALID_MIN are considered filled (sentinel is exactly -999)
VALID_MIN = -900.0

TREE_NAME = "SelectedEvents"
MULTISIGMA_TREE = "multisigmaTree"
MULTISIM_TREE = "multisimTree"

# genie_mode / true_genie_mode codes (caf::genie_interaction_mode_)
MODE_QE = 0
MODE_RES = 1
MODE_DIS = 2
MODE_COH = 3
MODE_MEC = 10


def valid(*arrays):
    """Elementwise AND of 'branch is filled' (> -900) over the given arrays."""
    out = None
    for a in arrays:
        v = a > VALID_MIN
        out = v if out is None else (out & v)
    return out


class SBruceFile:
    """Read-only accessor for one sBruce file.

    Branch arrays from SelectedEvents are cached; friend-tree access is
    provided for the multisigma weights (e.g. the ZExpPCAWeighter MINERvA
    CV weight used for validation and the divide-out-form-factor option).
    """

    def __init__(self, path):
        self.path = path
        self._file = uproot.open(path)
        self._tree = self._file[TREE_NAME]
        self._cache = {}
        self.n_entries = self._tree.num_entries

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def has_branch(self, name):
        return name in self._tree

    def arrays(self, branches):
        """Load SelectedEvents branches as a dict of numpy arrays (cached)."""
        todo = [b for b in branches if b not in self._cache]
        if todo:
            loaded = self._tree.arrays(todo, library="np")
            self._cache.update(loaded)
        return {b: self._cache[b] for b in branches}

    def array(self, branch):
        return self.arrays([branch])[branch]

    def has_multisigma(self, knob):
        try:
            t = self._file[MULTISIGMA_TREE]
        except KeyError:
            return False
        return knob in t

    def multisigma_at_sigma(self, knob, sigma=0.0):
        """Weight at the given sigma value for a multisigma knob.

        Each knob X is stored as vector<double> branches X (weights) and
        X_sigma (sigma values, typically [-3,-2,-1,0,1,2,3]). Returns a
        float64 array of length n_entries; entries where the requested sigma
        is absent (e.g. empty vectors) are filled with NaN.
        """
        t = self._file[MULTISIGMA_TREE]
        arrs = t.arrays([knob, knob + "_sigma"], library="ak")
        w = arrs[knob]
        s = arrs[knob + "_sigma"]
        match = s == sigma
        # pick the weight where sigma matches; NaN if no match in that entry
        import awkward as ak

        picked = ak.firsts(w[match])
        return ak.to_numpy(ak.fill_none(picked, np.nan)).astype(np.float64)
