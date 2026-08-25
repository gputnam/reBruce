"""Per-bin W-tercile machinery for the lo-/mid-/hi-W reweight modes.

For each measured bin of the reweighting observable, the MC events in that
bin are split into cvwgt-weighted equal-population terciles of the genie_W
spectrum (as computed in that bin, from the input file itself). In mode X
(lo/mid/hi), the events in tercile X absorb the entire data-MC difference
of the bin:

    w_X = 1 + (w_bin - 1) * (sum_bin cvwgt / sum_tercile cvwgt)

and events in the other terciles get weight 1. This preserves the bin's
total reweighted yield: sum cvwgt*w == w_bin * sum cvwgt.

Events in a measured bin whose genie_W is not filled (-999; e.g. most SBND
events, see MISSING_INFO.md) cannot be assigned a tercile: they receive the
NOMINAL bin weight w_bin, and the tercile split is performed among (and
normalized to) the valid-W population only -- the bin total is still
preserved exactly.

Per-event W-mode weights are clipped to the module-level
fakedata.calculator.WEIGHT_CLIP range (default [0, 10]): the lower bound
guards against a strong downward reweight driving the tercile weight
negative, the upper bound against a sparsely-populated tercile absorbing a
huge enhancement. When the clip engages, exact bin-yield preservation is
broken by the clipped amount (by construction;
validation/check_outputs.py reports the residual).
"""

import numpy as np

from .calculator import WEIGHT_CLIP  # single module-level clip config

W_MODES = ("loW", "midW", "hiW")


def weighted_terciles(w_values, weights):
    """cvwgt-weighted 1/3 and 2/3 quantiles of w_values."""
    order = np.argsort(w_values, kind="stable")
    wv = w_values[order]
    cw = np.cumsum(weights[order])
    tot = cw[-1]
    if tot <= 0:
        return wv[0], wv[-1]
    q1 = wv[np.searchsorted(cw, tot / 3.0)]
    q2 = wv[np.searchsorted(cw, 2.0 * tot / 3.0)]
    return q1, q2


def wmode_weights(bin_idx, n_bins, bin_weights, W, w_valid, cvwgt, mode):
    """Per-event weights for one W mode.

    bin_idx:     (N,) measured-bin index per event (-1 = outside).
    bin_weights: (B,) nominal per-bin weight w_bin.
    W:           (N,) genie_W; w_valid: (N,) bool, W is filled.
    cvwgt:       (N,) central-value weights.
    mode:        "loW", "midW" or "hiW".
    """
    imode = W_MODES.index(mode)
    out = np.ones(len(bin_idx), dtype=np.float64)

    for b in range(n_bins):
        in_bin = bin_idx == b
        if not np.any(in_bin):
            continue
        w_bin = bin_weights[b]
        # unknown-W events in the bin: nominal weight
        unk = in_bin & ~w_valid
        out[unk] = np.clip(w_bin, *WEIGHT_CLIP)
        val = in_bin & w_valid
        nval = np.count_nonzero(val)
        if nval == 0:
            continue
        cv = cvwgt[val]
        q1, q2 = weighted_terciles(W[val], cv)
        wv = W[val]
        tercile = np.where(wv <= q1, 0, np.where(wv <= q2, 1, 2))
        sel = tercile == imode
        cv_sel = np.sum(cv[sel])
        if cv_sel <= 0:
            # empty tercile: fall back to nominal for the valid-W events
            out[val] = np.clip(w_bin, *WEIGHT_CLIP)
            continue
        w_terc = 1.0 + (w_bin - 1.0) * np.sum(cv) / cv_sel
        w_ev = np.ones(nval)
        w_ev[sel] = np.clip(w_terc, *WEIGHT_CLIP)
        out[val] = w_ev

    return out
