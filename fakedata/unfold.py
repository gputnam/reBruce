"""D'Agostini iterative (Bayesian) unfolding on a 2D migration matrix.

Used by scripts/build_osc_ic2024_unfolded.py to map a reco-energy spectrum
onto true energy. The migration matrix is the event-weighted joint histogram
N[r, t] (reco bin r, true bin t). Every event is assumed to land in some reco
bin (efficiency 1), which holds when the reco binning has open under- and
overflow bins.

G. D'Agostini, Nucl. Instrum. Meth. A 362, 487 (1995).
"""

import numpy as np


def dagostini(migration, data, n_iter, prior=None):
    """Unfold `data` (reco bins) to true bins with n_iter Bayesian iterations.

    migration: (n_reco, n_true) joint histogram N[r, t].
    data:      (n_reco,) reco spectrum to unfold.
    prior:     (n_true,) starting true spectrum; default the migration
               matrix's true marginal (the MC truth), for which unfolding
               the MC's own reco marginal returns the prior exactly.

    Returns a list of the true spectra after each iteration (length n_iter).
    """
    m = np.asarray(migration, dtype=np.float64)
    d = np.asarray(data, dtype=np.float64)
    true = m.sum(axis=0)
    if np.any(true <= 0):
        raise ValueError("every true bin needs MC events (empty column in "
                         "the migration matrix)")
    p_r_given_t = m / true                     # columns sum to 1
    u = true.copy() if prior is None else np.asarray(prior, dtype=np.float64)
    out = []
    for _ in range(int(n_iter)):
        joint = p_r_given_t * u                # P(r|t) u_t
        norm = joint.sum(axis=1, keepdims=True)
        p_t_given_r = np.divide(joint, norm, out=np.zeros_like(joint),
                                where=norm > 0)
        u = p_t_given_r.T @ d
        out.append(u)
    return out


def refold(migration, true_spectrum):
    """Reco spectrum M . u, with M = P(r|t) from the migration matrix."""
    m = np.asarray(migration, dtype=np.float64)
    return (m / m.sum(axis=0)) @ np.asarray(true_spectrum, dtype=np.float64)
