"""Pure-numpy inference for hep_ml GBReweighter models exported to JSON.

Reimplements PROfit/MEC-BDT-WGT/BDTReweight/json_reweighter.py (format
"hep_ml_gb_reweighter", version 1) with vectorized tree traversal:

    score(x)  = initial_step + learning_rate * sum_trees leaf_value(tree, x)
    weight(x) = exp(score(x))

Bit-level detail preserved from the reference implementation: the feature
value is cast to float32 before comparing against the float64 threshold
(`float32(x[feature]) <= threshold -> left child`).

Non-finite feature values are rejected with an exception (policy
"non_finite_policy": "reject") -- callers must mask events first. Note the
sBruce sentinel -999 is finite: mask it out explicitly before calling.
"""

import json

import numpy as np


class GBReweighterJSON:
    def __init__(self, path):
        with open(path) as f:
            model = json.load(f)
        if model.get("format") != "hep_ml_gb_reweighter":
            raise ValueError(f"unexpected model format: {model.get('format')}")
        if model.get("format_version") != 1:
            raise ValueError(f"unexpected format_version: {model.get('format_version')}")

        self.n_features = int(model["n_features"])
        self.learning_rate = float(model["learning_rate"])
        self.initial_step = float(model["initial_step"])

        # flatten each tree into parallel arrays for vectorized descent
        self._trees = []
        for tree in model["trees"]:
            nodes = tree["nodes"]
            n = len(nodes)
            feature = np.zeros(n, dtype=np.int64)
            threshold = np.zeros(n, dtype=np.float64)
            left = np.zeros(n, dtype=np.int64)
            right = np.zeros(n, dtype=np.int64)
            is_leaf = np.zeros(n, dtype=bool)
            value = np.zeros(n, dtype=np.float64)
            for i, node in enumerate(nodes):
                if node["is_leaf"]:
                    is_leaf[i] = True
                    value[i] = node["value"]
                else:
                    feature[i] = node["feature"]
                    threshold[i] = node["threshold"]
                    left[i] = node["left"]
                    right[i] = node["right"]
            self._trees.append((feature, threshold, left, right, is_leaf, value))

    def decision_function(self, events):
        """events: (n_events, n_features) array. Returns float64 scores."""
        x = np.asarray(events)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} features, got {x.shape[1]}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("non-finite feature values (mask events first)")
        # reference implementation casts features to float32 before comparison
        x32 = x.astype(np.float32)

        scores = np.full(x.shape[0], self.initial_step, dtype=np.float64)
        idx_events = np.arange(x.shape[0])
        for feature, threshold, left, right, is_leaf, value in self._trees:
            node = np.zeros(x.shape[0], dtype=np.int64)
            active = ~is_leaf[node]
            while np.any(active):
                nact = node[active]
                fv = x32[idx_events[active], feature[nact]]
                go_left = fv <= threshold[nact]
                node[active] = np.where(go_left, left[nact], right[nact])
                active = ~is_leaf[node]
            scores += self.learning_rate * value[node]
        return scores

    def predict_weights(self, events):
        return np.exp(self.decision_function(events))
