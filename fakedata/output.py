"""Output writing: copy the input sBruce file and append a fakedataTree.

The input file is copied byte-for-byte (original trees untouched), then a new
TTree named "fakedataTree" is appended with uproot.update. The new tree is
entry-aligned with SelectedEvents (one entry per event) and holds one scalar
float64 branch per calculator weight, used in PROfit via a friend declaration:

    <friend treename="fakedataTree" />
    ... additional_weight="(...)*cvwgt*wgt_mec_bdt" ...
"""

import os
import shutil

import numpy as np
import uproot

FAKEDATA_TREE = "fakedataTree"


def write_output(input_path, output_path, weights, n_entries):
    """Copy input -> output and append fakedataTree with the given weights.

    weights: dict of branch name -> float64 array, each of length n_entries.
    """
    for name, arr in weights.items():
        if len(arr) != n_entries:
            raise ValueError(
                f"branch {name} has {len(arr)} entries, expected {n_entries}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"branch {name} contains non-finite weights")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    shutil.copyfile(input_path, output_path)

    with uproot.update(output_path) as f:
        # mktree/extend writes a classic TTree (plain dict assignment would
        # produce an RNTuple in uproot >= 5.7, which PROfit cannot friend)
        tree = f.mktree(
            FAKEDATA_TREE,
            {name: np.float64 for name in weights},
            title="Fake-data reweighting weights",
        )
        tree.extend(
            {name: np.asarray(arr, dtype=np.float64) for name, arr in weights.items()}
        )

    return output_path
