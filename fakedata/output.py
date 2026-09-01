"""Output writing: copy the input sBruce file and append a fakedataTree.

The input file is copied byte-for-byte (original trees untouched), then a new
TTree named "fakedataTree" is appended. The new tree is entry-aligned with
SelectedEvents (one entry per event) and holds every fake-data weight in the
same shape the sBruce multisigmaTree uses for a one-sided variation: a
two-knot dial, cv (sigma 0) and ps1 (sigma 1), with a companion _sigma list.

    multisigma_fdwgt_mec_bdt        [1.0, 1.0234]     <- cv, ps1
    multisigma_fdwgt_mec_bdt_sigma  [0.0, 1.0]

The cv knot is identically 1.0, matching the CV-normalized convention of every
stored sBruce weight, so the dial contributes nothing on top of cvwgt until
PROfit pulls it to +1 sigma. 35 of the file's GENIE knobs already use exactly
this grid (e.g. GENIEReWeight_SBN_v1_multisigma_VecFFCCQEshape).

Both halves of the name are load-bearing: "multisigma" is the routing keyword
MakesBruceNew.C uses to sort weight branches, and PROfit binds a knob-value
list off the "_sigma" suffix, on friend trees as well as the main chain.

Two writers:

  * uproot (default) -- no ROOT needed, but uproot cannot write STL vectors, so
    each dial lands as an int32 counter branch plus a double[] leaf array.
    This is the shape gump.py hands to MakesBruceNew.C.
  * PyROOT (stl_vectors=True) -- genuine std::vector<double> branches, which is
    what PROfit's SetBranchAddress binds to. Needs ROOT on PYTHONPATH.
"""

import os
import shutil

import numpy as np
import uproot

from . import ReBruceError

FAKEDATA_TREE = "fakedataTree"
TREE_TITLE = "Fake-data reweighting weights"

# Prefixed onto every weight branch name. The substring "multisigma" is what
# MakesBruceNew.C routes on; do not drop it.
DIAL_PREFIX = "multisigma_"
SIGMA_SUFFIX = "_sigma"

# The knot grid: cv at sigma 0, ps1 at sigma 1. Same two-knot grid the stored
# one-sided GENIE knobs use.
SIGMA_KNOTS = (0.0, 1.0)
CV_WEIGHT = 1.0


def dial_name(branch):
    """Dial branch name for a calculator weight branch (fdwgt_x -> ...)."""
    return DIAL_PREFIX + branch


def build_dials(weights, n_entries):
    """{dial branch -> (n_entries, 2) float64 block} for a weights dict.

    Column 0 is the cv knot (always 1.0), column 1 the fake-data weight.
    """
    return {
        dial_name(name): np.stack(
            [np.full(n_entries, CV_WEIGHT), np.asarray(arr, dtype=np.float64)],
            axis=1,
        )
        for name, arr in weights.items()
    }


def write_output(input_path, output_path, weights, n_entries, stl_vectors=False):
    """Copy input -> output and append fakedataTree with the given weights.

    weights: dict of branch name -> float64 array, each of length n_entries.
    stl_vectors: write std::vector<double> branches via PyROOT instead of
        uproot's counter + leaf-array form (see the module docstring).
    """
    for name, arr in weights.items():
        if len(arr) != n_entries:
            raise ValueError(
                f"branch {name} has {len(arr)} entries, expected {n_entries}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"branch {name} contains non-finite weights")

    dials = build_dials(weights, n_entries)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    shutil.copyfile(input_path, output_path)

    try:
        if stl_vectors:
            _write_pyroot(output_path, dials, n_entries)
        else:
            _write_uproot(output_path, dials, n_entries)
    except BaseException:
        # the copy is already on disk; without this a failed append leaves a
        # file that looks like an output but has no fakedataTree
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

    return output_path


def _write_uproot(output_path, dials, n_entries):
    """Append fakedataTree with uproot: counter branch + double[] leaf array."""
    import awkward as ak

    sigma = np.broadcast_to(np.asarray(SIGMA_KNOTS, dtype=np.float64),
                            (n_entries, len(SIGMA_KNOTS)))
    data = {}
    for name, block in dials.items():
        # from_regular: a 2D numpy array is a *regular* awkward type, which
        # uproot writes without the counter branch the leaf array needs
        data[name] = ak.from_regular(ak.Array(block))
        data[name + SIGMA_SUFFIX] = ak.from_regular(
            ak.Array(np.ascontiguousarray(sigma)))

    with uproot.update(output_path) as f:
        # mktree/extend writes a classic TTree (plain dict assignment would
        # produce an RNTuple in uproot >= 5.7, which PROfit cannot friend)
        tree = f.mktree(
            FAKEDATA_TREE,
            {name: "var * float64" for name in data},
            title=TREE_TITLE,
        )
        tree.extend(data)


def _write_pyroot(output_path, dials, n_entries):
    """Append fakedataTree with PyROOT: genuine std::vector<double> branches."""
    try:
        import ROOT
    except ImportError as e:
        raise ReBruceError(
            "--stl-vectors needs PyROOT, which is not importable:\n"
            f"  {e}\n"
            "  PyROOT is not on PyPI; it ships with ROOT itself. Install ROOT\n"
            "  (brew install root, or conda install -c conda-forge root) and\n"
            "  put it on the path:  export PYTHONPATH=$(root-config --libdir)\n"
            "  Or drop --stl-vectors to use the uproot writer."
        ) from None

    ROOT.gROOT.SetBatch(True)
    f = ROOT.TFile.Open(output_path, "UPDATE")
    if not f or f.IsZombie():
        raise ReBruceError(f"ROOT could not open for update: {output_path}")
    try:
        tree = ROOT.TTree(FAKEDATA_TREE, TREE_TITLE)
        knots = list(SIGMA_KNOTS)
        bound = []
        for name, block in dials.items():
            wvec = ROOT.std.vector("double")()
            svec = ROOT.std.vector("double")()
            tree.Branch(name, wvec)
            tree.Branch(name + SIGMA_SUFFIX, svec)
            bound.append((wvec, svec, block))

        for i in range(n_entries):
            for wvec, svec, block in bound:
                wvec.assign(block[i].tolist())
                svec.assign(knots)
            tree.Fill()

        tree.Write("", ROOT.TObject.kOverwrite)
    finally:
        f.Close()
