#!/usr/bin/env python3
"""Fake-data reweighter for sBruce files.

Reads a YAML configuration naming an input sBruce file and a list of weight
calculators; writes a copy of the input file with a new friend TTree
("fakedataTree") holding one two-knot systematic dial per calculator /
per W-mode, in the same shape the sBruce multisigmaTree uses for a one-sided
variation:

    multisigma_fdwgt_mec_bdt        [1.0, 1.0234]     <- cv, ps1
    multisigma_fdwgt_mec_bdt_sigma  [0.0, 1.0]

The cv knot is identically 1.0 and the ps1 knot is the fake-data weight, which
is itself exactly 1.0 for events outside a calculator's domain. The tree is
declared as a friend like multisigmaTree, and PROfit picks the dials up as
type="spline" allowlist entries.

Usage:
    ./venv/bin/python reweight.py <config.yaml> [--input FILE] [--output FILE]
                                  [--check-branches] [--skip-incomplete]
                                  [--stl-vectors]

--input/--output override the config's input/output entries (useful for
running one config over many files). --check-branches is a dry run that only
verifies the input has every branch the configured calculators declare;
--skip-incomplete drops the calculators whose branches are absent and runs
the rest (for older sBruce schemas). --stl-vectors writes the dials as real
std::vector<double> branches with PyROOT (see fakedata/output.py).

================================================================================
TBranches of SelectedEvents ASSUMED PRESENT by the reweighting code
================================================================================
The table below is the annotated union over all calculators. The machine-
readable source of truth is each calculator's branches_needed(); the driver
checks a given file against the configured set before running anything (see
--check-branches). Keep this table in step with it for the physics notes.

Event bookkeeping / weights:
    cvwgt                       central-value event weight (used for
                                normalizations and W-tercile populations)

Interaction-level truth:
    genie_mode                  float; 0=QE 1=RES 2=DIS 3=COH 10=MEC, -999 none
    true_isnc                   char; 0=CC, 1=NC, -128 sentinel
    true_pdg                    int; matched-neutrino PDG (14, -14, ...)

GENIE event-record (pre-FSI) kinematics [GeV], -999 when unfilled:
    genie_Enu                   true neutrino energy
    genie_q0, genie_q3          energy/3-momentum transfer (lab)
    genie_W                     GENIE-convention W (on-shell nucleon at rest)
    genie_pmiss, genie_emiss    struck-nucleon |p| and removal energy
                                (cross-checks for the Nieves QE evaluation)
    genie_prefsi_{lep,p,p2,n}_{px,py,pz}
                                pre-FSI particle 3-momenta in the nu/lepton
                                frame (z = nu direction, lepton pT along +y;
                                genie_prefsi_lep_px == 0):
                                  lep = primary lepton
                                  p   = leading proton    (status 14)
                                  p2  = subleading proton (status 14)
                                  n   = leading neutron   (status 14;
                                        sBruce schema >= 20, used for the
                                        antineutrino QE reweight)
    genie_prefsi_{cpi,pi0,g}_{px,py,pz}
                                pre-FSI leading charged pion / pi0 / photon
                                3-momenta in the same frame
    genie_prefsi_cpi_fsi        INTRANUKE rescatter code of the pre-FSI pion

Post-FSI final-state truth (momentum-ordered) [GeV], -999 when unfilled:
    true_mu_p,  true_mu_dir_{x,y,z}     true muon momentum and direction
    true_p_p,   true_p_dir_{x,y,z}      leading true proton
    true_p2_p,  true_p2_dir_{x,y,z}     subleading true proton
    true_cpi_p, true_cpi_dir_{x,y,z}    leading true charged pion
                                        (sBruce schema >= 20)
    true_g_p                            leading true photon momentum (= energy;
                                        SPP photon veto)
    true_np, true_npi, true_npi0        final-state particle counts

Friend tree multisigmaTree (OPTIONAL -- not read by any calculator):
    multisigma_ZExpPCAWeighter_SBN_v3_MvA_b1 (+ _sigma)
                                stored GENIE deuterium->MINERvA axial-FF
                                variations. These are CV-normalized, so there
                                is no stored CV weight for divide_out_ff to
                                use -- that option recomputes the weight with
                                the Nieves port. Read only by
                                validation/validate_nieves.py.
================================================================================

Sentinel convention: float branches use -999 for "not filled"; the code treats
values <= -900 as invalid and assigns weight 1.0 to those events.
"""

import argparse
import os
import sys

import numpy as np
import yaml

from fakedata import ReBruceError, calculator
from fakedata.output import dial_name, write_output
from fakedata.sbruce import SBruceFile

# importing the package registers all calculators
import fakedata.calculators  # noqa: F401


def run(config, input_path=None, output_path=None,
        check_only=False, skip_incomplete=False, stl_vectors=False):
    input_path = input_path or config["input"]
    output_path = output_path or config.get("output")
    if output_path is None:
        base = os.path.basename(input_path).replace(".root", "")
        output_path = os.path.join(
            config.get("output_dir", "output"), base + "_fakedata.root"
        )

    calcs = [calculator.build(entry) for entry in config["calculators"]]

    print(f"[reweight] input:  {input_path}", flush=True)
    if not check_only:
        print(f"[reweight] output: {output_path}", flush=True)

    weights = {}
    with SBruceFile(input_path) as sbruce:
        # preflight: fail fast, before any calculator runs, on a file whose
        # schema lacks branches the configured calculators declare
        report = calculator.check_branches(sbruce, calcs)
        bad = calculator.blocked(report)
        # flush: the report is stdout, the MissingBranchError message is
        # stderr -- without this they interleave out of order
        print(calculator.format_branch_report(report, input_path), flush=True)

        if check_only:
            if bad:
                n_ok = len(calcs) - len(bad)
                remedy = (f"re-run with --skip-incomplete to drop them and "
                          f"run the other {n_ok}" if n_ok else
                          "no calculator can run on this file")
                raise calculator.MissingBranchError(
                    f"--check-branches: {len(bad)} of {len(calcs)} configured "
                    f"calculators cannot run on {input_path}; {remedy}")
            print("[reweight] --check-branches: OK, no output written")
            return None

        if bad:
            if not skip_incomplete:
                raise calculator.MissingBranchError(
                    f"{len(bad)} of {len(calcs)} configured calculators need "
                    f"branches absent from {input_path} (listed above); "
                    f"re-run with --skip-incomplete to drop them")
            dropped = [c for c, _ in bad]
            print(f"[reweight] --skip-incomplete: dropping {len(dropped)} of "
                  f"{len(calcs)} calculators "
                  f"({', '.join(c.type_name for c in dropped)})")
            calcs = [c for c, missing in report if not missing]
            if not calcs:
                raise calculator.MissingBranchError(
                    "--skip-incomplete dropped every configured calculator; "
                    "nothing to do")

        for calc in calcs:
            print(f"  running calculator: {calc.type_name}")
            branches = calc.compute(sbruce)
            for name, arr in branches.items():
                if name in weights:
                    raise ValueError(f"duplicate weight branch name: {name}")
                # module-level weight clip (fakedata.calculator.WEIGHT_CLIP)
                n_clipped = int(np.sum((arr < calculator.WEIGHT_CLIP[0])
                                       | (arr > calculator.WEIGHT_CLIP[1])))
                if n_clipped:
                    print(f"    [{name}] clipped {n_clipped} weights to "
                          f"{list(calculator.WEIGHT_CLIP)}")
                weights[name] = np.clip(arr, *calculator.WEIGHT_CLIP)
        n = sbruce.n_entries

    write_output(input_path, output_path, weights, n, stl_vectors=stl_vectors)
    writer = "PyROOT, std::vector<double>" if stl_vectors else "uproot"
    print(f"[reweight] wrote {len(weights)} cv/ps1 dials ({writer}) "
          f"for {n} events: "
          f"{', '.join(dial_name(name) for name in weights)}")
    return output_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="YAML configuration file")
    ap.add_argument("--input", help="override config input file")
    ap.add_argument("--output", help="override config output file")
    ap.add_argument(
        "--check-branches", action="store_true",
        help="dry run: report which SelectedEvents branches the configured "
             "calculators need and which the input file is missing, then exit "
             "without writing an output file (status 1 if any are missing)")
    ap.add_argument(
        "--skip-incomplete", action="store_true",
        help="instead of failing, drop the calculators whose branches are "
             "missing from the input file and run the rest (ignored with "
             "--check-branches, which always reports the full picture)")
    ap.add_argument(
        "--stl-vectors", action="store_true",
        help="write the dials as genuine std::vector<double> branches using "
             "PyROOT, which is what PROfit's SetBranchAddress binds to. "
             "Without it the tree is written by uproot, which cannot write "
             "STL vectors and emits a counter branch plus a double[] leaf "
             "array instead. Needs ROOT on PYTHONPATH")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    try:
        run(config, input_path=args.input, output_path=args.output,
            check_only=args.check_branches,
            skip_incomplete=args.skip_incomplete,
            stl_vectors=args.stl_vectors)
    except ReBruceError as e:
        print(f"[reweight] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
