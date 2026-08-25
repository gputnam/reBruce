#!/usr/bin/env python3
"""Fake-data reweighter for sBruce files.

Reads a YAML configuration naming an input sBruce file and a list of weight
calculators; writes a copy of the input file with a new friend TTree
("fakedataTree") holding one scalar float64 weight branch per calculator /
per W-mode. Weights are exactly 1.0 for events outside a calculator's domain,
so every branch can be multiplied unconditionally into a PROfit
additional_weight expression (alongside cvwgt), with the tree declared as a
friend like multisigmaTree.

Usage:
    ./venv/bin/python reweight.py <config.yaml> [--input FILE] [--output FILE]

--input/--output override the config's input/output entries (useful for
running one config over many files).

================================================================================
TBranches of SelectedEvents ASSUMED PRESENT by the reweighting code
================================================================================
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
    genie_prefsi_{lep,p,p2,cpi,pi0,g}_{px,py,pz}
                                pre-FSI particle 3-momenta in the nu/lepton
                                frame (z = nu direction, lepton pT along +y;
                                genie_prefsi_lep_px == 0):
                                  lep = primary lepton
                                  p   = leading proton   (status 14)
                                  p2  = subleading proton (status 14)
                                  cpi = leading charged pion (status 14)
    genie_prefsi_{...}_{status,fsi}
                                GHEP status and FSI fate codes (loaded with the
                                momenta; used only for validity masking)

Post-FSI final-state truth (momentum-ordered) [GeV], -999 when unfilled:
    true_mu_p,  true_mu_dir_{x,y,z}     true muon momentum and direction
    true_p_p,   true_p_dir_{x,y,z}      leading true proton
    true_p2_p,  true_p2_dir_{x,y,z}     subleading true proton
    true_np, true_npi, true_npi0        final-state particle counts

Friend tree multisigmaTree (OPTIONAL -- used when present):
    multisigma_ZExpPCAWeighter_SBN_v3_MvA_b1 (+ _sigma)
                                stored GENIE deuterium->MINERvA axial-FF CV
                                weight (sigma=0 entry); used by the
                                divide_out_ff: stored option and for
                                validating the Nieves cross-section port
================================================================================

Sentinel convention: float branches use -999 for "not filled"; the code treats
values <= -900 as invalid and assigns weight 1.0 to those events.
"""

import argparse
import os
import sys

import numpy as np
import yaml

from fakedata import calculator
from fakedata.output import write_output
from fakedata.sbruce import SBruceFile

# importing the package registers all calculators
import fakedata.calculators  # noqa: F401


def run(config, input_path=None, output_path=None):
    input_path = input_path or config["input"]
    output_path = output_path or config.get("output")
    if output_path is None:
        base = os.path.basename(input_path).replace(".root", "")
        output_path = os.path.join(
            config.get("output_dir", "output"), base + "_fakedata.root"
        )

    calcs = [calculator.build(entry) for entry in config["calculators"]]

    print(f"[reweight] input:  {input_path}")
    print(f"[reweight] output: {output_path}")

    weights = {}
    with SBruceFile(input_path) as sbruce:
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

    write_output(input_path, output_path, weights, n)
    print(f"[reweight] wrote {len(weights)} weight branches "
          f"({', '.join(weights)}) for {n} events")
    return output_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="YAML configuration file")
    ap.add_argument("--input", help="override config input file")
    ap.add_argument("--output", help="override config output file")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run(config, input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    sys.exit(main())
