#!/usr/bin/env python3
"""Run every validation/<name>/validate_*.py on the CV MC sBruce files.

Each writes validation/plots/<name>/ (PNGs, one PDF, summary.txt; all
gitignored). Prints a one-line FAIL / WARN tally per script at the end.

--legacy also runs the top-level validators: validate_nieves.py (one ICARUS
file) and validate_ccqe_xsec_corr.py (needs PyROOT on PYTHONPATH).
check_outputs.py checks reweight.py outputs, not sBruce inputs, so it is
not run here.

Usage: ./venv/bin/python validation/run_all.py [--max-files N] [--legacy]
                                               [--only name ...]
"""

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subdirectory names to run (default: all)")
    args = ap.parse_args()

    scripts = sorted(glob.glob(os.path.join(HERE, "*", "validate_*.py")))
    if args.only:
        scripts = [s for s in scripts
                   if os.path.basename(os.path.dirname(s)) in args.only]
    extra = ["--max-files", str(args.max_files)] if args.max_files else []
    runs = [(s, extra) for s in scripts]
    if args.legacy:
        runs += [(os.path.join(HERE, "validate_nieves.py"), []),
                 (os.path.join(HERE, "validate_ccqe_xsec_corr.py"), [])]

    results = []
    for script, a in runs:
        name = os.path.relpath(script, HERE)
        print(f"\n######## {name}", flush=True)
        proc = subprocess.Popen([sys.executable, script] + a, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tally = ""
        for line in proc.stdout:
            sys.stdout.write(line)
            if "FAIL," in line and "WARN" in line:
                tally = line.strip()
        rc = proc.wait()
        results.append((name, rc, tally))

    print("\n======== summary")
    for name, rc, tally in results:
        print(f"  {name:55s} exit {rc}  {tally}")
    return max(rc for _, rc, _ in results) if results else 0


if __name__ == "__main__":
    sys.exit(main())
