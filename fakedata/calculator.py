"""Weight-calculator base class and registry.

A calculator produces one or more named per-event weight branches from a
SBruceFile. Weights must be exactly 1.0 for events outside the calculator's
domain (so the branch can be multiplied into any PROfit additional_weight
expression unconditionally).

To add a calculator:
  1. Subclass Calculator, set `name`, implement `compute(sbruce) -> dict`
     mapping branch name -> float64 numpy array of length sbruce.n_entries.
  2. Implement `branches_needed(self) -> list[str]`: every SelectedEvents
     branch compute() reads, given this instance's options. Load arrays with
     `sbruce.arrays(self.branches_needed())` so the two cannot diverge.
  3. Decorate the class with @register("your_type_name").
  4. Reference it in the config: `calculators: [{type: your_type_name, ...}]`.
     The full config dict for the entry is passed to __init__ as `options`.
"""

import numpy as np

from . import ReBruceError


class MissingBranchError(ReBruceError):
    """The input file lacks branches the configured calculators need."""


def dedup(*branch_lists):
    """Concatenate branch lists, dropping repeats, preserving first-seen order."""
    out, seen = [], set()
    for lst in branch_lists:
        for b in lst:
            if b not in seen:
                seen.add(b)
                out.append(b)
    return out


# (min, max) clip applied to every produced weight branch (by the driver,
# after each calculator runs; the W-tercile machinery also applies it
# internally). Module-level configuration: adjust with e.g.
# `fakedata.calculator.WEIGHT_CLIP = (0, 50)` before running.
# Note: clipping happens after any calculator-internal normalization, so a
# clipped event slightly breaks that calculator's closure (e.g. the MEC BDT
# per-file shape normalization).
WEIGHT_CLIP = (0.0, 10.0)

REGISTRY = {}


def register(type_name):
    def wrap(cls):
        cls.type_name = type_name
        REGISTRY[type_name] = cls
        return cls

    return wrap


def build(options):
    """Construct a calculator from one config entry (dict with 'type')."""
    opts = dict(options)
    type_name = opts.pop("type")
    if type_name not in REGISTRY:
        raise KeyError(
            f"unknown calculator type '{type_name}'; known: {sorted(REGISTRY)}"
        )
    return REGISTRY[type_name](**opts)


class Calculator:
    """Base class. Subclasses implement compute() and branches_needed()."""

    type_name = None

    def branches_needed(self):
        """The COMPLETE list of SelectedEvents branches this INSTANCE reads.

        Deduplicated and order-preserving, and dependent on the constructor
        options (e.g. divide_out_ff pulls in the QE form-factor branches).

        compute() MUST load its arrays through this method -- never through a
        second, separately-written list -- so the declaration cannot drift
        from what is actually read. The driver uses it to preflight the input
        file before any calculator runs (see check_branches).
        """
        raise NotImplementedError

    def label(self):
        """Identifier used in driver messages: config type + weight branch."""
        branch = getattr(self, "branch", None)
        return f"{self.type_name} [{branch}]" if branch else str(self.type_name)

    def compute(self, sbruce):
        """Return {branch_name: float64 array of per-event weights}."""
        raise NotImplementedError

    @staticmethod
    def ones(n):
        return np.ones(n, dtype=np.float64)

    @staticmethod
    def report_coverage(label, mask, n):
        frac = float(np.count_nonzero(mask)) / n if n else 0.0
        print(f"    [{label}] weighted events: {np.count_nonzero(mask)}/{n} ({100*frac:.1f}%)")


def check_branches(sbruce, calcs):
    """Preflight: per-calculator missing-branch report for an open sBruce file.

    sbruce: anything with has_branch(name) -> bool (a SBruceFile, or a stub).
    calcs:  the built calculators, in config order.

    Returns [(calc, missing)] with one entry per calculator IN THE GIVEN
    ORDER; `missing` preserves that calculator's branches_needed() order and
    is [] for a calculator that can run. Use blocked() to filter.
    """
    return [(c, [b for b in c.branches_needed() if not sbruce.has_branch(b)])
            for c in calcs]


def blocked(report):
    """The check_branches() entries that have at least one missing branch."""
    return [(c, missing) for c, missing in report if missing]


def format_branch_report(report, path):
    """Human-readable text for a check_branches() result (see reweight.py)."""
    n_calc = len(report)
    n_needed = len(dedup(*[c.branches_needed() for c, _ in report]))
    bad = blocked(report)

    if not bad:
        return (f"[reweight] branch check: {n_calc} calculators, "
                f"{n_needed} distinct branches, all present")

    by_branch = {}
    for c, missing in bad:
        for b in missing:
            by_branch.setdefault(b, []).append(c.label())
    width = max(len(b) for b in by_branch)
    ok = [c.label() for c, missing in report if not missing]

    lines = [f"[reweight] branch check: {n_calc} calculators, "
             f"{n_needed} distinct branches, {len(by_branch)} MISSING",
             f"[reweight]   file: {path}",
             "",
             f"  missing SelectedEvents branches ({len(by_branch)}):"]
    for b in sorted(by_branch):
        lines.append(f"    {b:<{width}}  blocks: {', '.join(by_branch[b])}")
    lines += ["", f"  blocked calculators ({len(bad)} of {n_calc}):"]
    for c, missing in bad:
        lines.append(f"    {c.label()}")
        lines.append(f"        missing: {', '.join(missing)}")
    lines += ["", f"  runnable calculators ({len(ok)} of {n_calc}): "
                  + (", ".join(ok) if ok else "(none)"), ""]
    return "\n".join(lines)
