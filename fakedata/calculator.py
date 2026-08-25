"""Weight-calculator base class and registry.

A calculator produces one or more named per-event weight branches from a
SBruceFile. Weights must be exactly 1.0 for events outside the calculator's
domain (so the branch can be multiplied into any PROfit additional_weight
expression unconditionally).

To add a calculator:
  1. Subclass Calculator, set `name`, implement `compute(sbruce) -> dict`
     mapping branch name -> float64 numpy array of length sbruce.n_entries.
  2. Decorate the class with @register("your_type_name").
  3. Reference it in the config: `calculators: [{type: your_type_name, ...}]`.
     The full config dict for the entry is passed to __init__ as `options`.
"""

import numpy as np

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
    """Base class. Subclasses implement compute()."""

    type_name = None

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
