"""Fake-data reweighting for sBruce files.

Weight calculators evaluate per-event multiplicative weights from truth /
GENIE-pre-FSI branches of a sBruce file. The driver (reweight.py) writes the
weights into a new friend TTree ("fakedataTree") in a copy of the input file.
"""

__version__ = "0.1.0"
