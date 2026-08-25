"""Weight calculators. Importing this package registers all calculator types."""

# Calculators are imported here as they are implemented; each module
# registers its class(es) with fakedata.calculator.register at import time.
from . import mec_bdt  # noqa: F401
from . import qe_zexp  # noqa: F401
from . import xsec_meas  # noqa: F401
