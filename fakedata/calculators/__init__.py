"""Weight calculators. Importing this package registers all calculator types."""

# Calculators are imported here as they are implemented; each module
# registers its class(es) with fakedata.calculator.register at import time.
from . import mec_bdt  # noqa: F401
from . import minerva_qelike  # noqa: F401
from . import pi_fsi  # noqa: F401
from . import qe_zexp  # noqa: F401
from . import spp_lowq2  # noqa: F401
from . import xsec_meas  # noqa: F401
