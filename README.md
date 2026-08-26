# fake-data-reweighter

Fake-data reweighting for sBruce files (SBN oscillation analysis). A
configurable driver script evaluates per-event weight calculators on the
truth / GENIE-pre-FSI content of a sBruce file and writes the weights to a
new **friend TTree** (`fakedataTree`) in a byte-identical copy of the input
file, one scalar `double` branch per calculator (and per W-mode).

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m pytest tests/          # unit tests
```

No external python installs, ROOT, GENIE, or NUISANCE are needed at runtime.

## Usage

```bash
./venv/bin/python reweight.py configs/all_calculators.yaml \
    --input  /path/to/FILE_sbruce.root \
    --output output/FILE_sbruce_fakedata.root
```

The config lists the input (optional if given on the command line) and the
calculators to run:

```yaml
output_dir: output          # used if --output/output not given
calculators:
  - type: mec_bdt
  - type: qe_zexp_mva_to_lqcd
  - type: ub_cc1p0pi
    variable: dpt_dat                       # default; any CSV observable works
    w_modes: [nominal, loW, midW, hiW]      # default: all four
    divide_out_ff: false                    # default
  - type: ub_cc2p0pi      # default variable: delta_PT
  - type: ub_ccpi         # default variable: PionMomentum
  - type: t2k_nc1pi       # single 2D (p_pi, cos_theta_pi) measurement
```

The full list of `SelectedEvents` branches the reweighting code assumes is
documented in the header comment of `reweight.py`. Weight branches are
exactly `1.0` outside each calculator's domain, so in PROfit they can be
multiplied unconditionally into any `additional_weight`:

```xml
<friend treename="fakedataTree" />
...
<branch ... additional_weight="(!true_isnc)*(true_pdg==14||true_pdg==-14)*cvwgt*wgt_mec_bdt" ...>
```

## Calculators

### `mec_bdt` -> branch `wgt_mec_bdt`
Reweights numu CC MEC events (`genie_mode == 10`) from AR23/SuSAv2 to the
exclusive-Valencia 2p2h model using a BDT (hep_ml GBReweighter, from
PROfit/MEC-BDT-WGT) evaluated on **GENIE pre-FSI** kinematics: the two
pre-FSI protons and the muon in the reaction frame (neutrino along +z, muon
pT along -y). The sBruce `genie_prefsi_*` momenta are already stored in a
nu/lepton frame; the conversion to the BDT frame is a sign flip of the y
components. Applies only to events with a pre-FSI pp nucleon pair (matching
the BDT's training selection). Weights are normalized per input file so the
cvwgt-weighted mean over reweighted events is exactly 1 (shape-only; MEC
rate unchanged). Options: `normalize: per-file | none | fixed` (+
`norm_scale`).

### `qe_zexp_mva_to_lqcd` -> branch `wgt_qe_zexp_mva_to_lqcd`
Reweights the axial form factor of numu and numubar CC QE events from the MINERvA
measurement (Nature 614 (2023) 48, z-expansion a1..a4 = {1.50, -1.2, -0.1,
0.2}, T0 = -0.75) to the LQCD average (a1..a2 = {1.721, -0.31}, T0 = -0.5,
Tcut = 0.161604), as defined by the GENIE tunes AR23_20i_01_001 and
AR23_20i_02_000 in SBNSoftware/Generator (z-expansions per
nusystematics/GENIE `ZExpAxialFormFactorModel`, including the Q4limit
constraint solve).

The weight is a **full ratio of Nieves/Valencia CCQE differential cross
sections** (`fakedata/nieves.py`, a numpy port of GENIE's
`NievesQELCCPXSec` as configured in AR23: RPA + Coulomb + local Fermi gas +
BBA07 vector form factors), evaluated at the event's true kinematics with
only the axial form factor swapped -- NOT an FA^2 ratio, which would ignore
the interference of FA with the other form factors and the RPA structure.
The per-event kinematics (struck-nucleon momentum, binding) are
reconstructed from `genie_Enu`, `genie_prefsi_lep_*` and the pre-FSI recoil
nucleon (`genie_prefsi_p_*` for neutrinos, `genie_prefsi_n_*` for
antineutrinos, sBruce schema >= 20); the unstored hit-nucleon radius is
marginalized over the vertex distribution (see MISSING_INFO.md). The port
is validated event-by-event against the GENIE-computed
`ZExpPCAWeighter_SBN_v3_MvA` dial weights stored in the sBruce
multisigmaTree: for neutrinos, agreement at all 24 sigma points to mean
1.0000, RMS <= 0.2% (`validation/validate_nieves.py`); for antineutrinos
the bulk agrees to ~2%, with rare high-Q2 outliers where the Valencia RPA
tensor crosses zero and the missing radius genuinely matters
(MISSING_INFO.md).

Option `ga_convention: tune` (default; each coefficient set uses its own
FA(0)) or `nusyst` (both use the AR23 CV FA(0) = -1.2670, the
nusystematics ZExpPCAWeighter convention).

### `pi_fsi_ha2025` -> branch `wgt_pi_fsi_ha2025`
Reweights the INTRANUKE pion FSI fate fractions from hA2018 to hA2025:
`w = frac_hA2025 / frac_hA2018` for the FSI fate of the event's leading
pre-FSI charged pion (cex / abs / inel / pipro from `genie_prefsi_cpi_fsi`,
with absorption identified as INELAS + no surviving charged pion), at the
pion's pre-FSI kinetic energy on Ar (A = 40). Elastic / non-interacting
pions and pion-less events get weight 1. The fate-fraction information is
extracted from the `hA_TGraphs_2D/` ROOT TGraph2D files into
`data/ha_pion_fsi_weights_A40.csv` by `scripts/extract_ha_tgraphs.py`
using ROOT's own `TGraph2D::Interpolate` (the binary ROOT files stay out
of the repository); validated off-grid against the reference reweighter to
a mean |dw| ~ 6e-4 (up to ~1.6% locally at the steep pipro turn-on near
KE = 350 MeV).

### Cross-section-measurement calculators
`ub_cc1p0pi`, `ub_cc2p0pi`, `ub_ccpi`, `t2k_nc1pi` -> branches
`wgt_<calc>_<variable>[_loW|_midW|_hiW]`

Reweight from the AR23 GENIE cross section to the measured cross section,
per bin of a configurable observable: `w(x) = sigma_data(x) /
sigma_AR23(x)` on the published bin edges, with `w = 1` outside the
measured range and in bins with no AR23 prediction (no caps). The per-bin
weights were derived in the `fake-data-studies` repository from the
official data releases and a 1M-event AR23_20i_00_000 GHEP sample, and are
stored self-contained in `data/*.csv` (see `data/README.md`), so this
repository has no dependency on those inputs.

| calc | measurement | default variable | other variables |
|---|---|---|---|
| `ub_cc1p0pi` | uB CC1mu1p0pi, PRL 131 101802 | `dpt_dat` (2D dAlphaT in dPT slices) | DeltaPT, DeltaAlphaT, DeltaPhiT, MuonCosTheta, ProtonCosTheta, MuonMomentum, ProtonMomentum, DeltaPtx, DeltaPty, ECal |
| `ub_cc2p0pi` | uB CC1mu2p0pi, PLB 872 140052 | `delta_PT` | muon/leading/recoil mom, costheta, phi; delta_alphaT, delta_phiT, opening angles |
| `ub_ccpi` | uB CC1pi+-, PRD 113 032007 | `PionMomentum` | MuonCosTheta, MuonMomentum, PionCosTheta, ThetaMuPi |
| `t2k_nc1pi` | T2K ND280 NC1pi, PRL 135 171803 | `p_costh` (2D, only option) | -- |

Signal definitions mirror the papers/NUISANCE, computed from post-FSI truth
(`true_mu_*`, `true_p_*`, `true_p2_*`, `true_cpi_*` [sBruce schema >= 20],
counts `true_np/true_npi/true_npi0`); the pion's charge is not stored, so
either charge is accepted (see MISSING_INFO.md). TKI variable formulas are
in `fakedata/tki.py`.

**W modes** (`w_modes: [nominal, loW, midW, hiW]`): in mode X the events in
the relevant cvwgt-weighted equal-population tercile of the `genie_W`
spectrum *of each measured bin* absorb the entire data-MC difference of
that bin: `w_X = 1 + (w_bin - 1) * (sum_bin cvwgt / sum_tercile cvwgt)`,
other terciles get 1. Per-bin reweighted yields are identical to the
nominal mode (verified in tests). Events without `genie_W` get the nominal
`w_bin` and the tercile split renormalizes within the valid-W population.
All weights are clipped to the module-level configuration
`fakedata.calculator.WEIGHT_CLIP`, default `(0, 10)` (see below). Where the
clip engages in a W mode (e.g. the 0.21 CC1p0pi 2D cell driving tercile
weights negative) the bin yield is preserved only up to the clipped amount
(<~2e-4 of the file total; reported by `validation/check_outputs.py`).

**divide_out_ff** (default `false`): divides the weight by the per-event
deuterium->MINERvA axial-form-factor weight (nusystematics ZExpPCAWeighter
CV convention: MINERvA a1..a4 and T0 = -0.75 with the CV FA(0)), computed
with the validated Nieves port. Use for input MC in which that reweight has
been applied to the CV weight, to avoid double-counting the QE form-factor
change when reweighting to data. (The sBruce multisigmaTree stores only
CV-normalized dial variations -- the sigma=0 entries are identically 1 --
so there is no stored branch to divide by; it must be computed.)

## Weight clip

Every produced weight branch is clipped to the module-level configuration
`fakedata.calculator.WEIGHT_CLIP`, default `(0, 10)`. The driver applies it
after each calculator runs (and reports how many events were clipped); the
W-tercile machinery applies the same range internally. In practice the cap
mainly tames rare out-of-distribution MEC BDT weights (raw values up to
O(1000) were observed on the ICARUS CV files) and extreme W-tercile
enhancements. Clipping happens after any calculator-internal normalization,
so clipped events slightly break that calculator's closure. To change the
range, edit the constant or set it before running:

```python
import fakedata.calculator
fakedata.calculator.WEIGHT_CLIP = (0.0, 50.0)
```

## Writing a new calculator

1. Create `fakedata/calculators/my_calc.py`:

```python
import numpy as np
from ..calculator import Calculator, register

@register("my_calc")                       # the config `type:` name
class MyCalculator(Calculator):
    def __init__(self, branch="wgt_my_calc", **options):
        self.branch = branch               # config keys arrive as kwargs

    def compute(self, sbruce):             # sbruce: fakedata.sbruce.SBruceFile
        a = sbruce.arrays(["genie_mode", "cvwgt"])   # numpy arrays
        w = self.ones(sbruce.n_entries)              # start from 1.0
        mask = a["genie_mode"] == 0                  # your domain
        w[mask] = ...                                # your weights
        return {self.branch: w}                      # 1+ named branches
```

2. Import it in `fakedata/calculators/__init__.py`.
3. Rules: weights must be finite, exactly 1.0 outside the calculator's
   domain, and one array entry per `SelectedEvents` entry. Treat any float
   branch value <= -900 as unfilled (`fakedata.sbruce.valid` helps).
   Document any newly-assumed branches in the `reweight.py` header.

## Repository layout

```
reweight.py            driver (header lists all assumed TBranches)
fakedata/              package: sbruce I/O, output writer, physics modules
fakedata/calculators/  the weight calculators
data/                  BDT model + measurement weight tables (provenance inside)
configs/               example configs
tests/                 pytest unit tests
validation/            validation scripts (Nieves vs stored GENIE weights)
MISSING_INFO.md        sBruce information gaps and the placeholders used
```

## Known approximations

The tool targets sBruce schema 20 (schema 19 lacked pre-FSI neutrons,
post-FSI pion kinematics, and full pre-FSI-block coverage on SBND). See
MISSING_INFO.md for the remaining gaps with regeneration recommendations:
missing hit-nucleon radius (marginalized; matters only for rare
antineutrino high-Q2 events), missing pion charge, signal-definition
thresholds assumed equal to the `true_n*` count definitions, and
CV-normalized stored multisigma weights.
