# reBruce

Fake-data re-weighting for sBruce files (SBN oscillation analysis). A
configurable driver script evaluates per-event weight calculators on the
truth / GENIE-pre-FSI content of a sBruce file and writes the weights to a
new **friend TTree** (`fakedataTree`) in a byte-identical copy of the input
file, one two-knot systematic dial per calculator (and per W-mode).

Each dial is stored in the same shape the sBruce `multisigmaTree` uses for a
one-sided variation -- **cv** at sigma 0 and **ps1** at sigma 1, with a
companion `_sigma` knot list:

```
multisigma_fdwgt_mec_bdt        [1.0, 1.0234]     <- cv, ps1
multisigma_fdwgt_mec_bdt_sigma  [0.0, 1.0]
```

The cv knot is identically 1.0, matching the CV-normalized convention of every
stored sBruce weight, so a dial contributes nothing on top of `cvwgt` until
PROfit pulls it to +1 sigma. 35 of the file's GENIE knobs already use exactly
this grid (e.g. `GENIEReWeight_SBN_v1_multisigma_VecFFCCQEshape`).

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m pytest tests/          # unit tests
```

No external python installs, ROOT, GENIE, or NUISANCE are needed at runtime --
with one optional exception. `--stl-vectors` (below) writes the dials through
PyROOT, which is not pip-installable and ships with ROOT itself:

```bash
brew install root                       # or: conda install -c conda-forge root
export PYTHONPATH=$(root-config --libdir)
./venv/bin/python -c "import ROOT; print(ROOT.gROOT.GetVersion())"   # 6.36.00
```

Without it the default uproot writer is unaffected.

## Usage

```bash
./venv/bin/python reweight.py configs/all_calculators.yaml \
    --input  /path/to/FILE_sbruce.root \
    --output output/FILE_sbruce_fakedata.root
```

`--check-branches` verifies, without writing anything, that the input file
has every `SelectedEvents` branch the configured calculators declare, and
exits nonzero if not, naming the missing branches and the calculators each
one blocks. A normal run performs the same check first and fails fast rather
than dying part-way through with an uproot `KeyError`. `--skip-incomplete`
drops the calculators whose branches are absent and runs the rest -- useful
for older sBruce schemas, at the cost of an output with fewer weight
branches than its siblings.

`--stl-vectors` writes the dials as genuine `std::vector<double>` branches
using PyROOT. That is the type PROfit's `SetBranchAddress` binds to
(`PROcreate.cxx`, `eweight_type = double`), so it is what a file destined for a
PROfit fit needs. Without the flag the tree is written by uproot, which cannot
write STL vectors and emits an `int32` counter branch plus a `double[]` leaf
array instead -- readable by uproot and by `TTreeFormula`, and the same shape
`gump.py` hands to `MakesBruceNew.C`, but not bindable as a vector. Values are
identical either way.

### A whole production directory

`reweight.py` takes one file at a time. To run a full production directory
-- environment setup, working out which files are central-value MC,
preflighting the branches, running and validating -- use the `reweight-sbruce`
skill (`.claude/skills/reweight-sbruce/SKILL.md`):

```
/reweight-sbruce /Users/gputnam/Work/osc/sbn-rewgted-20-sBruce/sbn-rewgted-20
```

It defaults to `configs/all_calculators.yaml` and writes to
`output/<input-folder-name>/`.

The config lists the input (optional if given on the command line) and the
calculators to run:

```yaml
output_dir: output          # used if --output/output not given
calculators:
  - type: mec_bdt
  - type: qe_zexp_mva_to_lqcd
  - type: jaesung_lowq2_pi_enhancement       # 2 branches: _postfsi + _prefsi
  - type: ub_cc1p0pi
    variable: dpt_dat                       # default; any CSV observable works
    w_modes: [nominal, loW, midW, hiW]      # default: all four
    divide_out_ff: false                    # default
  - type: ub_cc2p0pi      # default variable: delta_PT
  - type: ub_ccpi         # default variable: PionMomentum
  - type: t2k_nc1pi       # single 2D (p_pi, cos_theta_pi) measurement
  - type: minerva_3dqelike                  # 2 branches: 3D + p_z-marginalized
    pz_ref: LE                              # default; p_z scaling reference
```

Each calculator declares the `SelectedEvents` branches it reads in
`branches_needed()` -- that is the source of truth, and what
`--check-branches` verifies a file against. The header comment of
`reweight.py` carries the same list annotated with the physics meaning of
each branch.

The ps1 knot is exactly `1.0` outside each calculator's domain, so a dial is a
no-op there at any sigma. Declare the tree as a friend and list the dials on
the allowlist, exactly like any other spline systematic:

```xml
<friend treename="fakedataTree" />
...
<allowlist type="spline" tag="fakedata" plotname="MEC BDT">multisigma_fdwgt_mec_bdt</allowlist>
```

Both halves of the branch name are load-bearing: `multisigma` is the routing
keyword `MakesBruceNew.C` sorts weight branches on, and PROfit binds the knob
value list off the `_sigma` suffix, on friend trees as well as the main chain.

## Calculators

### `mec_bdt` -> branch `fdwgt_mec_bdt`
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

### `qe_zexp_mva_to_lqcd` -> branch `fdwgt_qe_zexp_mva_to_lqcd`
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

### `pi_fsi_ha2025` -> branch `fdwgt_pi_fsi_ha2025`
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

### `jaesung_lowq2_pi_enhancement` -> branches `fdwgt_jaesung_lowq2_pi_enhancement_{postfsi,prefsi}`
The single-pion-production (SPP) central-value correction from the ICARUS NuMI
numu cross-section analysis (J. Kim), ported from
[`sbnana/SBNAna/Cuts/NuMIXSecSysts.cxx`][NuMIXSecSysts]: for events passing the
`IsSPP` definition (L14-L81), the low-`Q^2` SPP enhancement template
`GetSPPQ2Reweight` (L94-L118, 1.25 - 1.73, peaking at `Q^2 ~ 0.075 GeV^2`) times
the MINERvA untracked-pion `T_pi` suppression `GetSPPTpiMINERvAFittedReweight`
(L241-L265, a Landau fit below `T_pi = 225 MeV`, a binned template above). Their
product is `kTruth_NuMISPPCVCorrection`. The branch holds that correction
itself (AR23 -> corrected model), **not** the `1/CVCorr` "back to nominal"
orientation the `ISyst::Shift` methods in that file use. `Q^2` is
`genie_q3^2 - genie_q0^2` (there is no `genie_Q2` branch); events without it,
and all events outside the SPP definition, get weight 1.

**Two branches**, differing only in where the signal definition and the leading
pion's kinetic energy come from:

| branch | signal definition | `T_pi` |
|---|---|---|
| `_postfsi` | final state: `true_npi == 1`, `true_npi0 == 0`, no `true_g_p` above 10 MeV | `true_cpi_p` |
| `_prefsi` | initial state: a pre-FSI charged pion, no pre-FSI `pi0`, no pre-FSI photon above 10 MeV | `\|genie_prefsi_cpi_p\|` |

The reference's `IsSPP` loops over the G4 primary list, so `_postfsi` is its
direct analogue; `_prefsi` moves the same correction onto the GENIE event
record, and the difference between the two branches is the FSI sensitivity of
the correction. The pre-FSI signal is the wider of the two (13.4% vs 10.1% of
events on `SBNDMCCV_12`, since FSI absorbs pions); the two weights differ on
~10% of all events. The product stays within `[0.39, 2.10]`, so `WEIGHT_CLIP`
never engages.

The `pi0` and photon vetoes are **exact**: the stored particle is the leading
one by energy, so "the leading photon is above 10 MeV" is equivalent to "some
photon is above 10 MeV". Approximations (MISSING_INFO.md): no pion charge is
stored, so the reference's "exactly one `pi+`" becomes "exactly one charged
pion of either sign"; the `n_mesons == 1` requirement reduces to the `pi0` veto
(no kaon/eta information); only the *leading* pre-FSI pion is stored, so
`_prefsi` cannot veto events with more than one pre-FSI charged pion; and the
reference's `TargetA == 1` (hydrogen) veto is dropped, the SBN target being
argon. Following the reference, no CC / numu requirement is imposed -- `IsSPP`
is a final-state definition only.

`TMath::Landau` is reproduced by `fakedata/landau.py`, a numpy port of ROOT's
`ROOT::Math::landau_pdf` (CERNLIB G110 `denlan`), validated in the tests
against the Landau integral representation
`p(z) = 1/pi * int_0^inf exp(-u ln u - z u) sin(pi u) du` to a relative 1e-6.

[NuMIXSecSysts]: https://github.com/jedori0228/sbnana/blob/feature/jskim_NuMINumuXSec_v09_93_01_TrackSplitSystShiftTest_TestSaveOneTrack/sbnana/SBNAna/Cuts/NuMIXSecSysts.cxx#L14-L81

### Cross-section-measurement calculators
`ub_cc1p0pi`, `ub_cc2p0pi`, `ub_ccpi`, `t2k_nc1pi` -> branches
`fdwgt_<calc>_<variable>[_loW|_midW|_hiW]`

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

### `minerva_3dqelike` -> branch `fdwgt_minerva_3dqelike_bnb`
Reweights numu CC QE-like events from AR23 to the MINERvA triple-differential
QE-like measurement (D. Ruterbories *et al.*, [arXiv:2606.00745], d3sigma /
dp_z dp_T dSumT_p on hydrocarbon, LE and ME NuMI beams), after a per-bin
**linear extrapolation in effective neutrino energy to the BNB flux peak**:

    w_BNB = 1.9808 * w_LE - 0.9808 * w_ME,   clipped to [0, 10]

with each beam treated as a point at its flux peak (E_LE = 3.25 GeV,
E_ME = 5.85 GeV, E_BNB = 0.7 GeV). The derivation -- fluxes, the GENIE
v3_06_00 / AR23_20i_00_000 prediction, per-CH-nucleon normalization, and
plots -- is the standalone `MvA_LE_ME/` analysis; its 450-bin result is
copied into `data/minerva_3dqelike_bnb.csv` by
`scripts/build_minerva_3dqelike_table.py` so this repository stays
self-contained.

**The p_z axis is scaled relative to the neutrino energy**, not treated as
absolute momentum. A measured edge `p_z` corresponds at BNB to
`E_BNB * (p_z / E_LE)`, so an event's measurement-frame longitudinal
momentum is `p_z / pz_scale` with `pz_scale = E_BNB / E_LE = 0.2154`. This
matters: taken literally the measured window 1.5 <= p_z < 4.5 GeV/c is
essentially empty at BNB energies (0 of 15203 events in `SBNDMCCV_0`, 25 of
32650 in `ICARUSRun4_rewgt_0`) and the calculator would be a no-op; with the
scaling those edges map to 0.323 - 0.969 GeV/c, the bulk of the BNB muon
spectrum, and 28% (ICARUS) / 18% (SBND) of events by cvwgt are reweighted.
`pz_ref: LE` (default) | `ME` | `none`; `none` (scale 1.0) reproduces the
literal published binning and is for reference only.

**p_T and SumT_p are NOT scaled.** p_T is set by the Q^2 scale and SumT_p is
hadronic recoil energy; neither tracks E_nu the way the longitudinal boost
does, and both already populate the measured bins with ~0% overflow
(scaling them would push 86% / 41% of in-range events past the last edge).

Signal (NUISANCE `isCC0pi_MINERvAPTPZ`): numu CC; no mesons; no photon above
10 MeV; theta(mu, nu) < 20 deg applied on the **scaled** kinematics, i.e. in
the same frame as the bin lookup (it is nearly redundant with the (p_z, p_T)
binning there -- 51 of 10009 in-range ICARUS events). `SumT_p = sum(E - m_p)`
over the stored protons; sBruce keeps only the leading two, so events with
`true_np > 2` (19% in region) are under-counted, and the heavy-baryon/charm
veto cannot be applied (both in MISSING_INFO.md).

**Two branches are produced.** `fdwgt_minerva_3dqelike_bnb` is the 3D lookup
above. `fdwgt_minerva_3dqelike_bnb_pzmarg` **marginalizes the measurement over
p_z** and applies the result to every signal event in a (p_T, SumT_p) cell
*regardless of its p_z*, and **also drops the theta(mu, nu) < 20 deg cut**:
for each cell the five p_z bins are averaged with the cvwgt-weighted p_z
spectrum **of the input file itself**,

    w_marg(j,k) = sum_i N_ijk * w_ijk / sum_i N_ijk

with `N_ijk` the cvwgt sum of that file's signal events in 3D bin (i,j,k).
Both the spectrum and the events the weight is applied to use the no-theta
signal, so the two branches have *different* signal populations. Every
in-p_z-range bin contributes, including the 11 `excluded` bins at their
conventional 1.0, so the marginalized weight reproduces the 3D lookup's
total yield in each cell *exactly* over its own population (verified to
1e-12 in the tests). Cells with no in-p_z-range events in a given file fall
back to 1.0 -- worth watching on small files, where a third of the 90 cells
can be empty (the driver prints the count).

The theta cut is dropped only here, and the asymmetry is deliberate: for the
3D weight it is nearly redundant with the p_z window and costs ~0.2% of
cvwgt, so it stays; once the weight is applied at any p_z that redundancy
disappears and theta becomes the binding acceptance constraint. Note this
lets backward-going muons (p_z < 0) into the marginalized signal, since only
p_T and SumT_p are then required to be in range.

The marginalized branch exists because the p_z window is what limits the 3D
branch's reach: it alone costs ~12% of cvwgt, and 63% (ICARUS) / 77% (SBND)
of the QE-like sample sits below its 0.323 GeV/c floor. Marginalizing buys
that reach at the price of a stronger assumption -- that the data/AR23
discrepancy within a (p_T, SumT_p) cell is p_z-independent (which is the
p_z analogue of the paper's headline energy-independence, but not something
it measures). Being spectrum-weighted per file, it is **not** a fixed table:
files with different p_z spectra get slightly different marginalized
weights. Use one branch or the other, never both at once.

Nominal weight only -- no W-tercile branches. Note that **82 of the 450 bins
carry weight 0** (their raw extrapolation went negative and was clipped), so
this fake data kills events outright in those cells; the overall
cvwgt-weighted mean weight is ~0.90 (ICARUS) / ~0.93 (SBND). The
extrapolation runs from [3.25, 5.85] GeV down to 0.7 GeV, far outside the
lever arm: it is a linear trend estimate, not a measurement. The measurement
is on CH, so applying it to argon assumes the discrepancy transfers.

[arXiv:2606.00745]: https://arxiv.org/abs/2606.00745

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
    def __init__(self, branch="fdwgt_my_calc", **options):
        self.branch = branch               # config keys arrive as kwargs

    def branches_needed(self):             # every branch compute() reads
        return ["genie_mode", "cvwgt"]

    def compute(self, sbruce):             # sbruce: fakedata.sbruce.SBruceFile
        a = sbruce.arrays(self.branches_needed())    # numpy arrays
        w = self.ones(sbruce.n_entries)              # start from 1.0
        mask = a["genie_mode"] == 0                  # your domain
        w[mask] = ...                                # your weights
        return {self.branch: w}                      # 1+ named branches
```

2. Import it in `fakedata/calculators/__init__.py`.
3. Rules: weights must be finite, exactly 1.0 outside the calculator's
   domain, and one array entry per `SelectedEvents` entry. Treat any float
   branch value <= -900 as unfilled (`fakedata.sbruce.valid` helps).
   A calculator returns flat per-event weights; `fakedata/output.py` wraps
   each one into its cv/ps1 dial and prefixes the name, so `fdwgt_my_calc`
   reaches the file as `multisigma_fdwgt_my_calc` (+ `_sigma`).
   Declare every branch you read in `branches_needed()` and load them with
   `sbruce.arrays(self.branches_needed())`, so the declaration cannot drift
   from what is read -- `--check-branches` relies on it. Add the physics
   annotation for any new branch to the `reweight.py` header too.

## Repository layout

```
reweight.py            driver (--check-branches preflight; the header
                       annotates all assumed TBranches)
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
