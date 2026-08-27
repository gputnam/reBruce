# MvA_LE_ME: MINERvA LE/ME 3D QE-like → AR23 reweights

Standalone analysis computing bin-by-bin cross-section weights
**w = σ_MINERvA / σ_AR23** for the MINERvA triple-differential QE-like
measurement ([arXiv:2606.00745](https://arxiv.org/abs/2606.00745),
d³σ/dp_z dp_T dΣT_p on hydrocarbon, LE and ME NuMI beams), followed by a
per-bin **linear extrapolation in effective neutrino energy to the BNB flux
peak** (0.7 GeV). Intended as fake-data-style weights in the reBruce sense
(σ_data/σ_AR23 against GENIE AR23_20i_00_000, clip [0,10], weight 1 outside
the measurement), but this directory is self-contained and does not use
reBruce code.

## Measurement

- Data release: `../MINERvA_LE_ME_3DQELike_cross_section_data_and_covariances/`
  (`MINERvA_{LE,ME}_3DQELike_ptpzsumtp_data_cross_section.csv` + covariances).
- 450 bins: p_z(μ) [1.5,2,2.5,3,3.5,4.5] GeV/c × p_T(μ)
  [0,0.15,0.25,0.325,0.4,0.475,0.55,0.7,0.85,1.0] GeV/c × ΣT_p
  [0,0.02,0.04,0.08,0.12,0.16,0.24,0.32,0.4,0.6,0.799] GeV.
- The release stores **bin-integrated** cross sections (cm²/CH-nucleon per
  bin), *not* divided by bin widths. Verified by closure: the sum over bins
  matches the flux-averaged QE-like cross section in the measured region
  (LE: 2.87e-39 cm²/nucleon, ME: 2.08e-39) and the AR23 prediction agrees to
  ~3% (LE), as expected from the paper.
- Only the 10 zero bins per beam are treated as unmeasured.

## Truth signal definition (applied to GENIE)

Matches NUISANCE `isCC0pi_MINERvAPTPZ` (per D. Ruterbories *et al.*,
PRL 129 021803 and arXiv:2606.00745):
νμ CC; θ(μ,ν) < 20°; final state contains no mesons, no photons with
E > 10 MeV, no heavy baryons/charm (nucleons allowed); ΣT_p = Σ(E−m_p) over
**all** final-state protons; muon within the (p_z, p_T) binning above.

## Fluxes

minerva.fnal.gov currently 404s; durable mirrors used (see
`fluxes/flux_meta.json`):

- **LE**: `numu_fhc` from `minerva_flux.root`, the ancillary file of the
  MINERvA flux paper [arXiv:1607.00704v2](https://arxiv.org/abs/1607.00704)
  (PRD 94 092005 + addendum; PPFX prediction, ν/m²/1e6 POT, 0.5 GeV bins).
  *Caveat*: this is the flux-paper prediction; the measurement additionally
  constrains the LE flux with ν-e scattering (6%) — mostly a normalization
  effect that largely cancels in σ_data/σ_MC shape usage.
- **ME**: `flux_ch` from `MINERvA_ME_NuMu_Flux_Nuclear_Targets_Constrained.root`
  (ν-e-constrained ME flux, PRD 107 012001), mirrored in the
  [NUISANCE repo](https://github.com/NUISANCEMC/nuisance/tree/master/data/flux);
  also on the wayback machine copy of minerva.fnal.gov/minerva-fluxes.
  Variable-width bins storing flux **density** (verified continuous across
  bin-width changes).
- Both are converted to uniform 0.05 GeV piecewise-constant density TH1s for
  `gevgen` (`flux_{LE,ME}_gevgen.root`) — exact resampling, no interpolation.
- Effective energies (flux peaks): **E_LE = 3.25 GeV, E_ME = 5.85 GeV**;
  BNB target **E_BNB = 0.7 GeV** (approximate BNB νμ peak; `config.E_BNB`).

## AR23 prediction

GENIE **v3_06_00** with **AR23_20i_00_000** splines, from the cvmfs ups
install, run in the `fnal-dev-sl7` apptainer container
(`scripts/10_run_genie.sh` → `scripts/genie_job.sh`):
`gevgen --event-generator-list CC`, νμ on C12 (4×2M events/flux) and H1
(0.5M/flux), flux histograms above, E ∈ [0.1,100] GeV; `gntpc -f gst`.
Outputs in `/exp/sbnd/data/users/gputnam/MINERVA-Ar23-scratch/genie/full/`.

Normalization per CH nucleon (`scripts/20_prediction.py`): with
⟨σ_t⟩ = ∫φσ_t^CC dE / ∫φ dE from the spline `xsec_graphs.root` and f_{t,i}
the fraction of generated CC events on target t that are signal in bin i,

    σ_i = [ ⟨σ_C⟩ f_C,i + ⟨σ_H⟩ f_H,i ] / 13        (bin-integrated, cm²/nucleon)

H contributes ~zero signal (no CC0π on a free proton) but belongs in the
per-nucleon denominator; C and H are generated separately so no target-mix
ambiguity enters.

## Weights and BNB extrapolation

`scripts/30_weights.py`:

- w_LE,i = σ_data,LE,i / σ_AR23,LE,i (same for ME). Unmeasured bins → w = 1.
- Linear extrapolation treating each beam as a point at its flux peak:
  w_BNB = a·w_LE + b·w_ME with a = (E_ME−E_BNB)/(E_ME−E_LE) ≈ 1.98,
  b = 1−a ≈ −0.98. Final w_BNB clipped to [0,10] (raw retained).
- Uncertainties: data total uncertainty (covariance diagonal) ⊕ MC stat,
  propagated linearly; LE/ME treated as uncorrelated (conservative for the
  data-driven part; the paper notes interaction/detector systematics are
  correlated between beams).
- Output: `results/weights_ptpzsumtp.csv` (bin edges, σ_data, σ_AR23, w_LE,
  w_ME, w_BNB_raw, w_BNB, errors, flags), `results/weights_meta.json`.

## Caveats

- The extrapolation goes from [3.25, 5.85] GeV down to 0.7 GeV — far outside
  the lever arm; it is a *linear trend estimate*, not a measurement. Bins
  where w_LE ≈ w_ME (energy-independent mismodeling, the paper's headline
  observation) extrapolate stably; bins with strong LE/ME differences can
  extrapolate to <0 or ≫1 and are clipped.
- The measurement is on CH; applying the weights to argon assumes the
  data/model discrepancy transfers.
- The muon kinematic range (1.5 < p_z < 4.5 GeV/c) covers a small corner of
  BNB-energy phase space; weights are 1 outside the measured region.

## Running

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python scripts/00_download_fluxes.py
bash scripts/10_run_genie.sh smoke     # quick validation
bash scripts/10_run_genie.sh full      # ~10 h on 4 cores
./venv/bin/python scripts/20_prediction.py full
./venv/bin/python scripts/30_weights.py
./venv/bin/python scripts/40_plots.py  # -> plots/
```

## Results

Full production (2026-08-26/27): 8M CC events per flux on C12 + 0.5M on H1.
771,949 (LE) / 338,860 (ME) signal events land in the measured region; every
measured bin has MC, and the 10 zero-data bins per beam coincide exactly with
the 10 bins where AR23 predicts zero events.

| | LE | ME |
|---|---|---|
| flux-avg ⟨σ_CC⟩ per C12 | 3.95e-37 cm² | 5.28e-37 cm² |
| total QE-like σ, measured region (data) | 2.87e-39 | 2.08e-39 cm²/nucleon |
| total QE-like σ, measured region (AR23) | 2.93e-39 | 1.72e-39 cm²/nucleon |
| signal-weighted mean weight | 0.98 | 1.21 |
| median weight | 0.91 | 1.20 |

- In the paper's headline discrepancy region (ΣT_p > 0.32 GeV, p_T < 0.325
  GeV/c — pion production + absorption), median w_LE = 0.23, w_ME = 0.56;
  w_LE < w_ME reproduces the published LE/ME double ratio < 1 for AR23.
- BNB extrapolation (a=1.98, b=−0.98 at 0.7 GeV): median w_BNB = 0.63,
  signal-weighted mean 0.74. Of 439 extrapolated bins, 82 extrapolate
  negative (clipped to 0) and 4 above 10 (clipped) — bins with w_ME ≳ 2·w_LE.
- w_LE/w_ME are stored *unclipped* in the CSV (5 bins have w_LE > 5, all
  low-AR23-rate corners with 11–436 MC events); only w_BNB has a clipped
  column. Apply reBruce's global clip downstream as appropriate.

GENIE outputs (ghep + gst + logs, 27 GB) are kept in
`/exp/sbnd/data/users/gputnam/MINERVA-Ar23-scratch/genie/full/`.

## Outputs

- `results/weights_ptpzsumtp.csv` — one row per 3D bin (450 rows), sorted by
  (ipz, ipt, itp). Columns: bin indices and edges; `sigma_data_{LE,ME}`
  (+stat/tot errors) and `sigma_ar23_{LE,ME}` (+MC stat error), all
  bin-integrated cm²/CH-nucleon (`sigma_ar23_*_d3` are the width-divided
  differentials); `w_LE`, `w_ME` (+errors, unclipped); `w_BNB_raw`, `w_BNB`
  (clipped to [0,10]), `w_BNB_err`; flags `unmeasured_{LE,ME}`,
  `no_mc_{LE,ME}`, `excluded` (excluded ⇒ all weights = 1).
- `results/prediction_{LE,ME}.csv`, `prediction_{LE,ME}_meta.json` — AR23
  predictions and bookkeeping (event counts, ⟨σ⟩).
- `results/weights_meta.json` — effective energies, extrapolation
  coefficients, clip counts, weight summary statistics.
- `plots/` (png+pdf):
  - `01_fluxes` — LE/ME shapes, peaks, BNB extrapolation point.
  - `02_xsec_validation_{LE,ME}` — data vs AR23, panels per p_T, curves per
    p_z vs ΣT_p (paper Fig. 4 style).
  - `03_xsec_projections` — 1D projections onto p_z, p_T, ΣT_p.
  - `04_weight_map_{LE,ME,BNB}` — 2D (p_T × ΣT_p) weight heatmaps per p_z
    bin (grey = unmeasured/excluded).
  - `05_weight_grid_{LE,ME,BNB}` — weights with errors, panels per p_T,
    curves per p_z; BNB panel marks raw values of clipped bins with ×.
  - `06_weight_projections` — AR23-signal-weighted mean weight vs each
    variable, LE/ME/BNB overlaid.
  - `07_extrapolation_demo` — per-bin w vs E_ν lines through (E_LE, w_LE),
    (E_ME, w_ME) extended to 0.7 GeV for representative bins.
  - `08_weight_distributions` — weight histograms, w_ME vs w_LE scatter
    colored by w_BNB, clipping summary.
