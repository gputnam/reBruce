# Missing information in the sBruce files (schema 19)

Items found while building the fake-data reweighter that limit or
approximate the calculators, with the placeholder used and what a sBruce
regeneration should add. File set examined:
`/Users/gputnam/Work/osc/sbn-rewgted-19-sBruce` (SBND CV, ICARUS Run2/Run4
overlay CV).

## 1. Pre-FSI GENIE block only filled for ~10% of SBND CV events

`genie_Enu`, `genie_q0/q3/W`, `genie_prefsi_*` are filled (> -900) for only
~10% of selected events in every `SBNDMCCV_*` file, versus ~87% in the
ICARUS overlay files (while `nu_E`/`true_np` are filled for 95-99% in
both). Every pre-FSI-based calculator (MEC BDT, QE axial FF, the W-tercile
modes, pion-channel observables) silently defaults to weight 1 on the
unfilled events, so on SBND the fake-data weights only act on ~10% of the
events they should.

**Placeholder**: weight = 1 outside the filled subset; coverage is printed
per calculator at run time. **Regeneration**: fill the GENIE event-record
block for all MC events.

## 2. No post-FSI charged-pion kinematics

The final-state truth block stores muon and two protons only
(`true_mu_*`, `true_p_*`, `true_p2_*`). The uB CC1pi and T2K NC1pi
measurements are differential in *post-FSI* pion momentum/angle.

**Placeholder**: pion observables (and the theta_mu_pi cut) use the
pre-FSI `genie_prefsi_cpi_*` momenta, in the nu/lepton frame. FSI shifts
the pion spectrum, so bin migrations are not modeled. Pion charge is also
unavailable (`true_npi` counts |pdg| == 211; T2K measures pi+ only).
**Regeneration**: add `true_pi_p`, `true_pi_dir_{x,y,z}` (leading charged
pion, post-FSI, detector frame) and ideally a charge/pdg branch.

## 3. No hit-nucleon radius (or initial-nucleon vector / removal energy)

The Nieves QE cross section depends on the struck nucleon's radial position
r (local Fermi momenta for RPA and Pauli blocking, Coulomb potential). The
initial nucleon 3-momentum and off-shell energy are recoverable exactly
from `genie_prefsi_p`/`genie_prefsi_lep`/`genie_Enu` (verified against
`genie_pmiss`/`genie_emiss` at the 1e-7 level), but r is not stored.

**Placeholder**: the cross section is marginalized over r with the vertex
sampling prior rho(r) r^2 (posterior-weighted by the r-dependent parts of
the cross section). Validated against the GENIE-computed ZExp dial weights
in multisigmaTree: mean ratio 1.0000, RMS <= 0.2% at all 24 sigma points
(`validation/validate_nieves.py`) -- the approximation is excellent for
weight *ratios*. **Regeneration**: store the GHEP hit-nucleon radius (fm)
(plus, for convenience, the initial-nucleon 3-vector and removal energy).

## 4. Signal-definition thresholds vs. the true_n* counts

The measurement signal definitions count particles in phase-space windows
(protons 0.3-1.0 GeV/c; pi+- vetoes at 65/70 MeV/c for the uB CC0pi
channels; "no other mesons" for CC1pi). sBruce stores only threshold-less
primary counts (`true_np`, `true_npi`, `true_npi0`; cafpyana
`makedf.py:462`).

**Per analysis convention (user direction)**: the counts are used directly,
i.e. the count thresholds are ASSUMED to match the measurements'
phase-space definitions. Explicit momentum windows are still applied to the
particles whose kinematics are stored (mu, p, p2, leading pi). Kaon/other-
meson vetoes reduce to the pi0 veto. **Regeneration**: thresholded counts
(e.g. `true_np_300MeV`, `true_npi_65MeV`) or a small per-particle FS table
would make the definitions exact.

## 5. Pre-FSI record: no energies, no neutrons, protons only

`genie_prefsi_*` stores 3-momenta only (energies are recomputed from PDG
masses -- exact for the stored species) and only {lep, p, p2, cpi, pi0, g}.
Consequences:
  - Antineutrino CC QE cannot be reweighted (recoil neutron not stored):
    the QE axial-FF calculator applies to numu only, weight 1 for numubar
    (a ~few-% contamination of the CC sample).
  - The MEC BDT's pp-pair selection is inferred from `genie_prefsi_p2`
    being filled (two pre-FSI protons), matching the training's pp-only
    mask; np/nn initial states get weight 1, as in training.
**Regeneration**: add `genie_prefsi_n_*` (leading neutron) to cover
numubar QE and MEC np states.

## 6. Stored ZExp multisigma weights are CV-normalized

`multisigma_ZExpPCAWeighter_SBN_v3_MvA_b*` sigma=0 entries are identically
1.0: the deuterium->MINERvA CV weight was divided out in production. The
divide_out_ff option therefore cannot read a stored branch and instead
computes the deuterium->MINERvA weight with the validated Nieves port.
**Regeneration**: if the CV weight is applied to `cvwgt` in future
productions, also store it as its own branch so it can be divided out
exactly.

## 7. Duplicate/ambiguous truth blocks (no action needed)

`true_genie_mode` (int16, -1 sentinel) vs `genie_mode` (float, -999);
`nu_E` vs `true_E` vs `genie_Enu`; `true_vtx_*_x` vs `true_vtx_*_y`. The
reweighter uses `genie_mode` (float block) for mode selection, `nu_E` for
signal-definition energies (widest coverage), and `genie_Enu` for pre-FSI
kinematics. Consistency between the blocks was spot-checked on the CV
files.
