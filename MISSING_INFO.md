# Missing information in the sBruce files

Status of the information gaps found while building the fake-data
reweighter. The tool now targets **sBruce schema 20**
(`/Users/gputnam/Work/osc/sbn-rewgted-20-sBruce`), which resolved the
worst schema-19 gaps; the remaining items are listed first.

## Still missing (schema 20)

### 1. Hit-nucleon radius (or initial-nucleon vector / removal energy)

The Nieves QE cross section depends on the struck nucleon's radial position
r (local Fermi momenta for RPA and Pauli blocking, Coulomb potential). The
initial nucleon 3-momentum and off-shell energy are recovered exactly from
`genie_prefsi_{p,n}` / `genie_prefsi_lep` / `genie_Enu` (verified against
`genie_pmiss`/`genie_emiss` at the 1e-7 level), but r is not stored.

**Placeholder**: the cross section is marginalized over r with the vertex
sampling prior rho(r) r^2, posterior-weighted by the r-dependent parts of
the cross section. Validated against the GENIE-computed ZExp dial weights
in multisigmaTree (`validation/validate_nieves.py`):
  - neutrinos: mean ratio 1.0000, RMS <= 0.2% at all 24 sigma points --
    the approximation is excellent;
  - antineutrinos: bulk within ~2% (1st/99th pct [0.94, 1.14]), but ~3% of
    the dial weights differ by > 5%. These are high-Q2 events where the
    RPA-corrected Valencia tensor crosses zero as a function of r (a known
    model pathology, present in GENIE itself); there the ratio is genuinely
    r-sensitive. Extreme values are bounded by the global WEIGHT_CLIP.
**Regeneration**: store the GHEP hit-nucleon radius (fm).

### 2. Charged-pion charge / PDG

`true_cpi_*` (new in schema 20) has no pdg/charge branch, and `true_npi`
counts |pdg| == 211. The T2K measurement is NC1pi+ only; the calculator
accepts either charge. **Regeneration**: add a `true_cpi_pdg` branch.

### 3. Pre-FSI particle multiplicities

Only the *leading* pre-FSI particle of each species is stored
(`genie_prefsi_{cpi,pi0,g,p,p2,n,n2}_*`); there are no pre-FSI counts. The
`jaesung_lowq2_pi_enhancement` `_prefsi` branch therefore cannot apply the
reference's `nPip == 1` requirement and accepts any event with a leading
pre-FSI charged pion, no pre-FSI pi0 and no pre-FSI photon above 10 MeV.
**Per user direction** this leading-particle proxy is used as-is. Note the pi0
and photon vetoes *are* exact -- the stored particle is the leading one by
energy, so "the leading photon is above 10 MeV" is equivalent to "some photon
is above 10 MeV"; only the multiplicity requirement is approximated.
**Regeneration**: pre-FSI per-species counts (`genie_prefsi_ncpi`, ...).

### 4. Signal-definition thresholds vs. the true_n* counts

The measurements count particles in phase-space windows (protons 0.3-1.0
GeV/c; pi+- vetoes at 65/70 MeV/c; "no other mesons"). sBruce stores only
threshold-less primary counts (`true_np`, `true_npi`, `true_npi0`).
**Per analysis convention (user direction)**: the counts are used directly,
i.e. their thresholds are ASSUMED to match the measurements' phase-space
definitions; explicit momentum windows are still applied to the particles
whose kinematics are stored (mu, p, p2, cpi). Kaon/other-meson vetoes
reduce to the pi0 veto. **Regeneration**: thresholded counts or a small
per-particle FS table would make the definitions exact.

### 5. Only the two leading final-state protons are stored

The MINERvA QE-like measurement bins in `SumT_p = sum(E - m_p)` over **all**
final-state protons, but sBruce stores kinematics for only the leading two
(`true_p_p`, `true_p2_p`); `true_np` gives the count but not the momenta.

**Placeholder**: `fakedata.tki.sum_tp` sums the two stored protons. Events
with `true_np > 2` (19% of the in-region population on the ICARUS CV files)
therefore have `SumT_p` under-counted and migrate to lower `SumT_p` bins --
which is where the paper's headline discrepancy lives, so the bias is not
uniform. Per user direction those events are reweighted anyway rather than
dropped. **Regeneration**: store `SumT_p` directly, or a small per-particle
final-state table.

### 6. No heavy-baryon / charm veto

The MINERvA QE-like signal definition (NUISANCE `isCC0pi_MINERvAPTPZ`)
vetoes heavy baryons and charm in the final state. sBruce stores only
`true_np`, `true_nn`, `true_npi`, `true_npi0` and the leading photon, so
that part of the definition is not applied; "no other mesons" likewise
reduces to the pi+-/pi0 vetoes (the same limitation as item 4 above). The
contamination is small at BNB energies -- strange/charm production is far
below threshold for most of the flux -- but it is not zero for the DIS tail.
**Regeneration**: a thresholded final-state species table would make every
meson/baryon veto exact at once (see item 4).

### 7. Stored ZExp multisigma weights are CV-normalized

`multisigma_ZExpPCAWeighter_SBN_v3_MvA_b*` sigma=0 entries are identically
1.0: the deuterium->MINERvA CV weight was divided out in production. The
divide_out_ff option therefore computes that weight with the validated
Nieves port rather than reading a stored branch. **Regeneration**: if the
CV weight is applied to `cvwgt` in future productions, also store it as its
own branch.

### 8. Duplicate/ambiguous truth blocks (no action needed)

`true_genie_mode` (int16, -1 sentinel) vs `genie_mode` (float, -999);
`nu_E` vs `true_E` vs `genie_Enu`. The reweighter uses `genie_mode` for
mode selection, `nu_E` for signal-definition energies, `genie_Enu` for
pre-FSI kinematics. Consistency spot-checked on the CV files.

## Resolved by schema 20

- **Pre-FSI GENIE block coverage** (was ~10% on SBND CV, ~87% ICARUS): now
  filled for 96% (SBND) / 99% (ICARUS) of selected events, matching the
  `nu_E` truth block. All pre-FSI calculators now cover the full MC.
- **Post-FSI charged-pion kinematics**: `true_cpi_p`, `true_cpi_dir_{x,y,z}`
  added (plus `true_pi0_*`, `true_g_*`). The uB CC1pi and T2K NC1pi
  calculators now use post-FSI pion observables in the detector frame
  (previously a pre-FSI `genie_prefsi_cpi` proxy).
- **Pre-FSI neutrons**: `genie_prefsi_n_*`, `genie_prefsi_n2_*` added. The
  QE axial-form-factor calculators now also reweight numubar CC QE
  (recoil neutron), previously weight 1.
