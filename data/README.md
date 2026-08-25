# Data assets

| File | Contents | Provenance |
|---|---|---|
| `mec_bdt_susav2_to_valencia.json` | hep_ml GBReweighter (200 trees, lr 0.1, 8 features) trained to reweight AR23/SuSAv2 CCMEC (pp pre-FSI states) to exclusive-Valencia 2p2h, ICARUS-BNB numu on C12 | copied from `PROfit/MEC-BDT-WGT/models/ICARUS_mockBNB_numuC12_SuSAv2_to_newValencia_2p2h_pp_states_bdtreweighter.json` (originally trained by Zihao Lin; rsynced from icarusgpvm03.fnal.gov) |
| `ub_cc1p0pi_xsec.csv` | MicroBooNE CC1mu1p0pi xsec / AR23 weights, PRL 131 101802 (2023) | built by `fake-data-studies/build_ub_cc1p0pi_reweight.py` from the NUISANCE data release + AR23 GHEP sample |
| `ub_cc2p0pi_xsec.csv` | MicroBooNE CC1mu2p0pi xsec / AR23 weights, PLB 872 140052 (2026) | built by `fake-data-studies/build_ub_cc2p0pi_reweight.py` |
| `ub_ccpi_xsec.csv` | MicroBooNE CC1pi+- xsec / AR23 weights, PRD 113 032007 (2026) | built by `fake-data-studies/build_ub_ccpi_reweight.py` |
| `t2k_nc1pi_xsec.csv` | T2K ND280 NC1pi+ xsec / AR23 weights, PRL 135 171803 (2025) | built by `fake-data-studies/build_t2k_nc1pi_reweight.py`; sigma_ar23 digitized from Fig. 3 of the paper |

The CSV `weight` column is the per-bin multiplicative weight sigma_data/sigma_AR23
on the published measurement's own bin edges. Full provenance and normalization
details are in each file's `#` header. See the top-level README for how these
are applied (padding to weight 1 outside the measured range, W-tercile modes).
