# Data assets

| File | Contents | Provenance |
|---|---|---|
| `mec_bdt_susav2_to_valencia.json` | hep_ml GBReweighter (200 trees, lr 0.1, 8 features) trained to reweight AR23/SuSAv2 CCMEC (pp pre-FSI states) to exclusive-Valencia 2p2h, ICARUS-BNB numu on C12 | copied from `PROfit/MEC-BDT-WGT/models/ICARUS_mockBNB_numuC12_SuSAv2_to_newValencia_2p2h_pp_states_bdtreweighter.json` (originally trained by Zihao Lin; rsynced from icarusgpvm03.fnal.gov) |
| `ub_cc1p0pi_xsec.csv` | MicroBooNE CC1mu1p0pi xsec / AR23 weights, PRL 131 101802 (2023) | built by `fake-data-studies/build_ub_cc1p0pi_reweight.py` from the NUISANCE data release + AR23 GHEP sample |
| `ub_cc2p0pi_xsec.csv` | MicroBooNE CC1mu2p0pi xsec / AR23 weights, PLB 872 140052 (2026) | built by `fake-data-studies/build_ub_cc2p0pi_reweight.py` |
| `ub_ccpi_xsec.csv` | MicroBooNE CC1pi+- xsec / AR23 weights, PRD 113 032007 (2026) | built by `fake-data-studies/build_ub_ccpi_reweight.py` |
| `t2k_nc1pi_xsec.csv` | T2K ND280 NC1pi+ xsec / AR23 weights, PRL 135 171803 (2025) | built by `fake-data-studies/build_t2k_nc1pi_reweight.py`; sigma_ar23 digitized from Fig. 3 of the paper |
| `minerva_3dqelike_bnb.csv` | MINERvA LE/ME 3D QE-like (p_z, p_T, SumT_p) xsec / AR23 weights, extrapolated per bin to the BNB flux peak, arXiv:2606.00745 | built by `scripts/build_minerva_3dqelike_table.py` from the standalone `MvA_LE_ME/` analysis (`results/weights_ptpzsumtp.csv` + `weights_meta.json`) |
| `ha_pion_fsi_weights_A40.csv` | hA2018->hA2025 pi+ FSI fate-fraction weights vs KE at A=40 (runtime table for `pi_fsi_ha2025`) | tabulated from `hA_TGraphs_2D/TGraphs_{2018,2025}.root` with ROOT TGraph2D::Interpolate by `scripts/extract_ha_tgraphs.py` |
| `flux_hadprod_sw_piplus_u0387.root` | BNB flux, Sanford-Wang pi+ production **universe 387** (SW params c1-c9 in the `info` tree), weight maps per flavor x parent: `weight3` (E_nu, \|p_parent\|, theta_parent), `weight` (E_nu, \|p\|), `weight_E`; flow bins hold edge values. Runtime table for `flux_hadron_production` | byte-identical copy of `sbndgpvm04:/pnfs/sbn/persistent/users/kplows/fake_data/flux/hadroproduction/hadronProd_RW.root` (kplows; built by `caf_weight_maps.py build` from 2.37e8 flux entries); description in the file's `README` TObjString |
| `flux_horn_171p5kA_weight_maps.root` | BNB horn current 174 -> 171.5 kA weight maps per parent: `weight2` (\|p_parent\|, theta_parent) + `error2`; 2D (single v_z slab); **zero flow bins** (lookup clamps); validated for nu_mu only. Runtime table for `flux_horn_current` | byte-identical copy of `.../flux/focusing/horn_current_171p5kA_caf_reweight/bnb_horn_171p5kA_caf_weight_maps.root` (kplows, built 2026-09-18 by `caf_maps.build_f`; CV 1.324e8 POT / varied 1.998e8 POT, 4750 cells, 0.48% flux-weighted stat error); see the file's `provenance` TObjString |
| `ha_pion_fsi_tgraphs.csv` | raw TGraph2D points of the above (archival provenance) | same extraction script |

The CSV `weight` column is the per-bin multiplicative weight sigma_data/sigma_AR23
on the published measurement's own bin edges. Full provenance and normalization
details are in each file's `#` header. See the top-level README for how these
are applied (padding to weight 1 outside the measured range, W-tercile modes).
