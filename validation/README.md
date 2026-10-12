# Validation

Plots and checks of the calculators on the sbn-rewgted-20 CV MC sBruce files (13 SBND, 7 ICARUS). They are aimed mainly at one class of bug: an sBruce input branch read under a wrong assumption about what it means. Every script runs the real calculator on the real files, per detector, weighted by `cvwgt`. Each one writes the following to `validation/plots/<name>/`, which is gitignored:

- PNGs
- one PDF
- a `summary.txt` of PASS / WARN / FAIL checks

A FAIL is a finding to read, not a crash: no physics check aborts a script.

```bash
./venv/bin/python validation/run_all.py              # everything (~ minutes)
./venv/bin/python validation/run_all.py --max-files 1 --only inputs pion_fsi
./venv/bin/python validation/<dir>/validate_<x>.py   # one script
```

| directory | what it validates |
|---|---|
| `inputs/` | **What the sBruce branches mean.** GENIE knob response per (flavour, CC/NC, `genie_mode`); knob sigma-label orientation; sentinels on cosmic slices; energy branches; pre-FSI frame and recomputed q0 / q3 / W; detector z vs the neutrino direction; counts vs leading-particle branches; de-excitation photons; baseline, `cvwgt` per mode, duplicate truth matches. |
| `pion_fsi/` | `pi_fsi_ha2025`: FSI code -> fate against GENIE's Fr*_pi knobs; fate closure vs the hA tables. This check found the code-3/4 bug fixed in 6ab14e8. |
| `mec_bdt/` | `mec_bdt`: frame conversion against the training code; features; per-file normalization; comparison with the nusystematics SuSAv2 -> Valencia q0 knobs. |
| `qe_zexp/` | `qe_zexp_mva_to_lqcd`: domain; weight vs Q2; ZExp knob label orientation; `cvwgt` containing deuterium -> MINERvA. |
| `ccqe_xsec_corr/` | Whether `cvwgt` already contains the CCQEXSecCorr correction that `ccqe_xsec_corr` applies. The event-by-event port check is the top-level `validate_ccqe_xsec_corr.py`, which needs PyROOT. |
| `spp_lowq2/` | `jaesung_lowq2_pi_enhancement`: independent recomputation; `_postfsi` vs `_prefsi` signals; photon veto vs the reference `IsSPP`. |
| `xsec_meas/` | `ub_cc1p0pi`, `ub_cc2p0pi`, `ub_ccpi`, `t2k_nc1pi`: bin and W-mode closure; signal definition vs the AR23 table builders; frame; `divide_out_ff`. |
| `minerva_3dqelike/` | 3D lookup and p_z-marginalization closure; signal audit; frame migrations. |
| `osc_sterile/` | Both osc calculators: units; fixed 110 m vs per-event `baseline`; unfolded table vs the direct ratio. |

`common.py` holds the shared pieces: file discovery, the plot style, the `Output` / check bookkeeping, and the domain-matrix plot.

Older top-level scripts:

- `validate_nieves.py`: the Nieves port vs the stored ZExp knobs.
- `validate_ccqe_xsec_corr.py`: the CCQE port vs a ROOT C++ reference.
- `check_outputs.py`: checks reweight.py *output* files.

The flux calculators are not covered: schema 20 lacks their input branches.
