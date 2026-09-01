# Acceptance tests: branch preflight + the `reweight-sbruce` skill

Hand this file to Claude (or work through it yourself) to re-verify the branch
preflight and the `reweight-sbruce` skill after a change. Everything runs from the
repository root. Steps 1-6 are mechanical and should be run every time; step 7 costs
three sub-agents and is worth it whenever `SKILL.md` changes.

All scratch artefacts go in `scratch/` (gitignored) and are deleted in step 8.

## Paths this file assumes

| what | where |
|---|---|
| sBruce ROOT production (26 files) | `/Users/gputnam/Work/osc/sbn-rewgted-20-sBruce/sbn-rewgted-20` |
| GUMPLE HDF5 `.df` production (47 files, **not** sBruce) | `/Users/gputnam/Work/osc/sbn-rewgted-21` |
| reference outputs from a known-good run | `output/*_sbruce_fakedata.root` |

If the sBruce production has moved, substitute the new path throughout; if the
reference outputs are gone, skip step 3 and say so rather than silently passing.

---

## 1. Unit tests

```bash
./venv/bin/python -m pytest tests/ -q
```

**Expect:** all pass. As of the cv/ps1 dial change this is **80 tests** (55 original
+ 20 branch-declaration/preflight tests + 5 output-format tests). A drop in count
means tests were lost, not that things got faster.

`test_write_output_stl_vectors` is skipped unless PyROOT is importable
(`export PYTHONPATH=$(root-config --libdir)`); a run reporting `79 passed,
1 skipped` means ROOT was not on the path, not that anything is wrong.

The highest-value test in that set is `test_compute_reads_only_declared_branches`:
it drives each calculator through a stub that fails if `compute()` loads a branch
the calculator did not declare. If you add a calculator and it passes, the
declaration is honest.

## 2. Declared-branch inventory

```bash
./venv/bin/python - <<'EOF'
import fakedata.calculators  # noqa: F401
from fakedata.calculator import REGISTRY
from fakedata.calculators.qe_zexp import QE_BRANCHES
from fakedata.calculators.xsec_meas import UBCC1p0pi
tot = set()
for name, cls in sorted(REGISTRY.items()):
    b = cls().branches_needed()
    assert len(set(b)) == len(b), f"{name}: duplicate branches"
    tot |= set(b)
    print(f"{name:32s} {len(b):3d}")
print("union:", len(tot))
print("divide_out_ff adds QE branches:",
      set(QE_BRANCHES) <= set(UBCC1p0pi(divide_out_ff=True).branches_needed()),
      "| absent by default:",
      not set(QE_BRANCHES) <= set(UBCC1p0pi().branches_needed()))
EOF
```

**Expect** exactly:

| calculator | branches |
|---|---|
| `jaesung_lowq2_pi_enhancement` | 15 |
| `mec_bdt` | 13 |
| `minerva_3dqelike` | 14 |
| `pi_fsi_ha2025` | 5 |
| `qe_zexp_mva_to_lqcd` | 13 |
| `t2k_nc1pi` | 9 |
| `ub_cc1p0pi` | 20 |
| `ub_cc2p0pi` | 20 |
| `ub_ccpi` | 24 |
| **union** | **51** |

plus `True | True` on the last line. `ub_cc1p0pi(divide_out_ff=True)` is 31.

A count that moves is not automatically wrong — it is wrong if it moved without a
calculator changing.

## 3. Regression: weights must not drift

The refactor was meant to be behaviour-preserving. Reproduce one file and diff it
against a known-good output.

```bash
D=/Users/gputnam/Work/osc/sbn-rewgted-20-sBruce/sbn-rewgted-20
mkdir -p scratch
./venv/bin/python reweight.py configs/all_calculators.yaml \
  --input $D/SBNDMCCV_12_sbruce.root --output scratch/regress.root >/dev/null

./venv/bin/python - <<'EOF'
import awkward as ak, numpy as np, uproot

def weights(tree):
    """{calculator branch -> per-event weight}, either output layout.

    Reads the ps1 knot of a cv/ps1 dial, or a legacy flat wgt_* scalar, so a
    baseline in output/ predating the dial change still compares.
    """
    out = {}
    for key in tree.keys():
        if key.startswith('wgt_'):                       # legacy flat scalar
            out[key[len('wgt_'):]] = tree[key].array(library='np')
        elif key.startswith('multisigma_fdwgt_') and not key.endswith('_sigma'):
            knots = np.asarray(ak.to_numpy(ak.to_regular(tree[key].array())))
            sigma = np.asarray(ak.to_numpy(ak.to_regular(
                tree[key + '_sigma'].array())))
            assert np.all(sigma == np.array([0.0, 1.0])), key
            assert np.all(knots[:, 0] == 1.0), f"cv knot is not 1.0: {key}"
            out[key[len('multisigma_fdwgt_'):]] = knots[:, 1]
    return out

old = weights(uproot.open('output/SBNDMCCV_12_sbruce_fakedata.root')['fakedataTree'])
new = weights(uproot.open('scratch/regress.root')['fakedataTree'])
assert set(old) == set(new), f"branch set changed: {set(old) ^ set(new)}"
worst = 0.0
for b in sorted(new):
    a, c = old[b], new[b]
    d = np.max(np.abs(a - c)) if len(a) == len(c) else float('inf')
    if d: print(f"  DIFFERS {b}: max|delta| = {d:.3e}")
    worst = max(worst, d)
print(f"{len(new)} weights, max |delta| = {worst:.3e}")
print("BIT-IDENTICAL" if worst == 0 else "CHANGED -- justify before proceeding")
EOF
```

**Expect:** 23 weights, `max |delta| = 0.000e+00`, `BIT-IDENTICAL`. The cv knot
and sigma grid assertions must also hold for every dial.

Any non-zero delta must be explained by an intentional physics change. If you
intended one, re-baseline by regenerating `output/` and note it in the commit.

### 3b. The STL-vector writer

`--stl-vectors` must produce the same numbers through a different writer.
Skip this step if `import ROOT` fails.

```bash
./venv/bin/python reweight.py configs/all_calculators.yaml \
  --input $D/SBNDMCCV_12_sbruce.root --output scratch/regress_stl.root \
  --stl-vectors >/dev/null

./venv/bin/python - <<'EOF'
import awkward as ak, numpy as np, uproot
a = uproot.open('scratch/regress.root')['fakedataTree']
b = uproot.open('scratch/regress_stl.root')['fakedataTree']
types = {b[k].typename for k in b.keys()}
print("branches:", len(b.keys()), "| types:", types)
assert types == {'std::vector<double>'}, types
worst = max(np.max(np.abs(
    np.asarray(ak.to_numpy(ak.to_regular(a[k].array())))
    - np.asarray(ak.to_numpy(ak.to_regular(b[k].array())))))
    for k in b.keys())
print("max |uproot - PyROOT| =", worst)
EOF
```

**Expect:** `46 branches`, `{'std::vector<double>'}`, `max |uproot - PyROOT| = 0.0`.
The uproot output has 92 branches for the same 23 dials -- uproot adds an `int32`
counter (`nmultisigma_fdwgt_*`) it cannot avoid; PyROOT writes the vectors
directly and needs none.

## 4. Preflight against real files

```bash
D=/Users/gputnam/Work/osc/sbn-rewgted-20-sBruce/sbn-rewgted-20
./venv/bin/python reweight.py configs/all_calculators.yaml \
  --input $D/SBNDMCCV_12_sbruce.root --check-branches; echo "exit=$?"
./venv/bin/python reweight.py configs/all_calculators.yaml \
  --input $D/SBND_SpringBNBOffData_sbruce.root --check-branches; echo "exit=$?"
```

**Expect:**
- CV file: `9 calculators, 51 distinct branches, all present`, `exit=0`.
- Off-beam data: `30 MISSING`, `blocked calculators (9 of 9)`,
  `runnable calculators (0 of 9): (none)`, `exit=1`, and the error line ends
  `no calculator can run on this file`.

The off-beam file is a genuine missing-branch case in the real production — it has
no `genie_*` or `true_np*` branches at all. It is the best free test available.

Check the ordering too: the stdout report must appear **before** the stderr error
line. They are separate streams and only stay in order because the report is
flushed.

## 5. Fault-injected fixtures

Build sBruce files with specific branches removed, from a real file.

```bash
mkdir -p scratch
./venv/bin/python - <<'EOF'
import os, numpy as np, uproot
SRC = '/Users/gputnam/Work/osc/sbn-rewgted-20-sBruce/sbn-rewgted-20/SBNDMCCV_12_sbruce.root'
N = 2000
t = uproot.open(SRC)['SelectedEvents']
data = t.arrays(list(t.keys()), entry_stop=N, library='np')
for fn, drop in {
    'good_sbruce.root':          [],
    'drop_cpi_sbruce.root':      ['true_cpi_p'],
    'drop_prefsi_n_sbruce.root': ['genie_prefsi_n_px', 'genie_prefsi_n_py',
                                  'genie_prefsi_n_pz'],
    'drop_cvwgt_sbruce.root':    ['cvwgt'],
}.items():
    keep = {k: v for k, v in data.items() if k not in drop}
    with uproot.recreate('scratch/' + fn) as out:
        tree = out.mktree('SelectedEvents', {k: v.dtype for k, v in keep.items()})
        tree.extend(keep)
    print(f"scratch/{fn}: dropped {drop or '(nothing)'}")

# a ROOT file with no SelectedEvents
with uproot.recreate('scratch/no_tree.root') as f:
    f['Events'] = {'x': np.arange(5, dtype=np.float64)}
    f['MetaData'] = {'y': np.arange(3, dtype=np.float64)}
# HDF5 magic bytes behind a .root name
open('scratch/fake_hdf5_sbruce.root', 'wb').write(b'\x89HDF\r\n\x1a\n' + b'\x00' * 512)
print("scratch/no_tree.root, scratch/fake_hdf5_sbruce.root")
EOF

for f in good drop_cpi drop_prefsi_n drop_cvwgt; do
  echo "===== $f ====="
  ./venv/bin/python reweight.py configs/all_calculators.yaml \
    --input scratch/${f}_sbruce.root --check-branches 2>&1 \
    | grep -E "branch check|blocked calculators|ERROR|OK,"
done
```

**Expect:**

| fixture | missing | blocked | note |
|---|---|---|---|
| `good` | 0 | — | `all present`, exit 0 |
| `drop_cpi` | 1 (`true_cpi_p`) | **3 of 9** | `jaesung_lowq2_pi_enhancement`, `ub_ccpi`, `t2k_nc1pi` |
| `drop_prefsi_n` | 3 | **1 of 9** | `qe_zexp_mva_to_lqcd` only |
| `drop_cvwgt` | 1 (`cvwgt`) | **6 of 9** | the 3 survivors read no `cvwgt` |

Each failing case must end with a remedy, e.g.
`re-run with --skip-incomplete to drop them and run the other 6`. A diagnosis with
no next step is a regression — that was a review finding.

Then check `--skip-incomplete` actually produces the reduced set:

```bash
./venv/bin/python reweight.py configs/all_calculators.yaml \
  --input scratch/drop_cpi_sbruce.root --output scratch/partial.root \
  --skip-incomplete 2>&1 | grep -E "dropping|wrote"
```

**Expect:** `dropping 3 of 9 calculators (jaesung_lowq2_pi_enhancement, ub_ccpi,
t2k_nc1pi)` and `wrote 13 cv/ps1 dials` (vs 23 for a complete run).

## 6. Structural errors: readable, not tracebacks

```bash
for f in scratch/fake_hdf5_sbruce.root scratch/no_tree.root scratch/nope.root README.md; do
  echo "----- $f"
  ./venv/bin/python reweight.py configs/all_calculators.yaml --input $f --check-branches 2>&1 | tail -3
done
```

**Expect** one-line messages, **no Python traceback**, exit 1 in each case:
- HDF5 behind a `.root` name and `README.md` → `not a readable ROOT file`, quoting
  the first four bytes;
- `no_tree.root` → `ROOT file has no 'SelectedEvents' tree`, listing
  `Events, MetaData`, and `(is this an sBruce file?)`;
- `nope.root` → `cannot open input file`.

## 7. The skill itself

Run these as three **forked** sub-agents (`subagent_type: fork`), in parallel. Tell
each to follow `.claude/skills/reweight-sbruce/SKILL.md` **verbatim without
improvising around gaps**, to report every place the instructions were ambiguous or
wrong, and not to edit any repo file or touch `venv/`. Where SKILL.md says to ask
the user, they should state the question and proceed rather than block.

Ask each agent explicitly whether `Skill(skill: "reweight-sbruce")` resolved or
whether they had to `cat` the file — see the caveat at the bottom.

**(a) Happy path** — `/reweight-sbruce /Users/gputnam/Work/osc/sbn-rewgted-20-sBruce/sbn-rewgted-20`

Expect 20 CV files selected (13 `SBNDMCCV_*`, 3 `ICARUSRun2_SpringMCOverlay_rewgt_*`,
4 `ICARUSRun4_SpringMCOverlay_rewgt_*`); dirt / off-beam / unblind skipped;
`SBND_SpringLowEMC_sbruce.root` raised as `ask`; outputs in
`output/sbn-rewgted-20/`; `check_outputs.py` → **20 files, 0 failures**. Each file
takes well under a second, so the whole run is seconds, not minutes.

**(b) Negative path** — `/reweight-sbruce /Users/gputnam/Work/osc/sbn-rewgted-21`

Expect a clean stop at Discover: 47 `.df` files, zero `.root`, no traceback, no
output directory created, no attempt to feed a `.df` to `reweight.py`, and a message
explaining that the GUMPLE HDF5 dataframes and the sBruce ROOT files live in
separate directories. The agent should verify the file type itself rather than
repeating a claim from SKILL.md.

**(c) Fault path** — `/reweight-sbruce scratch/` (after step 5 has built the fixtures)

None of those names match the CV or skip rules, so all readable ones should land in
`ask` and the two broken ones in `unreadable`. Expect the preflight findings of step
5, and expect the agent to process only `good_sbruce.root`. Ask it to judge, as a
reader rather than the author, whether the missing-branch report tells a user what
to do next.

### Classification dry-run (cheap, no agents)

Worth running whenever the pattern lists change — it exercises them against real
sample names without processing anything:

```bash
./venv/bin/python - <<'EOF'
import os
SKIP = ["BNBOff","OffData","unblind","Data","OnBeam","Dirt","Intime","SCE","DENT",
        "WMYZ","WMNom","WMXThXW","WMXThetaXW"]
CV = ["MCCV","MCOverlay"]
from collections import Counter
names = sorted(os.path.splitext(f)[0]
               for f in os.listdir('/Users/gputnam/Work/osc/sbn-rewgted-21'))
res = {n: ("skip" if any(k in n for k in SKIP)
           else "CV MC" if any(k in n for k in CV) else "ask") for n in names}
print(Counter(res.values()))
print("ask:", [n for n, v in res.items() if v == "ask"])
EOF
```

**Expect** `{'skip': 24, 'CV MC': 20, 'ask': 3}` with the asks being `SBNDAr25`,
`SBND_SpringLowEMC`, `SBND_SpringMC_Nom`. Note that skip must beat CV in
precedence: a name matching both `MCOverlay` and a variation tag is a variation.

### Environment bootstrap

Only when `SKILL.md`'s setup section or `requirements.txt` changes. This downloads
packages and takes a few minutes. Run it when **no** sub-agent is active — they all
use `./venv/bin/python` and moving it mid-run breaks them.

```bash
mv venv venv.orig
./venv/bin/python -c "import numpy" 2>&1 | tail -1     # expect: No such file or directory
python3 -m venv venv && ./venv/bin/pip install -q -r requirements.txt
./venv/bin/python -c "import numpy, uproot, awkward, yaml" && echo ENV_OK
./venv/bin/python -m pytest tests/ -q                  # expect 75 passed
rm -rf venv && mv venv.orig venv                       # restore the known-good venv
./venv/bin/python -c "import uproot; print(uproot.__version__)"
```

## 8. Cleanup

```bash
rm -rf scratch output/scratch
```

Leave `output/sbn-rewgted-20/` alone unless you mean to discard it — it is a
complete, validated production (~1.6 GB), not scratch.

---

## Known caveat

A skill created or edited during a session does **not** resolve as
`/reweight-sbruce` in that session: the skill index is built at startup, so
`Skill(skill: "reweight-sbruce")` returns `Unknown skill` and agents fall back to
reading the file. That step can only be verified in a **fresh** session. Do that
once after any change to the frontmatter (`name`, `description`), since a
description that does not match how people phrase the request is the one failure
this checklist cannot catch.
