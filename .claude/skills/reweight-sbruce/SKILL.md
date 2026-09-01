---
name: reweight-sbruce
description: Run reBruce over a folder of sBruce files - work out which files are central-value MC, set up the venv, preflight every calculator's required branches, apply the fake-data weights and report what was written. Use when asked to run reBruce, reweight or process a folder / production directory of sBruce files, apply fake-data weights, or produce fakedataTree outputs for an sbn-rewgted-NN sample.
---

# Running a production directory through reBruce

reBruce evaluates fake-data weight calculators on the truth / GENIE-pre-FSI content
of an sBruce file and writes them as a `fakedataTree` friend tree in a copy of the
input. `reweight.py` handles one file at a time; this skill drives it over a whole
production directory and does the judgement work around it.

Run everything from the repository root
(`/Users/gputnam/Work/osc/fake-data-reweighter` unless the user is elsewhere).

## Arguments

Parse the skill's free-form argument string as:

- **first positional** — the input folder. Required. If absent, ask for it.
- **`--config PATH`** — defaults to `configs/all_calculators.yaml` (all 9
  calculators, all W modes; this is what the downstream analysis expects).
- **`--output DIR`** — defaults to `output/<basename of the input folder>/`.
- **everything else** — free-form context, to be honoured as instructions. It
  overrides the defaults and the classification below. Typical: "SBND only",
  "skip minerva", "also do the dirt files", "set WEIGHT_CLIP to 50",
  "just check the branches, don't run".

Report the four resolved values before doing anything.

## 1. Discover — do this first

Discovery comes before environment setup: there is no point spending minutes
building a virtualenv for a folder with nothing to process.

sBruce files are ROOT files, conventionally named `*_sbruce.root`. **Treat every
`*.root` file as a candidate** — the `_sbruce` suffix is a convention, not a
guarantee — and let the content checks below decide. Summarise the folder rather
than listing it file by file:

```bash
ls -1 <folder> | sed 's/.*\.//' | sort | uniq -c   # what extensions are here
ls -1 <folder>/*.root 2>/dev/null | wc -l          # candidate count
```

**If there are no `*.root` files, stop here. Do not go on to step 2.** Report what
the folder actually holds, and check the type of a representative file so the
message says *why* it is unusable rather than merely that it is:

```bash
head -c 8 <folder>/<a file> | od -c | head -1      # \211 H D F = HDF5, root = ROOT
```

This is a case worth expecting, not an edge case. A production's GUMPLE pandas/HDF5
`.df` dataframes and its sBruce ROOT files live in **separate directories** with
similar names, and reBruce reads only the latter — so a folder full of `.df` files
means the user pointed at the wrong one of the pair. Say so, name the sibling
directory convention (compare `sbn-rewgted-20` under a `-sBruce` parent), and offer
to run once they point at it. Never attempt a conversion.

## 2. Environment

reBruce runs from a local virtualenv; never use the system `python3` to run
`reweight.py`.

```bash
./venv/bin/python -c "import numpy, uproot, awkward, yaml" && echo ENV_OK
```

If that fails (missing `venv/`, or an incomplete one), create it — this is the
README's Setup block — then re-check, and confirm with the unit tests:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m pytest tests/ -q
```

Say which path you took. On an already-working venv the tests are optional.

## 3. Classify

Only central-value neutrino MC gets fake-data weights. Everything else either has
no GENIE truth to weight or is handled separately, and its weights would be 1.0.

First read each candidate, so an unreadable file never reaches the run loop:

```bash
./venv/bin/python -c "
import os, sys, uproot
for p in sys.argv[1:]:
    try:
        print(f'{os.path.basename(p):48s} {uproot.open(p)[\"SelectedEvents\"].num_entries:>8}')
    except Exception as e:
        print(f'{os.path.basename(p):48s} UNREADABLE  {type(e).__name__}: {str(e).splitlines()[0]}')
" <folder>/*.root
```

Then sort each candidate into one of four classes. **The rules are in precedence
order — the first one that matches wins.** In particular `skip` beats `CV MC`: a
detector variation named `..._SpringMCOverlay_..._SCE_sbruce.root` matches both, and
it is a variation, not CV.

| # | class | rule | examples |
|---|---|---|---|
| 1 | **unreadable** | the command above could not open it or find `SelectedEvents` | an HDF5 `.df` renamed `.root` |
| 2 | **skip** | off-beam / on-beam data (`BNBOff`, `OffData`, `unblind`, `Data`, `OnBeam`), dirt (`Dirt`), in-time cosmics (`Intime`), or a detector variation (`SCE`, `DENT`, or any `WM...` wire-modification tag: `WMYZ`, `WMNom`, `WMXThXW`, `WMXThetaXW`) | `SBND_SpringBNBOffData_sbruce.root`, `ICARUSRun2_Spring_Overlay_Dirt_sbruce.root`, `SBND_SpringMC_2xSCE_sbruce.root` |
| 3 | **CV MC** — process | name contains `MCCV` or `MCOverlay` | `SBNDMCCV_7_sbruce.root`, `ICARUSRun4_SpringMCOverlay_rewgt_2_sbruce.root` |
| 4 | **ask** | anything else readable | `SBND_SpringLowEMC_sbruce.root` |

The CV rule reproduces the existing production exactly and matches the globs in
`fake-data-reweighter-ana/config.py` (`SAMPLES`), which is what consumes the output.
The pattern lists are a starting point, not law — check them against the actual
names in the folder and say so if something does not fit. Two families worth
knowing: `ICARUSRun{2,4}_Spring_Overlay_*` and `SBND_SpringMC_*` are both
detector-variation sets, so their members skip even where the tag is unfamiliar --
including their own nominal reference (`SBND_SpringMC_Nom`), which is the CV *of
that variation set*, not the analysis CV.

Print the classification as a table — file, entries, class, reason — **before**
running anything.

- **unreadable** files are reported and excluded. Do not ask the user about them:
  nothing they can decide makes an unreadable file processable.
- For **ask** files, say what evidence you have before asking. A quick content check
  distinguishes "MC with truth" from "data-like" better than the name does:
  ```bash
  ./venv/bin/python -c "
  import uproot, numpy as np
  t = uproot.open('<file>')['SelectedEvents']
  if 'genie_mode' not in t: print('no genie_mode branch -> data-like, not reweightable')
  else:
      m = t['genie_mode'].array(library='np')
      print(f'genie_mode filled in {100*np.mean(m > -900):.1f}% of entries')
  "
  ```
  Present the ask files with that evidence and take the user's direction before
  running them. Free-form context from the arguments overrides any row.

## 4. Preflight the branches

Calculators declare exactly which `SelectedEvents` branches they read
(`branches_needed()`), and `reweight.py` checks them before doing any work:

```bash
./venv/bin/python reweight.py <config> --input <file> --check-branches
```

Exit 0 means every configured calculator can run. Exit 1 means either that some
branches are missing — the report names them, the calculators each one blocks, and
the calculators that would still run — or that the file could not be read at all.

Run this on **every** selected file before running any of them. Then:

- **Complete files: process them.** One incomplete file must not block the rest of
  the production.
- **Incomplete files: hold them back and ask.** Name the file, the missing branches,
  and the blocked calculators. Usually the cause is the wrong sample or an older
  sBruce schema. The three ways out:
  - fix the input (best — the file is normally the wrong one);
  - re-run just those files with `--skip-incomplete`, which drops the blocked
    calculators and runs the rest. **Say explicitly that the output will then carry
    fewer weight branches than its siblings** — that matters downstream, and it
    makes `validation/check_outputs.py` fail with a `KeyError` on the absent
    branches, so those files must be excluded from step 6 entirely;
  - drop those files.

Proceed without asking only when the user's free-form context already told you which
of these they want.

## 5. Run

Sequentially, one file at a time. Redirect each file's output to a log rather than
`tee`-ing it — 20 files of per-calculator coverage floods the transcript — and skip
files whose output already exists, so a re-invocation resumes rather than redoing
work:

```bash
mkdir -p <outdir>/logs
for f in <selected files>; do
  base=$(basename "$f" .root)
  out=<outdir>/${base}_fakedata.root
  [ -f "$out" ] && { echo "skip (exists): $out"; continue; }
  if ./venv/bin/python reweight.py <config> --input "$f" --output "$out" \
       > <outdir>/logs/${base}.log 2>&1; then
    echo "ok:   $base"
  else
    echo "FAIL: $base"; tail -20 <outdir>/logs/${base}.log
  fi
done
```

Add `--skip-incomplete` to that command line for the files the user chose to run
that way, and keep them in a separate pass so it is obvious which outputs are
reduced.

Pass `--output` explicitly as shown. Without it `reweight.py` falls back to the
config's `output_dir` (flat `output/`), not to this skill's per-folder default.

Files are fast — well under a second each for a typical ~15k-event sBruce file, so a
20-file production takes seconds. Run it in the foreground.

Afterwards, scan the logs rather than the transcript:

```bash
grep -h "clipped" <outdir>/logs/*.log | sort | uniq -c | sort -rn | head
```

Surface a large clip count. To change the weight clip (only if asked), set
`fakedata.calculator.WEIGHT_CLIP` before running; it is a module-level constant, not
a config key. See the README's "Weight clip" section.

## 6. Validate

If the default config was used, run the post-production checks and include the
table in your report:

```bash
./venv/bin/python validation/check_outputs.py '<outdir>/*_fakedata.root'
```

It verifies entry-count match, finite non-negative weights, per-calculator coverage
and W-mode closure. It hard-codes the default config's branch list, so it only works
for that config on files where every calculator ran. Skip it — and say why — for any
other config, and exclude any `--skip-incomplete` outputs from the glob, or it will
raise a `KeyError` rather than degrade.

## 7. Report

Close with:

- files processed, and where the outputs are;
- files skipped, and the reason for each class, including any unreadable ones;
- the weight branches written — state the count and confirm every file agrees,
  listing the names **once** for the set rather than per file (it is a 23-element
  list for the default config);
- the `check_outputs.py` verdict;
- anything left for the user to decide — `ask` files, incomplete files, large clip
  counts.
