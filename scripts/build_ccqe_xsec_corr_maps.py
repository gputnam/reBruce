#!/usr/bin/env python3
"""Build data/ccqe_xsec_corr_sbn2_maps.root from the nusyst_data QEFix release.

The nusystematics CCQEXSecCorr tool (GENIE v3_04_02a -> v3_06_02 "sbn2" CCQE
bugfix) reads 4 flavors x 57 energies = 228 ROOT files from
$NUSYST_DATA_DIR/QEFix, published as the NuSystematics/nusyst_data GitHub
release assets QEFix.part01..04.tar.gz (5.7 GB). Each ~25 MB file is almost
entirely an `evt_weights` TTree; the tool reads only its `h_weights_map`
TH2D (x = q3, y = q0). This script streams the four tarballs (sha256-checked
against the release manifest.json), keeps each file's `h_weights_map`
(copied verbatim -- same bins, flow bins, entries) and its one-row `meta`
tree, and writes them to one self-contained file:

    <flavor>/E<energy:.2f>   TH2D   e.g. numu/E0.85   (flavor: nue, nuebar,
                                                       numu, numubar)
    meta                     TTree  one row per map (flavor, energy_GeV and
                                    the source files' meta columns)
    provenance               TObjString

Nothing is written to disk but the output unless --cache-dir is given (then
the tarballs are downloaded there once and reused).

Usage:  ./venv/bin/python scripts/build_ccqe_xsec_corr_maps.py \
            [--out data/ccqe_xsec_corr_sbn2_maps.root] [--cache-dir DIR]
"""

import argparse
import datetime
import hashlib
import io
import json
import os
import re
import tarfile
import urllib.request

import numpy as np
import uproot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_REPO = "NuSystematics/nusyst_data"
MANIFEST_URL = f"https://raw.githubusercontent.com/{DATA_REPO}/main/manifest.json"
ASSET_URL = f"https://github.com/{DATA_REPO}/releases/download/{{tag}}/{{file}}"
TOOL_URL = ("https://github.com/NuSystematics/nusystematics/blob/develop/"
            "src/nusystematics/systproviders/CCQEXSecCorr_tool.cc")
FCL_URL = ("https://github.com/NuSystematics/nusystematics/blob/develop/"
           "fcl/CCQEXSecCorr.ToolConfig.fcl")

FLAVORS = {"nue": 12, "nuebar": -12, "numu": 14, "numubar": -14}
# fcl/CCQEXSecCorr.ToolConfig.fcl EnergyGrid: 0.20 .. 3.00 in 0.05 steps
ENERGIES = [f"{0.20 + 0.05 * i:.2f}" for i in range(57)]
FNAME = re.compile(r"^reweight_data_v3_04_02a_to_v3_06_02_sbn2_CCQE_"
                   r"(nue|nuebar|numu|numubar)_(\d\.\d\d)GeV_flat\.root$")
META_COLS = ["pdg_nu", "pdg_lep", "n_bins", "n_src", "n_tgt", "energy_GeV",
             "fSF_src", "fSF_tgt", "xsec_ratio"]


class HashingReader(io.RawIOBase):
    """Wrap a byte stream, hashing everything read through it."""

    def __init__(self, raw):
        self.raw = raw
        self.sha = hashlib.sha256()
        self.nbytes = 0

    def readable(self):
        return True

    def readinto(self, b):
        data = self.raw.read(len(b))
        n = len(data)
        b[:n] = data
        self.sha.update(data)
        self.nbytes += n
        return n


def open_asset(tag, name, cache_dir):
    url = ASSET_URL.format(tag=tag, file=name)
    if cache_dir:
        path = os.path.join(cache_dir, name)
        if not os.path.exists(path):
            print(f"  downloading {url} -> {path}")
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
        return open(path, "rb")
    print(f"  streaming {url}")
    return urllib.request.urlopen(url)


def read_asset(asset, tag, cache_dir, maps, meta):
    """Stream one tarball; fill maps[(flavor, energy)] and meta rows."""
    with open_asset(tag, asset["file"], cache_dir) as raw:
        hr = HashingReader(raw)
        with tarfile.open(fileobj=io.BufferedReader(hr, 1 << 20),
                          mode="r|gz") as tar:
            for member in tar:
                m = FNAME.match(os.path.basename(member.name))
                if not member.isfile() or not m:
                    continue
                flavor, energy = m.groups()
                with uproot.open(io.BytesIO(tar.extractfile(member).read())) as f:
                    h = f["h_weights_map"]
                    row = {k: v[0] for k, v in
                           f["meta"].arrays(META_COLS, library="np").items()}
                if row["pdg_nu"] != FLAVORS[flavor] or \
                        f"{row['energy_GeV']:.2f}" != energy:
                    raise ValueError(f"{member.name}: meta {row} disagrees "
                                     "with the file name")
                if (flavor, energy) in maps:
                    raise ValueError(f"duplicate map {flavor} {energy}")
                maps[(flavor, energy)] = h
                meta[(flavor, energy)] = row
            # drain the gzip trailer so the hash covers the whole asset
            while hr.read(1 << 20):
                pass
    digest = hr.sha.hexdigest()
    if digest != asset["sha256"]:
        raise ValueError(f"{asset['file']}: sha256 {digest} != manifest "
                         f"{asset['sha256']}")
    print(f"    sha256 OK ({hr.nbytes} bytes), {len(maps)} maps so far")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(
        REPO, "data", "ccqe_xsec_corr_sbn2_maps.root"))
    ap.add_argument("--cache-dir", default=None,
                    help="download the tarballs here and reuse them")
    args = ap.parse_args()

    with urllib.request.urlopen(MANIFEST_URL) as r:
        manifest = json.load(r)
    tag = manifest["releaseTag"]
    assets = [a for a in manifest["assets"] if a["extract_to"] == "QEFix"]
    print(f"nusyst_data {tag}: {[a['file'] for a in assets]}")
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    maps, meta = {}, {}
    for asset in assets:
        read_asset(asset, tag, args.cache_dir, maps, meta)

    want = {(fl, e) for fl in FLAVORS for e in ENERGIES}
    if set(maps) != want:
        raise ValueError(f"missing maps: {sorted(want - set(maps))}; "
                         f"unexpected: {sorted(set(maps) - want)}")
    for key, h in maps.items():
        ax = [h.member("fXaxis"), h.member("fYaxis")]
        if [(a.member("fNbins"), a.member("fXmin"), a.member("fXmax"))
                for a in ax] != [(500, 0.0, 3.5)] * 2:
            raise ValueError(f"{key}: unexpected binning")

    keys = [(fl, e) for fl in FLAVORS for e in ENERGIES]
    provenance = "\n".join([
        "CCQE cross-section correction maps, GENIE v3_04_02a -> v3_06_02",
        "(sbn2 CCQE bugfix), for the reBruce `ccqe_xsec_corr` calculator.",
        "",
        f"Upstream tool:   {TOOL_URL}",
        f"Upstream config: {FCL_URL}",
        "  (file_pattern reweight_data_v3_04_02a_to_v3_06_02_sbn2_CCQE_"
        "%FLAVOR%_%ENERGY%GeV_flat.root, histogram_name h_weights_map,",
        "   EnergyGrid 0.20..3.00 step 0.05, WeightLimits [0, 30], EnuMin 0.20,"
        " EnuMax 3.00, EnuSnapTol 0.005, UseNearestBin true, EdgeClamp true)",
        f"Data: {DATA_REPO} release {tag}, extract_to QEFix:",
        *[f"  {a['file']}  sha256 {a['sha256']}" for a in assets],
        "",
        "<flavor>/E<energy> is the source file's h_weights_map copied "
        "verbatim (x = q3, y = q0 [GeV]); `meta` holds each source file's "
        "meta row.",
        f"Built {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d} by "
        "scripts/build_ccqe_xsec_corr_maps.py.",
    ])

    with uproot.recreate(args.out) as out:
        for fl, e in keys:
            out[f"{fl}/E{e}"] = maps[(fl, e)]
        cols = {c: np.array([meta[k][c] for k in keys]) for c in META_COLS}
        out["meta"] = {"flavor": np.array([FLAVORS[fl] for fl, _ in keys],
                                          dtype=np.int32), **cols}
        out["provenance"] = provenance
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, "
          f"{len(keys)} maps)")


if __name__ == "__main__":
    main()
