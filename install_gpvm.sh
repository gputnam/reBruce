#!/usr/bin/env bash
#
# install_gpvm.sh -- one-time environment build for reBruce on a FNAL
# GPVM / build machine (AlmaLinux 9, no container).
#
#   ./install_gpvm.sh [--force] [--no-tests]
#
# Sets up ROOT (with PyROOT) from cvmfs via spack, then builds an isolated
# python virtualenv (./venv) from that spack python and pip-installs
# requirements.txt into it.  Because `spack load` exposes ROOT through
# PYTHONPATH, the clean venv still resolves `import ROOT` while pip keeps full
# control of numpy / uproot / awkward / ...
#
# After this succeeds, use the environment in any shell with:
#   source setup_gpvm.sh
#
# Options:
#   --force      delete and rebuild an existing ./venv
#   --no-tests   skip the pytest gate at the end
#   -h, --help   show this help

set -uo pipefail

# --- shared configuration (keep in sync with setup_gpvm.sh) ------------------
SPACK_SETUP="/cvmfs/larsoft.opensciencegrid.org/spack-fnal-v1.1.1/setup-env.sh"
ROOT_SPEC="root@6.28.12"

REBRUCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
VENV_DIR="${REBRUCE_DIR}/venv"
REQUIREMENTS="${REBRUCE_DIR}/requirements.txt"

die() { echo "install_gpvm.sh: $*" >&2; exit 1; }

# --- args --------------------------------------------------------------------
FORCE=0
RUN_TESTS=1
for arg in "$@"; do
    case "$arg" in
        --force)    FORCE=1 ;;
        --no-tests) RUN_TESTS=0 ;;
        -h|--help)
            # print the leading comment block (after the shebang), stripping "# "
            awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) die "unknown option '$arg' (try --help)" ;;
    esac
done

[ -f "$REQUIREMENTS" ] || die "requirements.txt not found at $REQUIREMENTS"

# --- ROOT from cvmfs via spack -----------------------------------------------
[ -f "$SPACK_SETUP" ] || die "spack setup not found at $SPACK_SETUP
  Is /cvmfs/larsoft.opensciencegrid.org mounted on this machine?"

echo ">> Setting up ROOT ($ROOT_SPEC) from cvmfs via spack ..."
# spack's setup script is not '-u' clean; relax it across the source + load.
set +u
# shellcheck disable=SC1090
source "$SPACK_SETUP"
# --first picks one of several equivalent almalinux9/python-3.11 root builds.
# NB: never pipe `spack load` -- a pipe subshells it and the env change is lost.
spack load --first "$ROOT_SPEC"
load_rc=$?
set -u
[ "$load_rc" -eq 0 ] || die "'spack load $ROOT_SPEC' failed"
command -v root-config >/dev/null 2>&1 || die "root-config not on PATH after spack load"

SPACK_PYTHON="$(command -v python3)"
echo "   ROOT    $(root-config --version)"
echo "   python  $("$SPACK_PYTHON" --version 2>&1 | awk '{print $2}')   ($SPACK_PYTHON)"

# --- build the venv ----------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
    if [ "$FORCE" -eq 1 ]; then
        echo ">> Removing existing venv (--force): $VENV_DIR"
        rm -rf "$VENV_DIR"
    else
        echo ">> Reusing existing venv: $VENV_DIR   (use --force to rebuild)"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo ">> Creating isolated venv from spack python: $VENV_DIR"
    "$SPACK_PYTHON" -m venv "$VENV_DIR" || die "failed to create venv"
fi

VPY="${VENV_DIR}/bin/python"
[ -x "$VPY" ] || die "venv python missing at $VPY"

echo ">> Upgrading pip and installing requirements ..."
"$VPY" -m pip install --upgrade pip || die "pip upgrade failed"
"$VPY" -m pip install -r "$REQUIREMENTS" || die "pip install -r requirements.txt failed"

# --- verify ------------------------------------------------------------------
echo ">> Verifying imports (venv packages + PyROOT) ..."
"$VPY" - <<'PY' || die "import verification failed"
import importlib
mods = ["numpy", "uproot", "awkward", "yaml", "matplotlib", "ROOT"]
for m in mods:
    mod = importlib.import_module(m)
    v = getattr(mod, "__version__", None)
    if m == "ROOT":
        v = mod.gROOT.GetVersion()
    print(f"   {m:12s} {v}")
print("   all imports OK")
PY

# --- pytest gate -------------------------------------------------------------
if [ "$RUN_TESTS" -eq 1 ]; then
    echo ">> Running test suite (pytest) ..."
    ( cd "$REBRUCE_DIR" && "$VPY" -m pytest tests/ -q ) || die "pytest failed"
else
    echo ">> Skipping tests (--no-tests)."
fi

echo ""
echo "reBruce environment installed successfully."
echo "To use it in any shell:"
echo "    source ${REBRUCE_DIR}/setup_gpvm.sh"
