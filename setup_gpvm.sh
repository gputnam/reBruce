# shellcheck shell=bash
#
# setup_gpvm.sh -- configure the current shell to run reBruce on a FNAL
# GPVM / build machine (AlmaLinux 9, no container).
#
#   source setup_gpvm.sh
#
# This must be *sourced*, not executed: it edits PATH / PYTHONPATH in place.
# It sets up ROOT (with PyROOT) from cvmfs via spack and activates the local
# virtualenv built by ./install_gpvm.sh.  ROOT is exposed through PYTHONPATH by
# `spack load`, so the isolated venv still resolves `import ROOT` while pip keeps
# full control of numpy / uproot / awkward / ...
#
# Run ./install_gpvm.sh once first to create the venv.

# --- shared configuration (keep in sync with install_gpvm.sh) ----------------
SPACK_SETUP="/cvmfs/larsoft.opensciencegrid.org/spack-fnal-v1.1.1/setup-env.sh"
ROOT_SPEC="root@6.28.12"

# Resolve the directory this file lives in, whether sourced from bash or zsh.
if [ -n "${BASH_SOURCE:-}" ]; then
    _reBruce_src="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _reBruce_src="${(%):-%N}"
else
    _reBruce_src="$0"
fi
REBRUCE_DIR="$(cd "$(dirname "$_reBruce_src")" >/dev/null 2>&1 && pwd)"
VENV_DIR="${REBRUCE_DIR}/venv"
unset _reBruce_src

# --- guard: must be sourced ---------------------------------------------------
# If $0 is this script (rather than the shell), it was executed, not sourced.
case "$0" in
    *setup_gpvm.sh)
        echo "setup_gpvm.sh: this script must be SOURCED, not executed --" >&2
        echo "  its PATH/PYTHONPATH changes are discarded in a child shell." >&2
        echo "  Run:  source setup_gpvm.sh" >&2
        exit 1
        ;;
esac

# --- ROOT from cvmfs via spack ------------------------------------------------
if [ ! -f "$SPACK_SETUP" ]; then
    echo "setup_gpvm.sh: cannot find spack setup at" >&2
    echo "  $SPACK_SETUP" >&2
    echo "  Is /cvmfs/larsoft.opensciencegrid.org mounted on this machine?" >&2
    return 1
fi

# spack's setup script trips `set -u`/`set -e`; relax them across the source
# and the load, then restore the caller's options.
_reBruce_had_u=$(case $- in *u*) echo 1;; *) echo 0;; esac)
_reBruce_had_e=$(case $- in *e*) echo 1;; *) echo 0;; esac)
set +u +e

# shellcheck disable=SC1090
source "$SPACK_SETUP"

# `root@6.28.12` matches several equivalent almalinux9 / python-3.11 builds;
# --first picks one deterministically (they are functionally identical).
# NB: never pipe `spack load` -- a pipe runs it in a subshell and the
# environment changes are lost.
spack load --first "$ROOT_SPEC"
_reBruce_load_rc=$?

[ "$_reBruce_had_u" = 1 ] && set -u
[ "$_reBruce_had_e" = 1 ] && set -e
unset _reBruce_had_u _reBruce_had_e

if [ "$_reBruce_load_rc" -ne 0 ] || ! command -v root-config >/dev/null 2>&1; then
    echo "setup_gpvm.sh: 'spack load $ROOT_SPEC' failed." >&2
    unset _reBruce_load_rc
    return 1
fi
unset _reBruce_load_rc

# --- virtualenv ---------------------------------------------------------------
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
else
    echo "setup_gpvm.sh: no venv at ${VENV_DIR}." >&2
    echo "  ROOT is set up, but run ./install_gpvm.sh first to build the venv." >&2
    return 1
fi

# --- sanity check -------------------------------------------------------------
if python -c "import ROOT" >/dev/null 2>&1; then
    echo "reBruce environment ready:"
    echo "  ROOT    $(root-config --version)   (PyROOT: $(python -c 'import ROOT; print(ROOT.gROOT.GetVersion())'))"
    echo "  python  $(python --version 2>&1 | awk '{print $2}')   ($(command -v python))"
else
    echo "setup_gpvm.sh: venv active but 'import ROOT' failed -- ROOT/venv mismatch?" >&2
    return 1
fi
