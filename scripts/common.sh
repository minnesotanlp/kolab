# Sourced by the numbered scripts: repo paths, data folder, python, and a command echo.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$REPO_DIR/code"
export KOLAB_DATA_DIR="${KOLAB_DATA_DIR:-$REPO_DIR/data}"   # see code/config.py
PYTHON="${PYTHON:-python3}"
DRY_RUN=0

trap 'echo; echo "!!! Stopped: the command above failed. Fix it and re-run: finished work is skipped automatically." >&2' ERR

# run <command...>: print it, then run it (print only with --dry-run).
run() {
    echo "  \$ ${*/#$PYTHON/python}"
    [[ $DRY_RUN -eq 1 ]] || "$@"
}

# Print the header comment of the calling script (everything up to "set -euo" / "source").
usage() { sed -n '2,/^source /p' "$0" | sed 's/^# \{0,1\}//' | head -n -2; }
