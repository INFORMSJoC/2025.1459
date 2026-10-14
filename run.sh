#!/usr/bin/env bash
set -euo pipefail
HPR_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export JULIA_LOAD_PATH='@:@stdlib'
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset LD_LIBRARY_PATH
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HPR_ROOT/.julia-depot}"
HPR_PYTHON=${PYTHON_BIN:-python3}
HPR_ACTION=${1:-list}
if [ "$#" -gt 0 ]; then shift; fi
case "$HPR_ACTION" in
  check)
    "$HPR_PYTHON" "$HPR_ROOT/scripts/verify_package.py"
    "$HPR_PYTHON" "$HPR_ROOT/scripts/check_scope.py"
    exec "$HPR_PYTHON" "$HPR_ROOT/scripts/manage_data.py" verify --bundled --include-competitors "$@"
    ;;
  tables) exec "$HPR_PYTHON" "$HPR_ROOT/scripts/postprocess.py" --no-plots "$@" ;;
  figures) exec "$HPR_PYTHON" "$HPR_ROOT/scripts/postprocess.py" "$@" ;;
  help|-h|--help)
    "$HPR_PYTHON" "$HPR_ROOT/scripts/reviewer_run.py" list
    cat <<'HELP'
  ./run.sh setup SOLVER                    Install the selected pinned environment
  ./run.sh check                           Check package, references and bundled data
  ./run.sh QAPLIB HPR-QP                    Run the complete QAPLIB dataset
  ./run.sh MM HPR-QP --cases CVXQP3_S.mps QSCSD1.mps
  ./run.sh ALL ALL --data-root /path/to/QP_data --plan
  ./run.sh refresh                         Refresh experiment summaries
  ./run.sh tables                          Regenerate tables (corrected synthetic HPR; other historical references)
  ./run.sh figures                         Regenerate grayscale paper figures
Main results: experiments/<name>/summary.html and summary.csv.
Full ALL can take days to weeks. See README.md for experiment/solver mappings.
Use --help after an experiment for device, budget, data and output options.
HELP
    ;;
  *) exec "$HPR_PYTHON" "$HPR_ROOT/scripts/reviewer_run.py" "$HPR_ACTION" "$@" ;;
esac
