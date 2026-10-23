#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
stage="${1:-preflight}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  preflight|audit_model|cache_baseline|run_completion|evaluate|sweep)
    exec python -m "zsdc.scripts.${stage}" --config "${ZSDC_CONFIG:-configs/zsdc_base.yaml}" "$@"
    ;;
  synthetic_smoke)
    exec python -m zsdc.scripts.synthetic_smoke "$@"
    ;;
  *) echo "Usage: bash launch_zsdc.sh {preflight|audit_model|cache_baseline|run_completion|evaluate|sweep|synthetic_smoke} [options]" >&2; exit 2 ;;
esac
