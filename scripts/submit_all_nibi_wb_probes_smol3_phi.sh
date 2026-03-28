#!/bin/bash
# Wrapper: submit all Nibi WB probe jobs with fixed
#   RESPONSE_HF=HuggingFaceTB/SmolLM3-3B
#   VERIFIER_HF=microsoft/Phi-4-mini-instruct

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BASE_SCRIPT="${PROJECT_ROOT}/scripts/submit_all_nibi_wb_probes.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "ERROR: missing base submit script: ${BASE_SCRIPT}" >&2
  exit 1
fi

export RESPONSE_HF="${RESPONSE_HF:-HuggingFaceTB/SmolLM3-3B}"
export VERIFIER_HF="${VERIFIER_HF:-microsoft/Phi-4-mini-instruct}"

exec bash "${BASE_SCRIPT}" "$@"
