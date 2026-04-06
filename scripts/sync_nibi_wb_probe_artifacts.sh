#!/bin/bash
# Sync white-box probe artifacts from Nibi to local machine.
# Includes:
#   - /scratch/.../wb_probe_out run outputs
#   - /home/.../slurm_wb_probe_*.out/.err logs
#   - sacct snapshots for wb_probe jobs (if available)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE="${REMOTE:-ssimran5@nibi.alliancecan.ca}"
REMOTE_HOME="${REMOTE_HOME:-/home/ssimran5}"
REMOTE_WB_OUT="${REMOTE_WB_OUT:-/scratch/ssimran5/LLM_Hallucination_Measure/wb_probe_out}"
REMOTE_WB_OUT_FALLBACK="${REMOTE_WB_OUT_FALLBACK:-/home/ssimran5/LLM_Hallucination_Measure/data/results/analysis/final_analysis_ready/whitebox/wb_cross_model_probe_emnlp2025}"
LOCAL_DEST="${LOCAL_DEST:-${PROJECT_ROOT}/downloads/nibi_wb_probe}"
WATCH_SECONDS=0
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-20}"
SSH_CONTROL_PERSIST="${SSH_CONTROL_PERSIST:-30m}"
SSH_CONTROL_DIR="${SSH_CONTROL_DIR:-/tmp/wb_probe_ssh_mux}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"
SUMMARY_SCRIPT=""
SUMMARY_OUTPUT_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sync_nibi_wb_probe_artifacts.sh [options]

Options:
  --remote USER@HOST           Remote SSH target (default: ssimran5@nibi.alliancecan.ca)
  --remote-home PATH           Remote home path for slurm logs (default: /home/ssimran5)
  --remote-wb-out PATH         Remote scratch wb output path
                               (default: /scratch/ssimran5/LLM_Hallucination_Measure/wb_probe_out)
  --remote-wb-out-fallback PATH
                               Secondary wb output path under home/project
  --local-dest PATH            Local destination directory
                               (default: <repo>/downloads/nibi_wb_probe)
  --watch-seconds N            Repeat sync every N seconds (default: one-shot)
  --skip-summary               Disable local post-sync summary generation
  --summary-script PATH        Override summary script path
  --summary-output-dir PATH    Override summary output directory
  -h, --help                   Show this message

Examples:
  bash scripts/sync_nibi_wb_probe_artifacts.sh
  bash scripts/sync_nibi_wb_probe_artifacts.sh --watch-seconds 300
  REMOTE=ssimran5@nibi.alliancecan.ca bash scripts/sync_nibi_wb_probe_artifacts.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --remote-home)
      REMOTE_HOME="$2"
      shift 2
      ;;
    --remote-wb-out)
      REMOTE_WB_OUT="$2"
      shift 2
      ;;
    --remote-wb-out-fallback)
      REMOTE_WB_OUT_FALLBACK="$2"
      shift 2
      ;;
    --local-dest)
      LOCAL_DEST="$2"
      shift 2
      ;;
    --watch-seconds)
      WATCH_SECONDS="$2"
      shift 2
      ;;
    --skip-summary)
      SKIP_SUMMARY=1
      shift
      ;;
    --summary-script)
      SUMMARY_SCRIPT="$2"
      shift 2
      ;;
    --summary-output-dir)
      SUMMARY_OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "$WATCH_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --watch-seconds must be a non-negative integer." >&2
  exit 1
fi

if ! [[ "$SKIP_SUMMARY" =~ ^[01]$ ]]; then
  echo "ERROR: --skip-summary expects no value; use flag only." >&2
  exit 1
fi

if [[ -z "$SUMMARY_SCRIPT" ]]; then
  SUMMARY_SCRIPT="${PROJECT_ROOT}/scripts/summarize_synced_wb_probe_runs.py"
fi
if [[ -z "$SUMMARY_OUTPUT_DIR" ]]; then
  SUMMARY_OUTPUT_DIR="${LOCAL_DEST}/summaries"
fi

mkdir -p "${LOCAL_DEST}/slurm_logs" "${LOCAL_DEST}/wb_probe_out" "${LOCAL_DEST}/job_history"
mkdir -p "${SSH_CONTROL_DIR}"
chmod 700 "${SSH_CONTROL_DIR}" >/dev/null 2>&1 || true

REMOTE_USER="${REMOTE%@*}"
SSH_CONTROL_PATH="${SSH_CONTROL_DIR}/%C"
SSH_OPTS=(
  -o "ControlMaster=auto"
  -o "ControlPersist=${SSH_CONTROL_PERSIST}"
  -o "ControlPath=${SSH_CONTROL_PATH}"
  -o "StreamLocalBindUnlink=yes"
  -o "ConnectTimeout=${SSH_CONNECT_TIMEOUT}"
)
RSYNC_RSH="ssh -o ControlMaster=auto -o ControlPersist=${SSH_CONTROL_PERSIST} -o ControlPath=${SSH_CONTROL_PATH} -o StreamLocalBindUnlink=yes -o ConnectTimeout=${SSH_CONNECT_TIMEOUT}"

ensure_master_connection() {
  if ssh "${SSH_OPTS[@]}" -O check "${REMOTE}" >/dev/null 2>&1; then
    return 0
  fi
  echo "[sync] opening persistent SSH connection to ${REMOTE}"
  ssh "${SSH_OPTS[@]}" -MNf "${REMOTE}"
}

close_master_connection() {
  ssh "${SSH_OPTS[@]}" -O exit "${REMOTE}" >/dev/null 2>&1 || true
}

trap close_master_connection EXIT

sync_once() {
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"

  ensure_master_connection
  echo "[sync] ${ts} connected to ${REMOTE}"

  echo "[sync] pulling slurm logs from ${REMOTE_HOME}"
  rsync -avz -e "${RSYNC_RSH}" \
    --prune-empty-dirs \
    --include='slurm_wb_probe_*.out' \
    --include='slurm_wb_probe_*.err' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_HOME}/" \
    "${LOCAL_DEST}/slurm_logs/"

  if ssh "${SSH_OPTS[@]}" "${REMOTE}" "[ -d '${REMOTE_WB_OUT}' ]"; then
    echo "[sync] pulling wb outputs from ${REMOTE_WB_OUT}"
    rsync -avz -e "${RSYNC_RSH}" --delete \
      "${REMOTE}:${REMOTE_WB_OUT}/" \
      "${LOCAL_DEST}/wb_probe_out/"
  else
    echo "[sync] primary wb output path missing: ${REMOTE_WB_OUT}" >&2
  fi

  if ssh "${SSH_OPTS[@]}" "${REMOTE}" "[ -d '${REMOTE_WB_OUT_FALLBACK}' ]"; then
    echo "[sync] pulling fallback wb outputs from ${REMOTE_WB_OUT_FALLBACK}"
    rsync -avz -e "${RSYNC_RSH}" \
      "${REMOTE}:${REMOTE_WB_OUT_FALLBACK}/" \
      "${LOCAL_DEST}/wb_probe_out_fallback/"
  fi

  if ssh "${SSH_OPTS[@]}" "${REMOTE}" "command -v sacct >/dev/null 2>&1"; then
    echo "[sync] saving sacct snapshot"
    ssh "${SSH_OPTS[@]}" "${REMOTE}" \
      "sacct -u '${REMOTE_USER}' --name=wb_probe --format=JobIDRaw,JobName,State,ExitCode,Start,End,Elapsed,Reason,WorkDir,StdOut,StdErr -P" \
      > "${LOCAL_DEST}/job_history/sacct_wb_probe_${ts}.csv" || true
    cp -f "${LOCAL_DEST}/job_history/sacct_wb_probe_${ts}.csv" "${LOCAL_DEST}/job_history/sacct_wb_probe_latest.csv" || true
  else
    echo "[sync] sacct unavailable on remote; skipping job-history snapshot"
  fi

  if [[ "$SKIP_SUMMARY" -eq 0 ]]; then
    if [[ -f "${SUMMARY_SCRIPT}" ]]; then
      echo "[sync] updating local summary tables"
      python3 "${SUMMARY_SCRIPT}" \
        --artifacts-root "${LOCAL_DEST}" \
        --output-dir "${SUMMARY_OUTPUT_DIR}" || echo "[sync] warning: summary generation failed" >&2
    else
      echo "[sync] warning: summary script not found: ${SUMMARY_SCRIPT}" >&2
    fi
  fi

  echo "[sync] done: ${LOCAL_DEST}"
}

if [[ "$WATCH_SECONDS" -eq 0 ]]; then
  sync_once
  exit 0
fi

echo "[watch] running continuous sync every ${WATCH_SECONDS}s"
while true; do
  sync_once || echo "[watch] sync failed; retrying after ${WATCH_SECONDS}s" >&2
  sleep "$WATCH_SECONDS"
done
