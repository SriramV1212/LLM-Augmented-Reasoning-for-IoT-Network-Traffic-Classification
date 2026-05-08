#!/usr/bin/env bash
#
# Full eval workflow (notebook test fold → balanced 4k → ML + zero-shot evaluate).
# Run from repo root, or anywhere:
#   nohup bash scripts/run_eval_suite.sh >> logs/eval_suite.log 2>&1 &
#
# Override defaults via env:
#   MODEL=openai/gpt-oss-120b:free EXPLAIN_MAX_ML=500 bash scripts/run_eval_suite.sh
#   SKIP_EXPORT=1 SKIP_SAMPLE=1     — reuse existing CSVs only
#   SKIP_ML_EXPLAINER=1 SKIP_ZS=1   — metrics-only ML + skip all OpenRouter calls
#
# Loads repo .env automatically (same as interactive shells with dotenv) so nohup sees OPENROUTER_*.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() {
  echo "[$(date -Iseconds)] $*"
}

ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  log "Loading environment from $ENV_FILE"
  set -a
  # shellcheck disable=SC1091
  source "$ENV_FILE"
  set +a
else
  log "No $ENV_FILE found (export OPENROUTER_API_KEY yourself if using LLM steps)"
fi

PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-data}"
MODELS_DIR="${MODELS_DIR:-models}"
MODEL="${MODEL:-${OPENROUTER_DEFAULT_MODEL:-tencent/hy3-preview:free}}"
PER_CLASS="${PER_CLASS:-1000}"

TEST_CSV="${TEST_CSV:-${DATA_DIR}/test_split_eval.csv}"
SUBSET_CSV="${SUBSET_CSV:-${DATA_DIR}/eval_balanced_4k.csv}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_SAMPLE="${SKIP_SAMPLE:-0}"
SKIP_ML_METRICS="${SKIP_ML_METRICS:-0}"
SKIP_ML_EXPLAINER="${SKIP_ML_EXPLAINER:-0}"
SKIP_ZS_METRICS="${SKIP_ZS_METRICS:-0}"
SKIP_ZS_NARRATIVES="${SKIP_ZS_NARRATIVES:-0}"

EXPLAIN_MAX_ML="${EXPLAIN_MAX_ML:-500}"
EXPLAIN_MAX_ZS="${EXPLAIN_MAX_ZS:-100}"

OUT_ML_METRICS="${OUT_ML_METRICS:-${DATA_DIR}/eval_ml_4k_metrics.json}"
OUT_ML_EXPL="${OUT_ML_EXPL:-${DATA_DIR}/eval_ml_4k_explainer.json}"
OUT_ZS_METRICS="${OUT_ZS_METRICS:-${DATA_DIR}/eval_zs_4k_metrics.json}"
OUT_ZS_NARR="${OUT_ZS_NARR:-${DATA_DIR}/eval_zs_4k_narratives.json}"

have_api_key() {
  [[ -n "${OPENROUTER_API_KEY:-}" ]]
}

run() {
  log "RUN $*"
  "$@"
}

log "Starting eval suite (ROOT=$ROOT MODEL=$MODEL PER_CLASS=$PER_CLASS)"

if [[ "$SKIP_EXPORT" != "1" ]]; then
  run "$PYTHON" cli.py export-test-split --data-dir "$DATA_DIR" -o "$TEST_CSV" --models-dir "$MODELS_DIR"
else
  log "SKIP_EXPORT=1 — not running export-test-split"
fi

if [[ "$SKIP_SAMPLE" != "1" ]]; then
  run "$PYTHON" cli.py sample-eval-subset -i "$TEST_CSV" -o "$SUBSET_CSV" \
    --per-class "$PER_CLASS" --models-dir "$MODELS_DIR"
else
  log "SKIP_SAMPLE=1 — not running sample-eval-subset"
fi

if [[ "$SKIP_ML_METRICS" != "1" ]]; then
  run "$PYTHON" cli.py evaluate -d "$SUBSET_CSV" -o "$OUT_ML_METRICS" \
    --mode ml-agent --no-llm-explainer
else
  log "SKIP_ML_METRICS=1"
fi

if [[ "$SKIP_ML_EXPLAINER" != "1" ]]; then
  if have_api_key; then
    run "$PYTHON" cli.py evaluate -d "$SUBSET_CSV" -o "$OUT_ML_EXPL" \
      --mode ml-agent --explain-max-rows "$EXPLAIN_MAX_ML" --model "$MODEL"
  else
    log "SKIP ML explainer: OPENROUTER_API_KEY unset"
  fi
else
  log "SKIP_ML_EXPLAINER=1"
fi

if [[ "$SKIP_ZS_METRICS" != "1" ]]; then
  if have_api_key; then
    run "$PYTHON" cli.py evaluate -d "$SUBSET_CSV" -o "$OUT_ZS_METRICS" \
      --mode zeroshot --no-llm-explainer --model "$MODEL"
  else
    log "SKIP zero-shot metrics: OPENROUTER_API_KEY unset"
  fi
else
  log "SKIP_ZS_METRICS=1"
fi

if [[ "$SKIP_ZS_NARRATIVES" != "1" ]]; then
  if have_api_key; then
    run "$PYTHON" cli.py evaluate -d "$SUBSET_CSV" -o "$OUT_ZS_NARR" \
      --mode zeroshot --explain-max-rows "$EXPLAIN_MAX_ZS" --model "$MODEL"
  else
    log "SKIP zero-shot narratives: OPENROUTER_API_KEY unset"
  fi
else
  log "SKIP_ZS_NARRATIVES=1"
fi

log "Done."
log "  subset:           $SUBSET_CSV"
log "  ml metrics:       $OUT_ML_METRICS"
log "  ml explainer:     $OUT_ML_EXPL"
log "  zeroshot metrics: $OUT_ZS_METRICS"
log "  zeroshot narr.:   $OUT_ZS_NARR"
