#!/usr/bin/env bash
# Run the full pipeline, unattended, on the gold papers.
#
#   ./run_pipeline.sh prepare                      # once: enrol, metadata, retrieve
#   ./run_pipeline.sh classify claude-opus-5-5     # per model: classify, verify, export
#   ESCALATE=1 ./run_pipeline.sh classify claude-haiku-4-5   # with tier escalation
#
# `prepare` retrieves every gold paper ONCE into runs/retrieved.sqlite and writes
# the exact full text each passage set was cut from to texts/ (DAP_TEXT_CACHE).
# Every `classify` run starts from a copy of that database, and baseline.py
# reads texts/, so every system in the comparison sees the same copy of every
# paper. texts/ holds publisher full text: keep it local, do not publish it.
#
# `prepare` uses DAP_CONTACT_EMAIL if set (Crossref and Unpaywall only; unset,
# Unpaywall is skipped). `classify` needs ANTHROPIC_API_KEY (or an
# `ant auth login` profile). Wall-clock time per stage goes to timing.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2="$HERE/../data_availability_pipeline/v2"
PY="${PYTHON:-python}"
RUNS="$HERE/runs"
GOLD_INPUT="${GOLD_INPUT:-$HERE/gold/input_papers.csv}"
mkdir -p "$RUNS"

stamp() { date +%s; }
timed() {  # timed <json-file> <label> <command...>
  local file="$1" label="$2"; shift 2
  local t0; t0=$(stamp); "$@"; local t1; t1=$(stamp)
  "$PY" - "$file" "$label" "$((t1 - t0))" <<'EOF'
import json, sys, os
f, k, s = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(f)) if os.path.exists(f) else {}
d[k] = s
json.dump(d, open(f, "w"), indent=2)
EOF
}

cd "$V2"
case "${1:-}" in
  prepare)
    DB="$RUNS/retrieved.sqlite"
    [ -e "$DB" ] && { echo "$DB exists; delete it to re-retrieve"; exit 1; }
    export DAP_TEXT_CACHE="$HERE/texts"
    T="$RUNS/retrieved.timing.json"
    timed "$T" enrol    "$PY" -m dap.pipeline --db "$DB" enrol --input "$GOLD_INPUT"
    timed "$T" metadata "$PY" -m dap.pipeline --db "$DB" stage metadata --workers 16 --year-min 0
    timed "$T" retrieve "$PY" -m dap.pipeline --db "$DB" stage retrieve --workers 8 --year-min 0
    "$PY" -m dap.pipeline --db "$DB" status
    echo "texts: $(ls "$HERE/texts"/*.txt 2>/dev/null | wc -l | tr -d ' ') papers in $HERE/texts"
    ;;
  classify)
    MODEL="${2:?usage: run_pipeline.sh classify <model> [name]}"
    NAME="${3:-pipeline_${MODEL}${ESCALATE:+_escalate}}"
    OUT="$RUNS/$NAME"; DB="$OUT/dap.sqlite"
    [ -e "$RUNS/retrieved.sqlite" ] || { echo "run '$0 prepare' first"; exit 1; }
    [ -e "$DB" ] && { echo "$DB exists; delete $OUT to re-run"; exit 1; }
    mkdir -p "$OUT"
    sqlite3 "$RUNS/retrieved.sqlite" ".backup '$DB'"
    cp "$RUNS/retrieved.timing.json" "$OUT/timing.json"
    ESC=(--no-escalate); [ -n "${ESCALATE:-}" ] && ESC=()
    timed "$OUT/timing.json" classify "$PY" -m dap.pipeline --db "$DB" stage classify \
        --mode api --model "$MODEL" "${ESC[@]}" --workers 4 --year-min 0 \
        --budget "${BUDGET:-25}"
    timed "$OUT/timing.json" verify "$PY" -m dap.pipeline --db "$DB" stage verify \
        --workers 16 --year-min 0
    "$PY" -m dap.pipeline --db "$DB" export --out "$OUT"
    "$PY" -m dap.pipeline --db "$DB" status
    echo "predictions: $OUT/analysis.csv"
    ;;
  *)
    sed -n '2,15p' "$0"; exit 2 ;;
esac
