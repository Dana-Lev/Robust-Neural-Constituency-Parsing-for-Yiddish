#!/usr/bin/env bash
# Run the frontier-LLM baseline conditions, budgeted around the per-day quota.
#
#   bash scripts/run_llm_baselines.sh day1     # 3 runs: both Lite conditions + frontier zero-shot
#   bash scripts/run_llm_baselines.sh day2     # 1 run:  frontier few-shot
#   bash scripts/run_llm_baselines.sh table    # assemble every run into one table
#
# Safe to re-run: if a result file already exists the run resumes it, so only
# the missing sentences are queried. Nothing is ever re-paid for.
#
# Override models if a tier is overloaded (each model has its OWN daily budget):
#   FRONTIER=gemini-3.7-flash LITE=gemini-3.1-flash-lite bash scripts/run_llm_baselines.sh day1
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA="yiddish_parser/data/processed/supar_ready/test.txt"
TRAIN="yiddish_parser/data/processed/supar_ready/train.txt"
FRONTIER="${FRONTIER:-gemini-3.5-flash}"    # 5 RPM / 20 RPD -> the headline number
LITE="${LITE:-gemini-3.5-flash-lite}"       # 15 RPM / 500 RPD -> the large samples
N_FRONTIER="${N_FRONTIER:-20}"              # exactly the daily cap
N_LITE="${N_LITE:-100}"
SHOTS="${SHOTS:-3}"

if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "ERROR: export GEMINI_API_KEY first (never commit it)."
    exit 1
fi
for f in "$DATA" "$TRAIN"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Copy the splits down from the cluster"
        echo "       (RUNBOOK Step 8) -- corpus data is gitignored."
        exit 1
    fi
done
mkdir -p results

# run <out> <model> <n> <extra args...>
run() {
    local out="$1" model="$2" n="$3"; shift 3
    local resume=()
    if [ -f "$out" ]; then
        echo ">>> $out exists -- resuming, only missing sentences will be queried"
        resume=(--resume "$out")
    fi
    echo ">>> $model  n=$n  -> $out"
    python3 testing.py --model "$model" --n "$n" --data "$DATA" \
        --out "$out" "${resume[@]}" "$@"
    echo
}

case "${1:-}" in
  day1)
    # Lite tier: 500/day, so both conditions fit comfortably in one day.
    run results/llm_lite_zeroshot.json "$LITE" "$N_LITE" --sleep 5
    run results/llm_lite_fewshot.json  "$LITE" "$N_LITE" --sleep 5 \
        --shots "$SHOTS" --train-file "$TRAIN"
    # Flash tier: 20/day, so one condition per day.
    run results/llm_frontier_zeroshot.json "$FRONTIER" "$N_FRONTIER" --sleep 13
    echo "Day 1 done. Run 'day2' tomorrow for the frontier few-shot condition."
    ;;
  day2)
    run results/llm_frontier_fewshot.json "$FRONTIER" "$N_FRONTIER" --sleep 13 \
        --shots "$SHOTS" --train-file "$TRAIN"
    echo "Day 2 done. Now: bash scripts/run_llm_baselines.sh table"
    ;;
  table)
    python3 scripts/llm_results_table.py "results/llm_*.json" --latex \
        | tee results/llm_comparison.md
    echo
    echo "Commit results/ and the table is ready for Table 2 of the report."
    ;;
  *)
    echo "usage: bash scripts/run_llm_baselines.sh {day1|day2|table}"
    exit 2
    ;;
esac
