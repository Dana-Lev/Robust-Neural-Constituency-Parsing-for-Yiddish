#!/usr/bin/env bash
# Build the SuPar-ready PPCHY splits from scratch.
#
# Run from the repository root, on a login node (this is CPU-only work -- no
# GPU, no sbatch needed). Assumes ppchyprep has already produced its JSON
# output; see RUNBOOK.md Step 3 for that part.
#
#   bash scripts/build_ppchy_data.sh
#   ONLY="hirshbein olsvanger" bash scripts/build_ppchy_data.sh   # Kulick subset
#
# Expected on the whole corpus: 17,105 trees -> 15,394 / 855 / 856.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/yiddish_parser"

JSON_DIR="data/raw/ppchyprep/out/data/json"
ONLY="${ONLY:-}"

if [ ! -d "$JSON_DIR" ]; then
    echo "ERROR: $JSON_DIR not found."
    echo "Run ppchyprep first (RUNBOOK.md Step 3), so that its ./out directory"
    echo "lands at yiddish_parser/data/raw/ppchyprep/out."
    exit 1
fi
echo "Found $(ls -1 "$JSON_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ') JSON files in $JSON_DIR"

mkdir -p data/processed

echo
echo "[1/4] JSON -> Hebrew-script bracketed trees"
if [ -n "$ONLY" ]; then
    # shellcheck disable=SC2086
    python "src/ppchy_formatting/build_final_trees.py" --only $ONLY
else
    python "src/ppchy_formatting/build_final_trees.py"
fi

echo
echo "[2/4] Wrap in (TOP ...), drop empty nodes, balance check"
python "src/ppchy_formatting/finalize_ppchy_for_supar.py"

echo
echo "[3/4] 90/5/5 split (seed 42)"
python src/split_supar_data.py

echo
echo "[4/4] Remove trees SuPar would reject"
python src/clean_tree_data.py

echo
echo "Done. Splits are in yiddish_parser/data/processed/supar_ready/"
echo "Next (RUNBOOK Step 4):"
echo "  cd $REPO_ROOT && mkdir -p results"
echo "  python scripts/dataset_stats.py \\"
echo "      --data-dir yiddish_parser/data/processed/supar_ready \\"
echo "      --encoder skulick/xlmb-ybc-ck05 --markdown | tee results/dataset_stats.md"
echo "  python scripts/split_provenance.py --write-labels | tee results/split_composition.txt"
