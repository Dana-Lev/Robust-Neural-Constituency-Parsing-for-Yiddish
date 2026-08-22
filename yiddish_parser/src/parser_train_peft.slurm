#!/bin/bash
#SBATCH --job-name=yiddish_peft
#SBATCH --output=logs/peft_%j.out
#SBATCH --error=logs/peft_%j.err
#SBATCH --partition=studentkillable
#SBATCH --gpus=1
# The partition mixes TITAN Xp (sm_61) and RTX 2080 (sm_75) nodes, and
# current PyTorch wheels build no kernels below sm_75 -- on a TITAN Xp the
# job starts, reports cuda, then dies at the first kernel launch. Pin to
# the supported cards. Override with `sbatch --constraint=...` (or drop it
# entirely if you install a torch that still ships sm_61 kernels).
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# --- SETUP ---
# Course storage (see project guidelines §2.2): use your own directory under
# /home/morg/NLP_2526b/<your user name>. Set it here or export NLP_STORAGE
# before submitting. Remember to fill in the Slurm access form first.
BASE_DIR="${NLP_STORAGE:-/home/morg/NLP_2526b/$USER}"
export HF_HOME="$BASE_DIR/cache"
export TORCH_HOME="$BASE_DIR/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Find conda wherever it was installed under the storage dir. RUNBOOK Step 2
# installs Miniconda to $NLP_STORAGE/miniconda3; anaconda3 is kept as a fallback.
CONDA_SH=""
for candidate in "$BASE_DIR/miniconda3" "$BASE_DIR/anaconda3" "$HOME/miniconda3"; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$candidate/etc/profile.d/conda.sh"
        break
    fi
done
if [ -z "$CONDA_SH" ]; then
    echo "ERROR: no conda found under $BASE_DIR (looked for miniconda3/ and anaconda3/)."
    echo "Run RUNBOOK.md Step 2, or set NLP_STORAGE to the directory holding miniconda3."
    exit 1
fi
echo "Using conda from $CONDA_SH"
source "$CONDA_SH"
conda activate yiddish

mkdir -p logs

# One-time, in the `yiddish` env:
#   pip install "peft>=0.7.0" sentencepiece

# --- CONFIG ---
# Usage: sbatch src/parser_train_peft.slurm [mode] [adapter-type] [backend]
#   mode         : baseline | bpe-dropout | adapters | both      (default: both)
#   adapter-type : lora | bottleneck | both                      (default: lora)
#   backend      : auto | spm | maxmatch                         (default: maxmatch)
#
# ENCODER defaults to the raw Yiddish XLM-R, which measures the PEFT
# contribution on its own. Override it to stack on FOCUS+DAPT, e.g.
#   ENCODER=./output/phase2_trained/checkpoint-6000 sbatch ...
ENCODER="${ENCODER:-skulick/xlmb-ybc-ck05}"
MODE="${1:-both}"
ADAPTER_TYPE="${2:-lora}"
BACKEND="${3:-maxmatch}"
SEED="${SEED:-1}"
ADAPTER_LR="${ADAPTER_LR:-3e-4}"
N_BERT_LAYERS="${N_BERT_LAYERS:-4}"
# Overridable so the sanity-overfit check (RUNBOOK Step 5b) can also run through
# sbatch, which survives an SSH drop -- srun --pty does not.
DATA_DIR="${DATA_DIR:-data/processed/supar_ready}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"

echo "Mode=$MODE  AdapterType=$ADAPTER_TYPE  Backend=$BACKEND  Seed=$SEED  Encoder=$ENCODER"

# Sanity check the segmentation sampler before burning GPU hours (skipped for
# modes that do not resegment, and when SKIP_SELFTEST=1).
if [ "${SKIP_SELFTEST:-0}" != "1" ] && [ "$MODE" != "baseline" ] && [ "$MODE" != "adapters" ]; then
    python src/train_parser_peft.py --mode "$MODE" --encoder-path "$ENCODER" \
        --backend "$BACKEND" --data-dir "$DATA_DIR" --selftest
fi

OUTPUT_DIR="${OUTPUT_DIR:-}"
EXTRA=""
if [ -n "$OUTPUT_DIR" ]; then EXTRA="--output-dir $OUTPUT_DIR"; fi

# shellcheck disable=SC2086
python src/train_parser_peft.py \
    --mode "$MODE" \
    --adapter-type "$ADAPTER_TYPE" \
    --encoder-path "$ENCODER" \
    --backend "$BACKEND" \
    --p 0.1 \
    --word-p 1.0 \
    --fix-len 32 \
    --lora-r 16 \
    --lora-alpha 32 \
    --adapter-lr "$ADAPTER_LR" \
    --n-bert-layers "$N_BERT_LAYERS" \
    --batch-size 2000 \
    --data-dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --seed "$SEED" $EXTRA

echo "Job Finished."
