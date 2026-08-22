#!/bin/bash
#SBATCH --job-name=yiddish_dapt
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=studentkillable   
#SBATCH --gpus=1
# The partition mixes TITAN Xp (sm_61) and RTX 2080 (sm_75) nodes, and
# current PyTorch wheels build no kernels below sm_75 -- on a TITAN Xp the
# job starts, reports cuda, then dies at the first kernel launch. Pin to
# the supported cards. Override with `sbatch --constraint=...` (or drop it
# entirely if you install a torch that still ships sm_61 kernels).
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --time=23:00:00               
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --signal=USR1@90              

# --- 1. TAU STORAGE SETUP ---
# Course storage (guidelines 2.2): set NLP_STORAGE to your own directory.
BASE_DIR="${NLP_STORAGE:-/home/morg/NLP_2526b/$USER}"
export HF_HOME="$BASE_DIR/cache"
export XDG_CACHE_HOME="$BASE_DIR/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 2. ENVIRONMENT ---
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

# --- 3. CONFIGURATION ---
MODEL_DIR="./output/phase1_focus_model"
TRAIN_FILE="./data/raw/ybc/context.txt"
OUTPUT_DIR="./output/phase2_trained"

# --- 4. CHECK FOR RESUME ---
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
    echo "🔄 Detected existing output. Attempting to RESUME..."
    OVERWRITE_FLAG=""
else
    echo "🆕 Starting fresh training..."
    OVERWRITE_FLAG="--overwrite_output_dir"
fi

# --- 5. RUN TRAINING ---
# ⚠️ CHANGED: num_train_epochs 3 (Much faster, sufficient for huge data)

srun python src/run_mlm.py \
    --model_name_or_path $MODEL_DIR \
    --train_file $TRAIN_FILE \
    --do_train \
    --do_eval \
    --validation_split_percentage 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --save_steps 3000 \
    --save_total_limit 1 \
    --output_dir $OUTPUT_DIR \
    $OVERWRITE_FLAG \
    --fp16 \
    --report_to none

echo "✅ Job Finished."