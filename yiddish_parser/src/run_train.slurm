#!/bin/bash
#SBATCH --job-name=yiddish_dapt
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=studentkillable   
#SBATCH --gpus=1
#SBATCH --time=23:00:00               
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --signal=USR1@90              

# --- 1. TAU STORAGE SETUP ---
BASE_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/omriboiman"
export HF_HOME="$BASE_DIR/cache"
export XDG_CACHE_HOME="$BASE_DIR/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 2. ENVIRONMENT ---
source "$BASE_DIR/anaconda3/etc/profile.d/conda.sh"
conda activate yiddish

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