#!/bin/bash
#SBATCH --job-name=yiddish_parser
#SBATCH --output=logs/parser_%j.out
#SBATCH --error=logs/parser_%j.err
#SBATCH --partition=studentkillable   
#SBATCH --gpus=1
#SBATCH --time=4:00:00               
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# --- SETUP ---
# Course storage (guidelines 2.2): set NLP_STORAGE to your own directory.
BASE_DIR="${NLP_STORAGE:-/home/morg/NLP_2526b/$USER}"
export HF_HOME="$BASE_DIR/cache"
export TORCH_HOME="$BASE_DIR/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$BASE_DIR/anaconda3/etc/profile.d/conda.sh"
conda activate yiddish

mkdir -p logs

# --- RUN ---
echo "🚀 Running Standard Supar with Runtime Freeze..."
python src/train_parser.py

echo "✅ Job Finished."