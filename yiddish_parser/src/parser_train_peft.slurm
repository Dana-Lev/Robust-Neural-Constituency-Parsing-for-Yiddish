#!/bin/bash
#SBATCH --job-name=yiddish_peft
#SBATCH --output=logs/peft_%j.out
#SBATCH --error=logs/peft_%j.err
#SBATCH --partition=studentkillable
#SBATCH --gpus=1
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

source "$BASE_DIR/anaconda3/etc/profile.d/conda.sh"
conda activate yiddish

mkdir -p logs

# One-time, in the `yiddish` env:
#   pip install "peft>=0.7.0" sentencepiece

# --- CONFIG ---
# Point at the raw Yiddish XLM-R to measure the PEFT contribution on its own,
# or at ./output/phase2_trained/checkpoint-6000 to stack it on FOCUS+DAPT.
ENCODER="skulick/xlmb-ybc-ck05"
MODE="${1:-both}"                 # baseline | bpe-dropout | adapters | both
ADAPTER_TYPE="${2:-lora}"         # lora | bottleneck | both

echo "Mode=$MODE  AdapterType=$ADAPTER_TYPE  Encoder=$ENCODER"

# Sanity check the segmentation sampler before burning GPU hours.
python src/train_parser_peft.py --mode "$MODE" --encoder-path "$ENCODER" --selftest

python src/train_parser_peft.py \
    --mode "$MODE" \
    --adapter-type "$ADAPTER_TYPE" \
    --encoder-path "$ENCODER" \
    --backend maxmatch \
    --p 0.1 \
    --word-p 1.0 \
    --fix-len 32 \
    --lora-r 16 \
    --lora-alpha 32 \
    --adapter-lr 3e-4 \
    --batch-size 2000 \
    --epochs 100 \
    --patience 10 \
    --seed 1

echo "Job Finished."
