#!/bin/bash
#SBATCH --partition=studentkillable
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --job-name=sanity_overfit
#SBATCH --output=../results/sanity_overfit.txt
#SBATCH --error=../results/sanity_overfit.err

# Load Conda and activate the environment
source /home/morg/NLP_2526b/$USER/miniconda3/bin/activate
conda activate yiddish

# Run the training
python src/train_parser_peft.py --mode baseline \
    --encoder-path skulick/xlmb-ybc-ck05 \
    --data-dir data/processed/overfit \
    --output-dir ./output/sanity_overfit \
    --epochs 40 --patience 40
