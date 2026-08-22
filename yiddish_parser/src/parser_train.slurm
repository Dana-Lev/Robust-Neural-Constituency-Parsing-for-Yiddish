#!/bin/bash
#SBATCH --job-name=yiddish_parser
#SBATCH --output=logs/parser_%j.out
#SBATCH --error=logs/parser_%j.err
#SBATCH --partition=studentkillable   
#SBATCH --gpus=1
# The partition mixes TITAN Xp (sm_61) and RTX 2080 (sm_75) nodes, and
# current PyTorch wheels build no kernels below sm_75 -- on a TITAN Xp the
# job starts, reports cuda, then dies at the first kernel launch. Pin to
# the supported cards. Override with `sbatch --constraint=...` (or drop it
# entirely if you install a torch that still ships sm_61 kernels).
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --time=4:00:00               
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# --- SETUP ---
# Course storage (guidelines 2.2): set NLP_STORAGE to your own directory.
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

# --- RUN ---
echo "🚀 Running Standard Supar with Runtime Freeze..."
python src/train_parser.py

echo "✅ Job Finished."