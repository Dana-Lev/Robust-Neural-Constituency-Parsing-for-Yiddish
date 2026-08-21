# Runbook: build the data on Slurm, run the experiments, push results

Follow these in order. Steps 0–4 are one-time setup and data building; 5–7 are the
experiments; 8 runs the LLM baseline; 9 pushes everything back to git.

Conventions used below:
- `$STORAGE` = your course storage, `/home/morg/NLP_2526b/<your user name>`.
- Login-node work is CPU-only and safe to run interactively. Anything that trains
  goes through `sbatch`. **Never train on the login node.**
- After each step there is a *check* — run it before moving on.

---

## Step 0 — Cluster access (do this first, it can take a day)

1. Fill the Slurm request form: <https://www.cs.tau.ac.il/system/SlurmRequestForm>
   In the PI/lab field write **"NLP class 2025/2026"**. You must be in the CS
   Slurm groups before login will work; a TA can request the course partition.
2. Read the usage notes: <https://www.cs.tau.ac.il/system/slurm>
3. Log in and confirm your storage exists.

**Credentials.** There is no special cluster password: you log in with your own
**TAU username and password** ("University Credentials", per the page above).
The login host is `slurm-client.cs.tau.ac.il`, which routes you to one of
`c-001`…`c-010`. Two things that block people on the first attempt:

- **Off campus?** CS servers are not reachable from outside the university.
  Connect to the **TAU VPN** first.
- **"Remote host key has changed"?** Expected, and it will recur. The pool name
  routes you to one of `c-001`…`c-010`, each with a *different* host key, so a
  single stored key breaks as soon as you land on another node. Clear the stale
  entry with `ssh-keygen -R slurm-client.cs.tau.ac.il`, or fix it permanently by
  trusting all ten at once (`known_hosts` accepts several keys per name):

```bash
ssh-keyscan -T 8 -t ed25519,rsa,ecdsa \
    slurm-client.cs.tau.ac.il c-00{1..9}.cs.tau.ac.il c-010.cs.tau.ac.il \
  | tee /tmp/slurm_keys \
  | sed -E 's/^c-[0-9]+\.cs\.tau\.ac\.il/slurm-client.cs.tau.ac.il/' >> ~/.ssh/known_hosts
cat /tmp/slurm_keys >> ~/.ssh/known_hosts && sort -u ~/.ssh/known_hosts -o ~/.ssh/known_hosts
```

If the password itself is rejected, that is a TAU/CS account issue — contact the
CS system team; it is not something any project document can supply.

A worthwhile `~/.ssh/config` block — the `User` line is the one people forget,
and without it SSH sends your *local* username and fails before the password is
even considered. `ControlMaster` means you type the password once and every
later `ssh`/`scp` reuses that connection:

```
Host slurm slurm-client.cs.tau.ac.il
  HostName slurm-client.cs.tau.ac.il
  User <your-tau-username>
  PreferredAuthentications keyboard-interactive,password
  IdentitiesOnly yes
  ControlMaster auto
  ControlPath ~/.ssh/controlmasters/%r@%h-%p
  ControlPersist 10m
  ServerAliveInterval 60
```

(`mkdir -p ~/.ssh/controlmasters` first.) Then `ssh slurm` is enough.

```bash
ssh <user>@slurm-client.cs.tau.ac.il
export STORAGE=/home/morg/NLP_2526b/$USER
mkdir -p "$STORAGE" && df -h "$STORAGE" && sinfo -p studentkillable
```

**Check:** `sinfo` lists the `studentkillable` partition and `$STORAGE` is writable.

---

## Step 1 — Put the repo and caches on the cluster

Everything heavy (HF cache, conda envs, checkpoints) must live under `$STORAGE`,
not in your home directory quota.

```bash
export STORAGE=/home/morg/NLP_2526b/$USER
export NLP_STORAGE="$STORAGE"          # every script in this repo reads this
export HF_HOME="$STORAGE/cache"
export TORCH_HOME="$STORAGE/cache"
mkdir -p "$HF_HOME"

cd "$STORAGE"
git clone https://github.com/Dana-Lev/Robust-Neural-Constituency-Parsing-for-Yiddish.git
cd Robust-Neural-Constituency-Parsing-for-Yiddish
```

Make the environment variables permanent so every future login and job inherits them:

```bash
cat >> ~/.bashrc <<'EOF'
export STORAGE=/home/morg/NLP_2526b/$USER
export NLP_STORAGE="$STORAGE"
export HF_HOME="$STORAGE/cache"
export TORCH_HOME="$STORAGE/cache"
EOF
```

**Check:** `echo $NLP_STORAGE $HF_HOME` prints paths under `/home/morg/NLP_2526b/`.

---

## Step 2 — Two conda environments

Two, deliberately: `ppchyprep` pins old `numpy`/`scikit-learn`/`yiddish` versions
that clash with the parser stack. Keeping them apart avoids a dependency fight.

```bash
# Parser + LLM baseline environment
conda create -y -n yiddish python=3.10
conda activate yiddish
pip install -r requirements.txt
python -c "import supar, torch, peft, transformers; print(supar.__version__, torch.__version__)"

# Data-preparation environment
conda create -y -n ppchyprep python=3.10
```

**Check:** the first `python -c` prints `1.1.4` and a torch version without error.

---

## Step 3 — Build the PPCHY treebank with `ppchyprep`

This converts the romanized PPCHY trees into Hebrew script. It is Kulick's own
tool, so the conversion matches the published setup.

```bash
conda activate ppchyprep
cd "$NLP_STORAGE/Robust-Neural-Constituency-Parsing-for-Yiddish/yiddish_parser/data/raw"

# Two dependencies that are not on PyPI
git clone https://github.com/skulick/yiddishycode.git && pip install ./yiddishycode
git clone https://github.com/skulick/ppctree.git      && pip install ./ppctree

# The tool itself
git clone https://github.com/skulick/ppchyprep.git
cd ppchyprep
pip install -r requirements.txt

# The corpus, at the exact commit ppchyprep expects
mkdir -p data && cd data
git clone https://github.com/beatrice57/penn-parsed-corpus-of-historical-yiddish.git
cd penn-parsed-corpus-of-historical-yiddish
git checkout 3cedddb2fa11b6873e92dbd043e29df39c8612e6
cd ../..

# Produces ./out, including out/data/json
./run.sh
```

**Check:**

```bash
ls out/data/json/*.json | wc -l      # should be non-zero
```

If `pp_psd.sh` fails at the end of `run.sh`, that is fine — it runs *after*
`make_json.py`, so the JSON you need already exists.

---

## Step 4 — Turn the JSON into SuPar-ready splits

```bash
conda activate yiddish
cd "$NLP_STORAGE/Robust-Neural-Constituency-Parsing-for-Yiddish"
./scripts/build_ppchy_data.sh
```

That runs four steps: JSON → Hebrew-script trees → `(TOP …)` normalization →
90/5/5 split (seed 42) → removal of trees SuPar would reject.

### One decision to make consciously

By default this ingests **every** PPCHY component. The prior project's report
claims to follow Kulick et al. (2022) and use only the two largest 20th-century
texts — but its code processed everything. Pick one and make the report match:

```bash
# whole corpus (default)
./scripts/build_ppchy_data.sh

# or the Kulick et al. subset
ONLY="hirshbein olsvanger" ./scripts/build_ppchy_data.sh
```

### Record the statistics — the report needs them

```bash
python scripts/dataset_stats.py \
    --data-dir yiddish_parser/data/processed/supar_ready \
    --encoder skulick/xlmb-ybc-ck05 --markdown | tee results/dataset_stats.md
```

**Check:** `train.txt`, `dev.txt`, `test.txt` all exist with non-zero sentence
counts, and `malformed` is 0 after cleaning. Use *these* numbers in the report —
never the inherited 15,394 / 855 / 856.

---

## Step 5 — Two sanity checks before spending GPU hours

**5a. Segmentation sampler.** Confirms the subword regularizer actually produces
varied segmentations, and that nothing exceeds `--fix-len`:

```bash
cd yiddish_parser
python src/train_parser_peft.py --selftest --backend maxmatch \
    --encoder-path skulick/xlmb-ybc-ck05 | tee ../results/selftest_maxmatch.txt
python src/train_parser_peft.py --selftest --backend spm \
    --encoder-path skulick/xlmb-ybc-ck05 | tee ../results/selftest_spm.txt
```

**Check:** `pct_words_with_variation` > 0 and `max_pieces_observed` < 32. If the
max approaches 32, raise `--fix-len` (SuPar truncates silently).

**5b. Overfitting check.** The grading guidelines ask explicitly for evidence that
the model *can* fit data — proof there is no bug:

```bash
mkdir -p data/processed/overfit
head -100 data/processed/supar_ready/train.txt > data/processed/overfit/train.txt
cp data/processed/overfit/train.txt data/processed/overfit/dev.txt
cp data/processed/overfit/train.txt data/processed/overfit/test.txt

srun -p studentkillable --gpus=1 --time=1:00:00 --mem=32G --pty \
    python src/train_parser_peft.py --mode baseline \
        --encoder-path skulick/xlmb-ybc-ck05 \
        --data-dir data/processed/overfit \
        --output-dir ./output/sanity_overfit \
        --epochs 40 --patience 40 2>&1 | tee ../results/sanity_overfit.txt
```

**Check:** F1 climbs toward ~100 on this tiny set. If it plateaus low, stop and
debug — do not launch the grid.

---

## Step 6 — Launch the experiment grid

Each cell writes to its own `output/parser_<cell>/`, so nothing overwrites
anything. Launch from the `yiddish_parser/` directory:

```bash
cd "$NLP_STORAGE/Robust-Neural-Constituency-Parsing-for-Yiddish/yiddish_parser"
mkdir -p logs

# arguments: mode  adapter-type  backend
sbatch src/parser_train_peft.slurm baseline
sbatch src/parser_train_peft.slurm bpe-dropout lora spm
sbatch src/parser_train_peft.slurm bpe-dropout lora maxmatch
sbatch src/parser_train_peft.slurm adapters   lora
sbatch src/parser_train_peft.slurm adapters   bottleneck
sbatch src/parser_train_peft.slurm both       lora maxmatch
```

Monitor:

```bash
squeue -u $USER
tail -f logs/peft_<jobid>.out
```

`studentkillable` jobs can be preempted. If a job dies, just resubmit the same
line — each cell is independent.

### Follow-up runs (Step 6b, after the first six land)

```bash
# Adapter learning-rate sweep -- a flat adapter result at too low an LR is an
# artifact, not a finding.
ADAPTER_LR=1e-4 OUTPUT_DIR=./output/parser_adapters_lora_lr1e-4 \
    sbatch src/parser_train_peft.slurm adapters lora
ADAPTER_LR=1e-3 OUTPUT_DIR=./output/parser_adapters_lora_lr1e-3 \
    sbatch src/parser_train_peft.slurm adapters lora

# Seed variance on the cells you will report
SEED=2 sbatch src/parser_train_peft.slurm baseline
SEED=3 sbatch src/parser_train_peft.slurm baseline

# Optional: 12-layer scalar mix, to connect to the prior project's 83.81 setup
N_BERT_LAYERS=12 sbatch src/parser_train_peft.slurm baseline
```

**Check:** every finished job leaves a `yiddish_parser.pt` in its output directory.

---

## Step 7 — Evaluate and collect the numbers

Use `evaluate_peft.py`, **not** `Parser.load` — SuPar loads with `strict=False`,
so an adapter checkpoint loaded without the adapter patch silently drops tensors
and reports a number from a partly random encoder.

```bash
cd "$NLP_STORAGE/Robust-Neural-Constituency-Parsing-for-Yiddish/yiddish_parser"

# no-adapter cells
for cell in baseline bpe-dropout_spm bpe-dropout_maxmatch; do
    python src/evaluate_peft.py --path ./output/parser_$cell/yiddish_parser.pt \
        --adapter-type none | tee ../results/eval_$cell.txt
done

# adapter cells -- the --adapter-type MUST match how the cell was trained
python src/evaluate_peft.py --path ./output/parser_adapters_lora/yiddish_parser.pt \
    --adapter-type lora | tee ../results/eval_adapters_lora.txt
python src/evaluate_peft.py --path ./output/parser_adapters_bottleneck/yiddish_parser.pt \
    --adapter-type bottleneck | tee ../results/eval_adapters_bottleneck.txt
python src/evaluate_peft.py --path ./output/parser_both_lora_maxmatch/yiddish_parser.pt \
    --adapter-type lora | tee ../results/eval_both_lora_maxmatch.txt
```

Then assemble `results/parsing_results.md` with one row per cell (UF, LF, UCM,
LCM, trainable parameters — the trainable count is printed by
`describe_trainable` at the top of each training log).

**Check:** `evaluate_peft.py` raises rather than printing a number if the adapter
configuration is wrong. A crash here is the script protecting you.

---

## Step 8 — Frontier-LLM baseline (run locally, not on the cluster)

The cluster has no outbound API access, and this needs none of its GPUs. Copy the
splits down to your Mac:

```bash
cd ~/Desktop/Robust-Neural-Constituency-Parsing-for-Yiddish
mkdir -p yiddish_parser/data/processed/supar_ready
scp '<user>@slurm-client.cs.tau.ac.il:/home/morg/NLP_2526b/<user>/Robust-Neural-Constituency-Parsing-for-Yiddish/yiddish_parser/data/processed/supar_ready/*.txt' \
    yiddish_parser/data/processed/supar_ready/

export GEMINI_API_KEY="your-key"

python testing.py --data yiddish_parser/data/processed/supar_ready/test.txt \
    --n 30 --out results/llm_zeroshot.json

python testing.py --data yiddish_parser/data/processed/supar_ready/test.txt \
    --n 30 --shots 3 --train-file yiddish_parser/data/processed/supar_ready/train.txt \
    --out results/llm_fewshot.json
```

Both runs use the same `--seed`, so they score the same 30 sentences — that is
what makes zero-shot and few-shot comparable. Note the model ID and the date in
the report; API models change under you.

**Check:** `valid_syntax_rate` and `labeled_f1` appear in the final report block,
and the JSON files land in `results/`.

---

## Step 9 — Push everything back to git

Data, checkpoints and Slurm logs are gitignored on purpose. What you commit is
code changes plus the small evidence files in `results/`.

**On the cluster**, commit the results:

```bash
cd "$NLP_STORAGE/Robust-Neural-Constituency-Parsing-for-Yiddish"
git status --short
git add results/
git commit -m "Add dataset statistics, sanity checks and parsing results"
git push
```

**On your Mac**, commit the LLM results and pull the cluster's commits:

```bash
cd ~/Desktop/Robust-Neural-Constituency-Parsing-for-Yiddish
git pull --rebase
git add results/ && git commit -m "Add frontier-LLM baseline results"
git push
```

Before every push:

```bash
# no credential ever enters the repo
git diff --cached | grep -inE "AIza|hf_[A-Za-z0-9]{20,}|GEMINI_API_KEY *= *\"" && echo "STOP: secret staged"
# no checkpoint sneaks in
git diff --cached --stat | tail -1
```

If a push is rejected because the remote moved, `git pull --rebase` then push again.

---

## Optional — reproduce the FOCUS + DAPT reference branch

**You probably do not need this.** Your two approaches need only PPCHY and the
Hugging Face encoder. Jochre and YBC are required solely to re-run the *prior*
project's static-adaptation pipeline, whose numbers you can cite instead
(83.81 → 83.02 LF). It costs many GPU hours and a large download. Only if you
want it first-hand:

```bash
# Jochre: clean vocabulary source
cd yiddish_parser/data/raw
git clone https://gitlab.com/jochre/corpora/jochre-yiddish-corpus.git jochre_src
# place the extracted .txt files under data/raw/jochre/, then
cd ../.. && python src/data_extraction/build_vocab_jochre.py

# FastText vectors for FOCUS (~4.5GB) -- must live under $NLP_STORAGE
mkdir -p "$NLP_STORAGE/fasttext_models"
curl -L -o "$NLP_STORAGE/fasttext_models/cc.yi.300.bin.gz" \
    https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.yi.300.bin.gz
gunzip "$NLP_STORAGE/fasttext_models/cc.yi.300.bin.gz"

pip install deepfocus
python src/inject_vocab.py                    # FOCUS injection
sbatch src/run_train.slurm                    # DAPT (MLM) -- long
python src/eval_token_usage.py                # "zombie token" activation analysis

# then stack PEFT on top of the DAPT checkpoint
ENCODER=./output/phase2_trained/checkpoint-6000 \
    sbatch src/parser_train_peft.slurm both lora maxmatch
```

---

## Quick reference

| Task | Command |
|---|---|
| Build data | `./scripts/build_ppchy_data.sh` |
| Data statistics | `python scripts/dataset_stats.py --data-dir yiddish_parser/data/processed/supar_ready --markdown` |
| Sampler self-test | `python src/train_parser_peft.py --selftest --backend maxmatch` |
| Train one cell | `sbatch src/parser_train_peft.slurm <mode> <adapter> <backend>` |
| Evaluate one cell | `python src/evaluate_peft.py --path output/parser_<cell>/yiddish_parser.pt --adapter-type <type>` |
| LLM baseline | `python testing.py --data .../test.txt --n 30 --out results/llm_zeroshot.json` |
| Watch jobs | `squeue -u $USER` · `tail -f logs/peft_<jobid>.out` |
