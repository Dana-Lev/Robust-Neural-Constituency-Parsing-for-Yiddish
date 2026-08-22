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
**TAU username and password** ("University Credentials", per the page above). The
login host is `slurm-client.cs.tau.ac.il`, which routes you to one of
`c-001`…`c-010`. Two things block people on the first attempt:

**Off campus?** CS servers are not reachable from outside the university.
Connect to the TAU VPN (via **GlobalProtect**) first.

**"Remote host key has changed"?** Expected, and it will recur. The pool name
routes you to a different node each time, and each node has its own host key, so
one stored key breaks as soon as you land elsewhere. Clear the stale entry with
`ssh-keygen -R slurm-client.cs.tau.ac.il`, or fix it permanently by trusting all
ten at once (`known_hosts` accepts several keys per name):

```bash
ssh-keyscan -T 8 -t ed25519,rsa,ecdsa \
    slurm-client.cs.tau.ac.il c-00{1..9}.cs.tau.ac.il c-010.cs.tau.ac.il \
  | tee /tmp/slurm_keys \
  | sed -E 's/^c-[0-9]+\.cs\.tau\.ac\.il/slurm-client.cs.tau.ac.il/' >> ~/.ssh/known_hosts
cat /tmp/slurm_keys >> ~/.ssh/known_hosts && sort -u ~/.ssh/known_hosts -o ~/.ssh/known_hosts
```

If the password itself is rejected, that is a TAU/CS account issue — contact the
CS system team; no project document can supply it.

A worthwhile `~/.ssh/config` block. The `User` line is the one people forget, and
without it SSH sends your local username and fails before the password is even
considered. `ControlMaster` means you type the password once and every later
`ssh`/`scp` reuses that connection — run `mkdir -p ~/.ssh/controlmasters` first:

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

Then `ssh slurm` is enough.

> **If a connection ever hangs forever**, a previous failed attempt left a dead
> control socket behind and every later attempt waits on it. Clear it and retry
> without multiplexing:
> `rm -f ~/.ssh/controlmasters/* && ssh -o ControlMaster=no -o ConnectTimeout=10 slurm`

**The shell.** If you log in and see a `>` prompt rather than `$`, you are in a C
shell. Type `bash -l` to switch to a bash login shell before proceeding —
everything below assumes bash.

```bash
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

Make them permanent so every later login and every Slurm job inherits them. The
last line puts conda on your `PATH` too, so you do not have to re-source it by
hand in every new shell (add it after Step 2 has actually installed conda):

```bash
cat >> ~/.bashrc <<'EOF'
export STORAGE=/home/morg/NLP_2526b/$USER
export NLP_STORAGE="$STORAGE"
export HF_HOME="$STORAGE/cache"
export TORCH_HOME="$STORAGE/cache"
[ -f "$STORAGE/miniconda3/etc/profile.d/conda.sh" ] && source "$STORAGE/miniconda3/etc/profile.d/conda.sh"
EOF
```

**Check:** `echo $NLP_STORAGE $HF_HOME` prints paths under `/home/morg/NLP_2526b/`.

---

## Step 2 — Install Miniconda, then create the environments

There is no usable system conda: the compute nodes have no environment modules
and no working `ensurepip`, so install your own Miniconda into persistent storage.

```bash
# 1. Miniconda, into your persistent storage
cd "$STORAGE"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "$STORAGE/miniconda3"

# 2. Activate it
source "$STORAGE/miniconda3/bin/activate"

# 3. Accept the Anaconda channel terms (required on first use, or env creation fails)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 4. Parser + LLM baseline environment
conda create -y -n yiddish python=3.10
conda activate yiddish
pip install -r "$NLP_STORAGE/Robust-Neural-Constituency-Parsing-for-Yiddish/requirements.txt"
python -c "import supar, torch, peft, transformers; print(supar.__version__, torch.__version__)"

# 5. Data-preparation environment, kept separate on purpose: ppchyprep pins old
#    numpy / scikit-learn / yiddish versions that clash with the parser stack
conda deactivate
conda create -y -n ppchyprep python=3.10
```

**Check:** that `python -c` prints `1.1.4` and a torch version, with no import error.

> Every new shell needs conda on its `PATH` again — either re-run
> `source "$STORAGE/miniconda3/bin/activate"`, or add the line from Step 1 to
> `~/.bashrc` once and forget about it. The Slurm job scripts in this repo find
> conda under `$NLP_STORAGE` themselves.

---

## Step 3 — Build the PPCHY treebank with `ppchyprep`

This converts the romanized PPCHY trees into Hebrew script. It is Kulick's own
tool, so the conversion matches the published setup. Note the `mkdir` — `data/`
is gitignored, so the directory does not exist in a fresh clone.

```bash
source "$STORAGE/miniconda3/bin/activate"
conda activate ppchyprep

mkdir -p "$NLP_STORAGE/Robust-Neural-Constituency-Parsing-for-Yiddish/yiddish_parser/data/raw"
cd "$NLP_STORAGE/Robust-Neural-Constituency-Parsing-for-Yiddish/yiddish_parser/data/raw"

# two dependencies that are not on PyPI
git clone https://github.com/skulick/yiddishycode.git && pip install ./yiddishycode
git clone https://github.com/skulick/ppctree.git      && pip install ./ppctree

# the tool itself
git clone https://github.com/skulick/ppchyprep.git
cd ppchyprep
pip install -r requirements.txt

# the corpus, at the exact commit ppchyprep expects
mkdir -p data && cd data
git clone https://github.com/beatrice57/penn-parsed-corpus-of-historical-yiddish.git
cd penn-parsed-corpus-of-historical-yiddish
git checkout 3cedddb2fa11b6873e92dbd043e29df39c8612e6
cd ../..

# run.sh ships without the execute bit; produces ./out, including out/data/json
chmod +x run.sh
./run.sh
```

**Check:**

```bash
ls out/data/json/*.json | wc -l      # should be non-zero
```

If `pp_psd.sh` fails at the very end of `run.sh`, that is fine — it runs *after*
`make_json.py`, so the JSON you need already exists.

---

## Step 4 — Turn the JSON into SuPar-ready splits

Make sure you are in the parser environment, at the root of the repository:

```bash
conda activate yiddish
cd "$NLP_STORAGE/Robust-Neural-Constituency-Parsing-for-Yiddish"
```

The build script runs four steps: JSON → Hebrew-script trees → `(TOP …)`
normalization → 90/5/5 split (seed 42) → removal of trees SuPar would reject.

### One decision to make consciously

By default this ingests **every** PPCHY component. The prior project's report
claims to follow Kulick et al. (2022) and use only the two largest 20th-century
texts — but its code processed everything. Pick one and make the report match:

```bash
# Option A: whole corpus (default) -- what this project uses
bash scripts/build_ppchy_data.sh

# Option B: the Kulick et al. subset
ONLY="hirshbein olsvanger" bash scripts/build_ppchy_data.sh
```

**Use Option A**, for three reasons: the older non-SYO components are exactly
where orthographic variation and fragmentation are worst, so dropping them
removes the phenomenon under test; the prior project's 83.81 LF came from the
whole corpus, so Option A is what makes those reference rows comparable; and
differences of ±1 LF need the larger test set to be visible. The report must then
describe the data as *all PPCHY components* and note the discrepancy with the
prior write-up.

### Record the statistics — the report needs them

```bash
mkdir -p results
python scripts/dataset_stats.py \
    --data-dir yiddish_parser/data/processed/supar_ready \
    --encoder skulick/xlmb-ybc-ck05 --markdown | tee results/dataset_stats.md
```

**Check:** read the output table. `train.txt`, `dev.txt` and `test.txt` must all
exist with non-zero sentence counts, and `Malformed` must be **0** after
cleaning. With Option A you should get exactly **15,394 / 855 / 856**.

That figure is worth pausing on: 15,394 + 855 + 856 = 17,105, and this script's
90/5/5 split with integer truncation maps 17,105 trees to precisely those three
numbers. So hitting them confirms the pipeline is deterministic *and* settles the
question above — the prior project's identical counts are only reachable from the
whole corpus, not from an 83k-token subset. (If the cleaning step removed any
trees, your counts will be slightly lower. Record what you actually get, and cite
those numbers in the report.)

### Record which components landed in each split

`build_final_trees.py` writes a `*.sources.tsv` manifest, because the split step
shuffles a flat file and otherwise loses track of provenance. Recover it:

```bash
python scripts/split_provenance.py \
    --manifest yiddish_parser/data/processed/ppchy_final_trees.sources.tsv \
    --data-dir yiddish_parser/data/processed/supar_ready \
    --write-labels | tee results/split_composition.txt
```

This gives the composition of each split for the report's data section, plus
`test.sources.txt` aligned line-for-line with `test.txt`. With that label file,
test F1 can be broken down per component **without retraining** — which is how
you answer "does subword regularization help more where the orthography is
noisier?", the sharpest version of this project's question.

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

### Pre-check: can the job find your conda?

Six jobs failing identically at activation is the most expensive way to discover
a path problem, so confirm it with **one** job before fanning out.

You do **not** need to hand-edit the job script. It locates conda itself, trying
`$NLP_STORAGE/miniconda3`, then `$NLP_STORAGE/anaconda3`, then `~/miniconda3`,
and exits immediately with a pointer to Step 2 if none of them exist. Since
`sbatch` inherits your submitting environment, exporting `NLP_STORAGE` (Step 1)
is all it needs.

```bash
cd "$NLP_STORAGE/Robust-Neural-Constituency-Parsing-for-Yiddish/yiddish_parser"
mkdir -p logs

# submit ONE cell first
sbatch src/parser_train_peft.slurm baseline

# then read its log: activation happens in the first few lines
tail -f logs/peft_<jobid>.out
```

**Check:** the log prints `Using conda from /…/miniconda3/etc/profile.d/conda.sh`,
then the parameter-count table from `describe_trainable`, then training starts.
If instead it prints `ERROR: no conda found under …`, either `NLP_STORAGE` did
not reach the job (`echo $NLP_STORAGE` in the submitting shell) or Step 2 has not
finished. Fix that before submitting anything else.

### The rest of the grid

Each cell writes to its own `output/parser_<cell>/`, so nothing overwrites
anything. Arguments are `mode`, `adapter-type`, `backend`.

```bash
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

`studentkillable` jobs can be preempted. If a job dies, resubmit the same line —
each cell is independent.

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

# Optional: 12 encoder layers into the scalar mix, to connect to the prior
# project's 83.81 configuration
N_BERT_LAYERS=12 sbatch src/parser_train_peft.slurm baseline
```

**Check:** every finished job leaves a `yiddish_parser.pt` in its own output
directory. Directory names include mode, adapter, backend, layer count and seed,
so no two cells can overwrite each other.

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

Both runs use the same `--seed`, so they score the same sentences — that is what
makes zero-shot and few-shot comparable. Note the model ID and the date in the
report; API models change under you.

### Budget the run against the daily cap

Free-tier limits are **per day**, not only per minute, and that decides the
design. Check yours in the API dashboard first:

| Tier | RPM | RPD | Use it for |
|---|---:|---:|---|
| Flash (3.x) | 5 | **20** | the headline "frontier model" number, ~20 sentences/day |
| Flash Lite (3.x) | 15 | **500** | large samples: 100 sentences × both conditions in a day |

Run two models, for two different jobs:

```bash
# exact ids for your key -- cheaper than guessing and spending a failed request
python testing.py --list-models

# frontier model, 20 sentences: fits the daily cap exactly, 100% coverage
python testing.py --model gemini-3.7-flash --n 20 --sleep 13 \
    --data yiddish_parser/data/processed/supar_ready/test.txt \
    --out results/llm_frontier_zeroshot.json
# ... then the few-shot condition the next day

# large sample on a Lite tier: 100 per condition, 200 requests total
python testing.py --model gemini-3.5-flash-lite --n 100 --sleep 5 \
    --data yiddish_parser/data/processed/supar_ready/test.txt \
    --out results/llm_lite_zeroshot.json
python testing.py --model gemini-3.5-flash-lite --n 100 --shots 3 --sleep 5 \
    --data yiddish_parser/data/processed/supar_ready/test.txt \
    --train-file yiddish_parser/data/processed/supar_ready/train.txt \
    --out results/llm_lite_fewshot.json
```

Report the frontier model as the primary result: the question is whether a
*frontier* LLM can do this, and a Lite tier is the cheapest model available, so a
Lite-only answer invites the objection that the strongest model was never tested.
Use the Lite runs for tight estimates and the per-component breakdown, and say
plainly which number came from which model.

**One condition per file.** Never mix models in a single `--out` — averaged
across models the scores mean nothing. `--resume` warns if you try.

If a run stops early on the daily cap, top up the gaps the next day instead of
repeating it:

```bash
python testing.py --resume results/llm_lite_zeroshot.json \
    --model gemini-3.5-flash-lite --n 100 \
    --data yiddish_parser/data/processed/supar_ready/test.txt
```

### Collect every run into one table

```bash
python scripts/llm_results_table.py "results/llm_*.json" --latex \
    | tee results/llm_comparison.md
```

That prints the Markdown table plus the LaTeX rows for Table 2, so no number is
retyped between the results and the report.

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
| Build data | `bash scripts/build_ppchy_data.sh` |
| Data statistics | `python scripts/dataset_stats.py --data-dir yiddish_parser/data/processed/supar_ready --markdown` |
| Split composition | `python scripts/split_provenance.py --write-labels` |
| Sampler self-test | `python src/train_parser_peft.py --selftest --backend maxmatch` |
| Train one cell | `sbatch src/parser_train_peft.slurm <mode> <adapter> <backend>` |
| Evaluate one cell | `python src/evaluate_peft.py --path output/parser_<cell>/yiddish_parser.pt --adapter-type <type>` |
| LLM baseline | `python testing.py --data .../test.txt --n 30 --out results/llm_zeroshot.json` |
| Watch jobs | `squeue -u $USER` · `tail -f logs/peft_<jobid>.out` |
