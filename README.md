# Robust Neural Constituency Parsing for Yiddish

**via Subword Regularization and Parameter-Efficient Fine-Tuning**

NLP final project — Tel Aviv University, 2025b (Dr. Mor Geva).
Dana Lev · Ayala May

---

> **Building this from scratch?** [RUNBOOK.md](RUNBOOK.md) is the step-by-step
> guide: cluster setup, data construction, the experiment grid, evaluation, and
> pushing results back to git.

## Research question

Yiddish is fragmented aggressively by multilingual subword tokenizers. Kulick et
al. (2022) built the first Yiddish POS tagger and identified this "tokenization
tax" as an obstacle to syntactic parsing. A previous course project attacked it
*statically* — injecting clean Jochre vocabulary with FOCUS and running
domain-adaptive pre-training (DAPT) — and found that although the injected tokens
became active, downstream parsing F1 did not improve (83.81 → 83.02 Labeled F1).

This project asks whether the same bottleneck can be addressed **dynamically and
parameter-efficiently**, without modifying the physical vocabulary and without any
continued pre-training:

1. **Subword regularization** — stochastic re-segmentation at training time
   (BPE-Dropout in spirit; Kudo unigram-lattice sampling in practice, since
   XLM-R's tokenizer is SentencePiece Unigram, not BPE).
2. **Language adapters** — MAD-X-style adaptation of a fully frozen backbone,
   via LoRA and Pfeiffer bottleneck adapters (~1–2M trainable parameters).

A third experiment answers a question that the pace of the field forces on any
2026 project built on a 2022 problem statement: **can a frontier LLM already do
this zero-shot?** `testing.py` measures that directly.

## Repository layout

| Path | What it is |
|---|---|
| `yiddish_parser/src/subword_regularization.py` | Approach 1. Stochastic segmentation sampler + the replacement training collator |
| `yiddish_parser/src/language_adapters.py` | Approach 2. LoRA + Pfeiffer bottleneck adapters, backbone freezing, optimizer re-rating |
| `yiddish_parser/src/train_parser_peft.py` | Single entrypoint for all four ablation modes |
| `yiddish_parser/src/evaluate_peft.py` | Safe evaluation of adapter checkpoints (see the warning below) |
| `yiddish_parser/src/parser_train_peft.slurm` | Cluster job script |
| `PEFT_INTEGRATION.md` | Design document: where each runtime patch attaches to SuPar 1.1.4, and why there |
| `testing.py` | Frontier-LLM (Gemini) zero-/few-shot parsing baseline with EVALB-style scoring |
| `report/` | ACL-format LaTeX report (`main.tex`, `references.bib`, ACL style files) |
| `PLAN.md` | Work plan and division of labour through the submission deadline |
| `yiddish_parser/data/README.md` | How to obtain and rebuild the corpora (no corpus data is redistributed here) |
| `RUNBOOK.md` | Step-by-step build, train, evaluate and publish procedure for the Slurm cluster |
| `scripts/build_ppchy_data.sh` | One command from ppchyprep JSON to SuPar-ready splits |
| `scripts/dataset_stats.py` | Sentence/token counts, label distribution and subword fertility per split |
| `scripts/split_provenance.py` | Which PPCHY component each split tree came from, for per-component result breakdowns |
| `results/` | Committed evidence trail: statistics, sanity checks, evaluation output |

Inherited pipeline scripts — `train_parser.py`, `inject_vocab.py`, `run_mlm.py`,
`eval_token_usage.py`, `clean_tree_data.py`, `split_supar_data.py`,
`data_extraction/`, `ppchy_formatting/` — come from the previous project (see
[Attribution](#attribution)) and are kept so the full pipeline stays reproducible.

## Setup

```bash
conda create -n yiddish python=3.10 && conda activate yiddish
pip install -r requirements.txt
```

That is the local recipe. **On the TAU cluster there is no usable system conda**
— you install Miniconda into your course storage first, and the environments
have to be split in two because `ppchyprep` pins conflicting dependency
versions. [RUNBOOK.md](RUNBOOK.md) Step 2 covers it.

The parser is built on [SuPar](https://github.com/yzhangcs/parser) 1.1.4. Every
adaptation is a **runtime patch**: no file under `supar/` is edited, and no
original pipeline script is modified.

## Running the experiments

All four ablation cells come from one entrypoint with identical seeding, which is
what makes them comparable:

```bash
cd yiddish_parser

# Inspect the segmentation sampler before spending GPU hours
python src/train_parser_peft.py --selftest --encoder-path skulick/xlmb-ybc-ck05

python src/train_parser_peft.py --mode baseline    --encoder-path skulick/xlmb-ybc-ck05
python src/train_parser_peft.py --mode bpe-dropout --encoder-path skulick/xlmb-ybc-ck05 --backend maxmatch --p 0.1
python src/train_parser_peft.py --mode adapters    --encoder-path skulick/xlmb-ybc-ck05 --adapter-type lora
python src/train_parser_peft.py --mode both        --encoder-path skulick/xlmb-ybc-ck05 --adapter-type lora
```

On the Slurm cluster (set `NLP_STORAGE` to your course storage directory first):

```bash
sbatch src/parser_train_peft.slurm both lora
```

Each cell writes to its own directory, named after every setting that
distinguishes it — mode, adapter type, backend, layer count, seed — so no two
runs of the grid can overwrite each other's checkpoints.

### Evaluation

```bash
python src/evaluate_peft.py --path ./output/parser_both_lora/yiddish_parser.pt --adapter-type lora
```

**Use this instead of `CRFConstituencyParser.load` for adapter checkpoints.**
SuPar loads with `strict=False`, so an adapter checkpoint loaded without the
adapter patch silently drops the adapter tensors — and because LoRA injection
renames `query.weight` to `query.base_layer.weight`, it silently drops the
*backbone* weights too. You would then report a number produced by a partly
random encoder. `load_adapted_parser` diffs the key sets and raises instead.

### Frontier-LLM baseline

```bash
export GEMINI_API_KEY="your-key"          # never commit a key

# Zero-shot on real PPCHY test trees
python testing.py --data yiddish_parser/data/processed/supar_ready/test.txt --n 30

# Few-shot, with exemplars drawn from the training split only
python testing.py --data .../test.txt --n 30 --shots 3 --train-file .../train.txt
```

Scoring follows the same conventions as the SuPar evaluation config (preterminals
excluded, `TOP`/punctuation deleted, co-indexation stripped, `ADVP`≡`PRT`), so the
Labeled F1 is directly comparable to the trained parser's. The script also reports
a syntactic validity rate and a token fidelity rate, which separate "the model
does not know Yiddish" from "the model does not know the PPCHY annotation scheme".

Running `testing.py` without `--data` falls back to 30 **synthetic** pilot
sentences. Those exist only to smoke-test the pipeline and must never be reported
as PPCHY results.

## Data

No corpus data is redistributed in this repository. `yiddish_parser/data/README.md`
documents how to obtain PPCHY, Jochre and YBC, and the exact command sequence that
rebuilds the SuPar-ready splits.

## Attribution

This project builds on the course project of **Amit Halfon and Omri Boiman**,
[Neural Constituency Parsing for Yiddish via Vocabulary Adaptation](https://github.com/omri-boiman/Neural-Constituency-Parsing-for-Yiddish-via-Vocabulary-Adaptation),
which established the first neural constituency parser for Yiddish and the
FOCUS + DAPT results used here as reference numbers. Their pipeline scripts are
reused, with thanks, as the structural foundation; the files listed in the first
table above (subword regularization, language adapters, the PEFT entrypoint, the
safe evaluator, and the LLM baseline) are new work for this project.

## References

- Kulick, Ryant & Wallenberg (2022). *A Part-of-Speech Tagger for Yiddish.* LREC.
- Provilkov, Emelianenko & Voita (2020). *BPE-Dropout: Simple and Effective Subword Regularization.* ACL.
- Kudo (2018). *Subword Regularization.* ACL.
- Pfeiffer, Vulić, Gurevych & Ruder (2020). *MAD-X.* EMNLP.
- Hu et al. (2022). *LoRA.* ICLR.
- Zhang, Zhou & Li (2020). *Fast and Accurate Neural CRF Constituency Parsing.* IJCAI.
- Dobler & de Melo (2023). *FOCUS.* EMNLP.
- Urieli (2025). *Jochre Yiddish Corpus.*
