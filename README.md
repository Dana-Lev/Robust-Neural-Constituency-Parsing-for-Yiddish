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
tax" as an obstacle to syntactic parsing.

This project asks whether that bottleneck can be addressed **dynamically and
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
| `scripts/run_llm_baselines.sh` | The four LLM baseline runs, budgeted around the per-day quota; resumes rather than repeating |
| `scripts/llm_results_table.py` | Every LLM run collected into one comparison table, with LaTeX rows for the report |
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
python src/evaluate_peft.py --path ./output/parser_both_lora_maxmatch/yiddish_parser.pt --adapter-type lora
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

# all four conditions, budgeted around the per-day API quota
bash scripts/run_llm_baselines.sh day1    # both Lite conditions + frontier zero-shot
bash scripts/run_llm_baselines.sh day2    # frontier few-shot, on fresh quota
bash scripts/run_llm_baselines.sh table   # one comparison table + LaTeX rows
```

Free-tier limits are per day *and per model*, so the design uses two: a Flash
tier (20/day) for the headline frontier number and a Flash Lite tier (500/day)
for the large samples. [RUNBOOK.md](RUNBOOK.md) Step 8 has the details, including
what to do about a `503` wave.

Scoring follows the same conventions as the SuPar evaluation config (preterminals
excluded, `TOP`/punctuation deleted, co-indexation stripped, `ADVP`≡`PRT`), so the
Labeled F1 is directly comparable to the trained parser's. The script also reports
a syntactic validity rate and a token fidelity rate, which separate "the model
does not know Yiddish" from "the model does not know the PPCHY annotation scheme".

Running `testing.py` without `--data` falls back to 30 **synthetic** pilot
sentences. Those exist only to smoke-test the pipeline and must never be reported
as PPCHY results.

## Results

Constituency parsing on the PPCHY test set (856 sentences), frozen
`skulick/xlmb-ybc-ck05` encoder throughout. Full table, deltas and caveats in
[`results/parsing_results.md`](results/parsing_results.md).

| Cell | Trainable | UF | LF |
|---|---:|---:|---:|
| Frozen baseline | 11.86M | 85.24 | 74.89 |
| + Subword regularization | 11.86M | 85.58 | 75.32 |
| + Adapters (LoRA r=16) | 12.45M | 89.75 | **82.02** |
| + Adapters (Pfeiffer bottleneck) | 12.76M | 89.43 | **82.09** |
| + Both | 12.45M | 89.23 | 81.24 |

Resampling subword segmentation is worth +0.43 LF — smaller than the
seed-to-seed spread we measure, so not an effect we can detect. Adapters are
worth ~+7.1 LF, replicated across two unrelated adapter families that agree to
0.07 LF, and reproduced at a second seed. **The bottleneck is not how Yiddish
words are split; it is that the encoder cannot adapt.**

The best of five Gemini configurations reaches 55.37 LF on the same test
sentences and the same metric — 19.5 points below the frozen baseline and 26.7
below the best adapted parser
([`results/llm_comparison.md`](results/llm_comparison.md)).

## Data

No corpus data is redistributed in this repository. `yiddish_parser/data/README.md`
documents how to obtain PPCHY, Jochre and YBC, and the exact command sequence that
rebuilds the SuPar-ready splits.

## Attribution

The data-preparation and original training pipeline — the scripts listed
directly above this section's table as *inherited* — come from the course project
of **Amit Halfon and Omri Boiman**,
[Neural Constituency Parsing for Yiddish via Vocabulary Adaptation](https://github.com/omri-boiman/Neural-Constituency-Parsing-for-Yiddish-via-Vocabulary-Adaptation),
and are reused here with thanks so the corpus can be rebuilt end to end. Nothing
from that project is reported as a result in this one. The files listed in the
first table above — subword regularization, language adapters, the PEFT
entrypoint, the safe evaluator, and the LLM baseline — are new work.

## References

- Kulick, Ryant & Wallenberg (2022). *A Part-of-Speech Tagger for Yiddish.* LREC.
- Provilkov, Emelianenko & Voita (2020). *BPE-Dropout: Simple and Effective Subword Regularization.* ACL.
- Kudo (2018). *Subword Regularization.* ACL.
- Pfeiffer, Vulić, Gurevych & Ruder (2020). *MAD-X.* EMNLP.
- Hu et al. (2022). *LoRA.* ICLR.
- Zhang, Zhou & Li (2020). *Fast and Accurate Neural CRF Constituency Parsing.* IJCAI.
- Dobler & de Melo (2023). *FOCUS.* EMNLP.
- Urieli (2025). *Jochre Yiddish Corpus.*
