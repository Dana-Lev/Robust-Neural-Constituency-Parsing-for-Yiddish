# Project plan — Yiddish parsing via subword regularization & PEFT

**Deadline: September 30, 2026.** Last updated: August 23, 2026.
Team: **Dana Lev** and **Ayala May**.

**Status: all experiments are finished.** Every number in the report is a real
measurement with committed raw output behind it. What remains is writing,
review, and the final compile.

---

## Done

### Data
- PPCHY rebuilt from `ppchyprep` through `build_final_trees.py` →
  `split_supar_data.py` → `clean_tree_data.py`. Splits: 15,394 / 855 / 856
  trees (190,007 tokens), 226 labels after stripping co-indexation. No tree
  rejected by the cleaning step.
- Statistics and per-split subword fertility: `results/dataset_stats.md`.
- Provenance per split tree recorded by `scripts/split_provenance.py`, so a
  per-component breakdown is recoverable without retraining.

### Infrastructure
- Both approaches implemented as **runtime patches** over SuPar 1.1.4 — no file
  under `supar/` edited. Design and patch seams: `PEFT_INTEGRATION.md`.
- Segmentation sampler verified before spending GPU hours:
  `results/selftest_maxmatch.txt` (fertility 2.04 → 2.55, 31.7% of words
  varying between epochs).
- **Sanity overfit** (rubric requirement): 97.99 LF on a 100-sentence subset —
  `results/sanity_overfit.txt`. This is what licenses reading a flat cell as a
  property of the intervention rather than a broken setup.

### Experiment grid — all five cells, one code path, seed 1
| Cell | LF | Raw output |
|---|---:|---|
| Frozen baseline | 74.89 | `results/eval_baseline.txt` |
| + Subword regularization (maxmatch) | 75.32 | `results/eval_bpe-dropout_maxmatch.txt` |
| + Adapters (LoRA r=16) | 82.02 | `results/eval_adapters_lora.txt` |
| + Adapters (Pfeiffer bottleneck) | 82.09 | `results/eval_adapters_bottleneck.txt` |
| + Both | 81.24 | `results/eval_both_lora_maxmatch.txt` |

### Seed replication
Baseline and LoRA retrained at seed 2 (`eval_*_seed2.txt`). Spread is 0.22 LF
(baseline) and 0.51 LF (LoRA); the adapter gain replicates at +6.84 against
+7.13. This is what let us say the +0.43 subword gain is below detection rather
than merely small.

### Frontier-LLM control
Four reported conditions across two Gemini tiers, all at 100% coverage
(`results/llm_comparison.md`), plus one superseded pilot documented in a
footnote. Best configuration: 55.37 LF, 19.5 below the frozen baseline.

### Report
All sections written with real numbers: Abstract, Introduction, Background,
Methodology, Experimental Setup, Results, Discussion, Conclusion, Limitations,
and four appendices (hyperparameters, seed replication, prompts, label
confusions).

---

## Left to do

| # | Task | Owner | Notes |
|---|---|---|---|
| 1 | **Review the AI Disclosure section** | Dana | `report/main.tex` — the draft is an AI-written account of the collaboration. Rewrite anything that does not match your own recollection; you own every claim in it. **Mandatory section.** |
| 2 | **Compile on Overleaf** | either | Nothing in `main.tex` has been through a compiler yet. Check the ≤8-page body limit (references and appendices excluded) and that no table floats away from its section. |
| 3 | Full proofread | both | Read for work-log phrasing ("we tried X, it failed") — the guidelines penalise it. |
| 4 | Self-grade against the rubric | both | Research question / ambition / lit review / methodology / results / presentation. |
| 5 | Optional figures | either | Three `\todo` markers, all optional: an example tree, a training curve from the per-epoch dev scores, a Gemini failure case. Cut them if the page limit bites. |
| 6 | **Revoke the exposed Gemini API key** | Dana | It was hardcoded in an early `testing.py`. Not in the current file, but it was live. |
| 7 | Final submission | both | Sept 30. |

### Deliberately not run
Documented in RUNBOOK Step 6b so nobody assumes these exist:

- **Adapter learning-rate sweep.** Insurance against "adapters don't help" being
  an under-training artifact. The adapters gained ~7 LF at 3e-4, so the
  insurance was never needed. The report does **not** claim a sweep was done.
- **Seed 3**, and a 12-layer scalar-mix baseline.

---

## Risks / gotchas that still apply

- **Page limit.** Body must be ≤8 pages. The report is dense and has never been
  compiled. This is the most likely late surprise — check it early, not on
  Sept 29.
- **No API key anywhere in the repo or its history.** Reproducibility is graded,
  and a leaked key is worse than a lost point.
- **Report numbers only from the real `supar_ready` splits.** `testing.py`
  without `--data` falls back to 30 synthetic sentences; those must never be
  reported as PPCHY results.
- **`evaluate_peft.py`, never `Parser.load`, for adapter checkpoints.** SuPar
  loads with `strict=False` and will silently drop adapter *and* backbone
  tensors, then print a confident number from a partly random encoder.
