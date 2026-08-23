# Results

Small, human-readable artifacts that back the numbers in the report. These *are*
committed — they are the evidence trail for reproducibility.

| File | Produced by |
|---|---|
| `dataset_stats.md` | `scripts/dataset_stats.py --markdown` |
| `selftest_maxmatch.txt` | `train_parser_peft.py --selftest` |
| `sanity_overfit.txt`, `.err` | the small-subset overfitting check (RUNBOOK Step 5) |
| `eval_<cell>.txt` | `evaluate_peft.py`, one per experiment cell |
| `eval_<cell>_seed2.txt` | the same cells retrained at seed 2 (RUNBOOK Step 6b) |
| `parsing_results.md` | the assembled main results table, deltas and seed replication |
| `llm_frontier_{zero,few}shot.json` | `testing.py --model gemini-3.5-flash` |
| `llm_lite_{zero,few}shot.json` | `testing.py --model gemini-3.5-flash-lite` |
| `llm_zeroshot.json` | **superseded pilot** — see below |
| `llm_baseline.md`, `llm_comparison.md` | `scripts/llm_results_table.py` |

`llm_zeroshot.json` is the first exploratory run: `gemini-3.6-flash`, zero-shot,
30 test sentences attempted and 22 returned before the daily quota stopped it.
It is kept because the report's label-confusion appendix is a manual audit of
those 22 answers, and because deleting a run that scored *above* one of the
reported cells would be the wrong instinct. It is not a headline result: the
coverage is incomplete and the sample differs from both reported tiers.

Model checkpoints, Slurm logs and corpora stay out of git (see `.gitignore`).
