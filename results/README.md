# Results

Small, human-readable artifacts that back the numbers in the report. These *are*
committed — they are the evidence trail for reproducibility.

| File | Produced by |
|---|---|
| `dataset_stats.md` | `scripts/dataset_stats.py --markdown` |
| `selftest_maxmatch.txt` | `train_parser_peft.py --selftest` |
| `sanity_overfit.txt`, `.err` | the 100-sentence overfitting sanity check |
| `eval_<cell>.txt` | `evaluate_peft.py`, one per experiment cell |
| `eval_<cell>_seed2.txt` | the baseline and LoRA cells retrained at seed 2 |
| `parsing_results.md` | the assembled main results table, deltas and seed replication |
| `llm_frontier_{zero,few}shot.json` | `testing.py --model gemini-3.5-flash` |
| `llm_lite_{zero,few}shot.json` | `testing.py --model gemini-3.5-flash-lite` |
| `llm_baseline.md`, `llm_comparison.md` | `scripts/llm_results_table.py` |

Model checkpoints, Slurm logs and corpora stay out of git (see `.gitignore`).
