# Results

Small, human-readable artifacts that back the numbers in the report. These *are*
committed — they are the evidence trail for reproducibility.

| File | Produced by |
|---|---|
| `dataset_stats.md` | `scripts/dataset_stats.py --markdown` |
| `selftest_<backend>.txt` | `train_parser_peft.py --selftest` |
| `sanity_overfit.txt` | the small-subset overfitting check (RUNBOOK Step 5) |
| `eval_<cell>.txt` | `evaluate_peft.py` output per experiment cell |
| `parsing_results.md` | the assembled main results table |
| `llm_zeroshot.json`, `llm_fewshot.json` | `testing.py --out ...` |

Model checkpoints, Slurm logs and corpora stay out of git (see `.gitignore`).
