# Frontier-LLM baseline — zero-shot

Model `gemini-3.6-flash`, temperature 0, 30 sentences sampled from the PPCHY test
split (seed 1, ≤40 tokens). Raw records: `llm_zeroshot.json`.

| Metric | Value | Over |
|---|---:|---|
| Coverage | 73.3% | 22 of 30 requests returned; 8 lost to free-tier rate limits |
| Syntactic validity | 100.0% | responses received |
| Token fidelity | 95.5% | responses received |
| Labeled F1 (micro) | **54.67** | scored constituents |
| Unlabeled F1 (micro) | **79.58** | scored constituents |
| Labeled P / R | 51.97 / 57.66 | |
| Unlabeled P / R | 75.66 / 83.94 | |
| Labeled F1 (macro) | 51.55 | per-sentence mean over responses received |

## Reading these numbers

**Bracketing is decent, labeling is not.** Unlabeled F1 exceeds labeled F1 by
24.9 points, and 18 of 22 scored sentences show the same pattern. The model
recovers much of the tree shape and then labels it wrongly.

**The label errors are scheme-variant confusions, not noise.** Of 152 predicted
constituents, 19 (12.5%) fall outside this treebank's 226-label inventory,
dominated by `NP-OB1` (11) and `NP-OB2` (2) — Penn-historical labels from a
sibling corpus, where this PPCHY release writes `NP-ACC` and `NP-DTV`.

**It does not displace a trained parser.** 54.67 LF is ~29 points below the
frozen-encoder baseline, and the model's *unlabeled* score still does not reach
the baseline's *labeled* score.

## Caveats

- Coverage is 73.3%, so the interval is wide. A re-run with a larger `--sleep`
  (or API credits) would tighten it.
- An API failure is not a model failure: all behavioural rates above are computed
  over responses received. Re-derive them from any saved run with
  `python testing.py --recompute results/llm_zeroshot.json`.
- Model versions change without notice; the identifier and access date belong in
  the report.

## Still to run

3-shot (`--shots 3 --train-file .../train.txt`). This is the decisive follow-up:
if three exemplars move labeled F1 sharply, the deficit is conventional; if they
do not, it is a real limit on zero-shot structural analysis of a low-resource
historical language.
