# Constituency parsing on the PPCHY test set

All cells: frozen `skulick/xlmb-ybc-ck05` encoder (278.0M parameters, none
trained), one entrypoint, seed 1, early stopping on dev with patience 10.
Raw evaluation output in `eval_*.txt`; per-epoch curves in the job logs.

| Cell | Trainable | UF | LF | UCM | LCM |
|---|---:|---:|---:|---:|---:|
| Frozen baseline | 11,862,119 | 85.24 | 74.89 | 43.93 | 31.43 |
| + Subword reg. (maxmatch, p=0.1) | 11,862,119 | 85.58 | **75.32** | 45.21 | 33.29 |
| + Adapters (LoRA r=16) | 12,451,943 | 89.75 | **82.02** | 58.53 | 46.73 |
| + Adapters (Pfeiffer bottleneck) | 12,756,647 | 89.43 | **82.09** | 56.66 | 45.09 |
| + Both (maxmatch + LoRA) | 12,451,943 | 89.23 | **81.24** | 55.84 | 42.99 |

Change against the baseline:

| Cell | ΔUF | ΔLF | ΔUCM | ΔLCM |
|---|---:|---:|---:|---:|
| + Subword reg. | +0.34 | **+0.43** | +1.28 | +1.86 |
| + Adapters (LoRA) | +4.51 | **+7.13** | +14.60 | +15.30 |
| + Adapters (bottleneck) | +4.19 | **+7.20** | +12.73 | +13.66 |
| + Both | +3.99 | **+6.35** | +11.91 | +11.56 |

## What this shows

**Segmentation is not the bottleneck.** Resampling subword segmentation every
batch — verified active, fertility 2.04 → 2.55, 31.7% of words varying — moves
labeled F1 by +0.43.

**Encoder capacity is.** Both adapter families gain ~+7.2 LF while leaving
segmentation untouched, for 589,824 (LoRA) and 894,528 (bottleneck) parameters:
under 0.35% of the frozen backbone, no pre-training.

**Two unrelated adapter designs agree to 0.07 LF.** Low-rank attention updates
versus bottleneck residuals after each LayerNorm. The bottleneck's up-projection
is zero-initialised, so at step 0 that model *is* the baseline — every point was
learned by 894,528 parameters starting from an identity function.

**Combining them is worse than adapters alone** (−0.78 LF, −3.74 LCM against
LoRA). Segmentation noise is not free once the encoder can adapt.

**Labels are harder than brackets**: the baseline's UF−LF gap is 10.35 points,
narrowing to 7.73 with adapters. PPCHY's 226 function-tagged labels are what
makes this task demanding, and it is the same asymmetry the LLM control shows far
more severely.

## Caveats

- Every cell is a single seed. The ~7-point adapter gap is far too large to be
  seed noise; the +0.43 subword result is not.
- The `spm` (Kudo lattice) backend was unavailable in this environment, so only
  the `maxmatch` backend was evaluated.
- Sanity check: the same pipeline reaches 97.99 LF overfitting a 100-sentence
  subset, so a flat cell reflects the intervention, not a broken setup.

## Reproduce

```bash
sbatch src/parser_train_peft.slurm baseline
sbatch src/parser_train_peft.slurm bpe-dropout lora maxmatch
sbatch src/parser_train_peft.slurm adapters   lora
sbatch src/parser_train_peft.slurm adapters   bottleneck
sbatch src/parser_train_peft.slurm both       lora maxmatch

python src/evaluate_peft.py --path ./output/parser_<cell>/yiddish_parser.pt \
    --adapter-type <none|lora|bottleneck> --device cpu
```
