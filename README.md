# Robust Neural Constituency Parsing for Yiddish

**via Subword Regularization and Parameter-Efficient Fine-Tuning**

NLP final project — Tel Aviv University, 2025b (Dr. Mor Geva)
Dana Lev · Ayala May

The report is [`report/main.tex`](report/main.tex).

---

## Research question

Multilingual subword vocabularies were not built for Hebrew-script Yiddish, so
words fragment aggressively — 2.05 XLM-R pieces per word in our data. Kulick et
al. (2022) identified this "tokenization tax" as an obstacle to building the
syntactic tools the language lacks. The established remedy is *static*: inject
clean in-language vocabulary into the embedding matrix and consolidate it with
continued pre-training.

We ask whether the bottleneck yields instead to **dynamic, parameter-efficient
adaptation** — neither of which touches the vocabulary or requires pre-training:

1. **Subword regularization** — resampling the segmentation every training
   batch, so the model cannot rely on any single one. XLM-R's tokenizer is
   SentencePiece *Unigram*, so BPE-Dropout does not apply literally; we
   implement Kudo lattice sampling and a maxmatch behavioural analogue.
2. **Language adapters** — MAD-X-style adaptation of a *fully frozen* encoder,
   via LoRA and Pfeiffer bottleneck adapters.

A third experiment answers the objection any 2026 project built on a 2022
problem statement must face: **can a frontier LLM already do this?**

## Results

Constituency parsing on the PPCHY test set (856 sentences), frozen
`skulick/xlmb-ybc-ck05` encoder throughout, one code path per cell.

| Cell | Trainable | UF | LF |
|---|---:|---:|---:|
| Frozen baseline | 11.86M | 85.24 | 74.89 |
| + Subword regularization | 11.86M | 85.58 | 75.32 |
| + Adapters (LoRA, r=16) | 12.45M | 89.75 | **82.02** |
| + Adapters (Pfeiffer bottleneck) | 12.76M | 89.43 | **82.09** |
| + Both | 12.45M | 89.23 | 81.24 |

**The two interventions disagree decisively.** Resampling segmentation is worth
+0.43 LF — smaller than the seed-to-seed spread we measure, so not an effect we
can detect. Adapters are worth ~+7.1 LF, replicated across two architecturally
unrelated adapter families that agree to 0.07 LF, and reproduced at a second
seed. Combining them is *worse* than adapters alone.

> The bottleneck is not how Yiddish words are split. It is that the encoder is
> not permitted to adapt.

**Frontier-LLM control.** The best of four Gemini configurations reaches 55.37
labeled F1 on the same sentences under the same metric — 19.5 points below the
frozen baseline and 26.7 below the best adapted parser. Most of the apparent
benefit of in-context examples turns out to be improved instruction-following
rather than improved parsing: the few-shot gain falls from +9.8/+13.0 to
+4.90/+4.94 once restricted to answers that reproduce the supplied tokens.

Full tables, deltas, seed replication and caveats:
[`results/parsing_results.md`](results/parsing_results.md) and
[`results/llm_comparison.md`](results/llm_comparison.md).

## What is in this repository

| Path | What it is |
|---|---|
| `yiddish_parser/src/subword_regularization.py` | Approach 1 — stochastic segmentation sampler and the replacement training collator |
| `yiddish_parser/src/language_adapters.py` | Approach 2 — LoRA and Pfeiffer adapters, backbone freezing, optimizer re-rating |
| `yiddish_parser/src/train_parser_peft.py` | One entrypoint for all ablation cells |
| `yiddish_parser/src/evaluate_peft.py` | Checkpoint-safe evaluation of adapter models |
| `testing.py` | Frontier-LLM baseline with EVALB-style scoring |
| `report/` | The ACL-format report and its style files |
| `results/` | Committed evidence: every number in the report, with its raw output |
| `scripts/` | Data build, dataset statistics, split provenance, LLM result tables |
| `PEFT_INTEGRATION.md` | Design document: where each runtime patch attaches to SuPar 1.1.4, and why there |
| `yiddish_parser/data/README.md` | How to obtain PPCHY and rebuild the splits |

Both approaches are implemented as **runtime patches** over
[SuPar](https://github.com/yzhangcs/parser) 1.1.4: no file under `supar/` is
edited. `PEFT_INTEGRATION.md` documents each patch seam.

Two properties of the library make naive implementations silently wrong, and
both are handled explicitly:

- `Dataset.build` numericalises the corpus **once**, before training, so a
  stochastic tokenizer attached to the field is sampled once per sentence for
  the whole run. Resampling must replace the training collator instead.
- `model.py:99` hardcodes `requires_grad=True` on the backbone for
  `encoder='bert'`, so SuPar's `finetune=False` does nothing. Freezing must be
  applied after construction and asserted.

## Data

No corpus data is redistributed here. See
[`yiddish_parser/data/README.md`](yiddish_parser/data/README.md).

## References

- Kulick, Ryant & Wallenberg (2022). *A Part-of-Speech Tagger for Yiddish.* LREC.
- Provilkov, Emelianenko & Voita (2020). *BPE-Dropout.* ACL.
- Kudo (2018). *Subword Regularization.* ACL.
- Pfeiffer, Vulić, Gurevych & Ruder (2020). *MAD-X.* EMNLP.
- Hu et al. (2022). *LoRA.* ICLR.
- Zhang, Zhou & Li (2020). *Fast and Accurate Neural CRF Constituency Parsing.* IJCAI.
- Conneau et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL.

## Attribution

The PPCHY data-preparation scripts (`src/ppchy_formatting/`,
`split_supar_data.py`, `clean_tree_data.py`) are reused, with thanks, from the
course project of Amit Halfon and Omri Boiman,
[Neural Constituency Parsing for Yiddish via Vocabulary Adaptation](https://github.com/omri-boiman/Neural-Constituency-Parsing-for-Yiddish-via-Vocabulary-Adaptation),
so the corpus can be rebuilt end to end. No result from that project is reported
in this one. Everything else in the table above is new work.
