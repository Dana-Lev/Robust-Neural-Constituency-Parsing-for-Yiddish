# Parameter-Efficient Approaches to Subword Fragmentation

Two additions to the existing pipeline that attack the "tokenization tax" **without**
modifying the physical vocabulary (no `inject_vocab.py`) and **without** continued
pre-training (no `run_mlm.py`):

1. **Subword regularization** — stochastic segmentation in the dataloader.
2. **Language adapters** — `peft` adapters in a fully frozen XLM-R.

Both plug into the existing SuPar `CRFConstituencyParser` via runtime patches. **No file
under `supar/` is edited**, and none of the original repository scripts are modified —
the new entrypoint sits alongside `train_parser.py`.

---

## 1. Install

```bash
conda activate yiddish
pip install "peft>=0.7.0" sentencepiece
```

## 2. New files

| File | Role |
|---|---|
| `src/subword_regularization.py` | BPE-Dropout / Kudo sampling + the replacement training collator |
| `src/language_adapters.py` | LoRA + Pfeiffer bottleneck adapters, freezing, optimizer re-rating |
| `src/train_parser_peft.py` | Replaces `src/train_parser.py`. One entrypoint, four modes |
| `src/evaluate_peft.py` | Loads an adapted checkpoint **safely** and prints Labelled F1 |
| `src/parser_train_peft.slurm` | Replaces `src/parser_train.slurm` |

## 3. Where each patch attaches, and why there

Everything hangs off four facts about SuPar 1.1.4, all verified against the installed source:

| Seam | SuPar source | What we do |
|---|---|---|
| The tokenizer callable | `supar/parsers/const.py:240-248` — with `encoder='bert'`, the `WORD` field *is* a `SubwordField(name='words', tokenize=t.tokenize)` | Swap `field.tokenize` for a stochastic callable |
| The collator | `supar/utils/data.py:78-84` — `Dataset.build()` numericalizes **all** sentences once; `collate_fn=lambda x: Batch(x)` only stacks tensors | Replace `collate_fn` on the training loader so segmentation is resampled per batch |
| The backbone | `supar/models/model.py:94` — `model.encoder` is a `TransformerEmbedding`, `model.encoder.bert` is the HF `XLMRobertaModel` | Subclass `CRFConstituencyParser.MODEL`, freeze `encoder.bert`, inject adapters |
| The optimizer | `supar/parsers/parser.py:50-54` — `from transformers import AdamW` *inside* `train()`, one param group per parameter | Rebind `transformers.AdamW` to a factory that re-rates adapter params |

### Why the collator, and not just the field

This is the critical structural point. Handing a sampling tokenizer to the field alone
would accomplish nothing: `Dataset.build()` calls `self.transform(self.sentences)` **once**,
before epoch 1, and the DataLoader thereafter serves those frozen tensors. Each sentence
would receive exactly one random segmentation for the whole run. `install_subword_regularization()`
therefore wraps `Dataset.build` and, for the training split only, substitutes a collator
that re-runs `SubwordField.transform` on the raw words of each sentence in the batch.

The training split is identified by `shuffle=True`, which `Parser.train` passes only for
`train` (`parser.py:40-42`). Dev and test keep SuPar's default collator and stay
deterministic, so reported F1 is never computed on perturbed input.

### Why `inject_adapter_in_model` and not `get_peft_model`

`TransformerEmbedding.forward` calls `self.bert(subwords, attention_mask=...)[-1]` and
indexes into the output tuple to reach `hidden_states`. `inject_adapter_in_model` mutates
the module tree in place and returns a plain `XLMRobertaModel`, so that call site — and
SuPar's `state_dict` save/load round-trip — keep working untouched. A `PeftModel` wrapper
would insert a forward indirection and change the top-level key prefix for no benefit here.

## 4. Run it

```bash
cd yiddish_parser

# Inspect the sampler before spending GPU hours (prints fertility + examples)
python src/train_parser_peft.py --selftest --encoder-path skulick/xlmb-ybc-ck05

# The four ablation cells
python src/train_parser_peft.py --mode baseline    --encoder-path skulick/xlmb-ybc-ck05
python src/train_parser_peft.py --mode bpe-dropout --encoder-path skulick/xlmb-ybc-ck05 --backend maxmatch --p 0.1
python src/train_parser_peft.py --mode adapters    --encoder-path skulick/xlmb-ybc-ck05 --adapter-type lora
python src/train_parser_peft.py --mode both        --encoder-path skulick/xlmb-ybc-ck05 --adapter-type lora

# Or on the cluster
sbatch src/parser_train_peft.slurm both lora
```

Each mode writes to its own `./output/parser_<mode>[_<adapter-type>]/yiddish_parser.pt`, so
runs never overwrite each other or the existing `phase3_parser_model`.

`--mode baseline` reproduces the original script's setup, so all four numbers come from one
code path with identical seeding — which is what makes the comparison publishable.

## 5. Evaluation

```bash
python src/evaluate_peft.py --path ./output/parser_both_lora/yiddish_parser.pt --adapter-type lora
```

**Use this rather than `parser.evaluate` directly.** `Parser.load` calls
`model.load_state_dict(state['state_dict'], False)` — *strict=False*. Load an adapter
checkpoint without re-applying the adapter patch and the adapter tensors are dropped
silently; worse, LoRA injection renames `query.weight` to `query.base_layer.weight`, so the
**backbone** weights would be dropped silently too and you would report a number produced
by a partly random encoder. `load_adapted_parser` diffs the key sets and raises instead.

---

## 6. Approach 1 details — subword regularization

### XLM-R is Unigram, not BPE

BPE-Dropout (Provost & Titov, 2020) drops merge operations from a BPE merge table.
XLM-R's tokenizer is SentencePiece **Unigram**; it has no merge table, so BPE-Dropout is
not literally applicable. Two backends are provided, and the distinction is worth a
paragraph in the report:

| Backend | Method | Can emit FOCUS-injected tokens? |
|---|---|---|
| `spm` | Kudo (2018) sampling from the unigram lattice — `sp.encode(..., enable_sampling=True, alpha=α, nbest_size=-1)` | **No** — the SentencePiece model predates `add_tokens` |
| `maxmatch` | Stochastic longest-match over the HF vocabulary: with probability `p`, skip the longest matching piece and take the next-longest | **Yes** — operates on `tokenizer.get_vocab()` |

`maxmatch` is the closer behavioural analogue of BPE-Dropout ("skip a merge") and is the
one to use on `phase1_focus_model` / `phase2_trained`, since it is the only backend that can
reach the 1,875 injected tokens. That connects directly to your Zombie Token finding: it
offers a *training-time* route to activating those tokens that costs no pre-training at all.

### The `--anchor` flag

`maxmatch`'s greedy base segmentation is not XLM-R's Viterbi segmentation. Left unhandled,
that means the parser trains on one canonical segmentation and is evaluated on another.

- `--anchor hf` (default): when no dropout event fires, return the **HF** segmentation.
  Dropout perturbs *around* the evaluation-time form — BPE-Dropout's own semantics, zero
  train/test skew. At `p=0.1` roughly 90% of draws are the canonical segmentation.
- `--anchor greedy`: return the greedy longest-match form, deliberately biasing the base
  distribution towards whole-word pieces. Use this if activating injected tokens is the
  goal, and report the skew.

### Safety properties

Verified by construction in the sampler:

- Sampled pieces are **always** checked against `tokenizer.get_vocab()`; any OOV piece
  triggers a fallback to the deterministic segmentation, so a sampled word can never
  degrade into `<unk>`.
- Single-piece words are never resegmented — this also protects whole-word FOCUS tokens.
- Segmentations longer than `--fix-len` fall back to deterministic (see the gotcha below).
- Only strings are emitted, never ids. `SubwordField.transform` does the vocabulary lookup
  itself, so XLM-R's fairseq id offset never enters the picture.
- Sampling uses a private `random.Random(seed)`, so segmentation noise is reproducible and
  does not perturb the torch RNG stream that drives dropout and initialization.

### `--fix-len`: the one silent failure mode

`SubwordField.transform` truncates each word to `fix_len` pieces **without warning**
(`supar/utils/field.py:329-331`), and SuPar's default is 20. Sampled segmentations are longer
than Viterbi ones, so the default is raised to **32** in `train_parser_peft.py`. The
`--selftest` output prints `max_pieces_observed`; if it approaches `--fix-len`, raise it.

---

## 7. Approach 2 details — MAD-X-style language adapters

### What gets frozen

```
freeze_backbone(model.encoder.bert)   # every backbone parameter, then re-enable adapters only
```

For XLM-R base the input embedding matrix is `250002 × 768 = 192,001,536` parameters — the
192M figure — and `freeze_backbone` **asserts** it stays frozen rather than assuming it.
After FOCUS injection it is `252002 × 768 ≈ 193.5M`; the assertion covers both.

This also replaces the original `apply_freeze_patch()`, whose primary check was

```python
if 'Embeddings' in name and 'Bert' in name:
```

`'Bert' in 'XLMRobertaEmbeddings'` is `False` (case-sensitive; the class contains `berta`),
so on XLM-R that branch never fired and only the generic fallback scan did — and that
fallback matches `XLMRobertaEmbeddings`, freezing the embeddings *only* and leaving all 12
transformer layers trainable at `lr=5e-5` (SuPar hard-codes `requires_grad=True` on the
backbone and ignores the `finetune: False` config value).

The README's own arithmetic corroborates this: "trainable parameters under 100M" is only
reachable if the ~86M of layer weights *are* in the optimizer (86M layers + ~8M head ≈ 94M),
which contradicts the adjacent claim that "the entire XLM-R backbone (278M parameters) is
frozen". Both statements cannot hold. **This is worth re-checking in your logs before
submission**, because it changes what the 83.81 baseline is a baseline *of* — a partially
fine-tuned encoder, not a frozen one. `freeze_backbone` freezes the whole backbone explicitly
and `describe_trainable` prints the breakdown with an explicit warning if any backbone
parameter is still trainable, so the claim becomes something the log verifies.

### Adapter families

`--adapter-type lora` — `peft.LoraConfig(r=16, lora_alpha=32, target_modules=["query","value"])`,
injected in place. `dense` is deliberately excluded from the targets: in XLM-R it names three
different sublayers, so matching it is ambiguous.

`--adapter-type bottleneck` — a faithful Pfeiffer adapter (MAD-X; Pfeiffer et al. 2020):
`h + W_up(GELU(W_down(h)))` after each layer's output LayerNorm, `hidden//16 = 48`
bottleneck. `peft` ships no bottleneck tuner, so this uses forward hooks. That choice is
load-bearing: hooks add **zero** entries under the XLM-R submodule names, so the frozen
backbone's `state_dict` keys stay identical to the baseline's, and the adapter parameters
live separately as `encoder.language_adapters.*` where SuPar still saves them.
`W_up` is zero-initialised, so at step 0 the adapted model *is* the frozen baseline —
the adapters cannot damage the starting point.

MAD-X also uses invertible adapters on the embedding layer to handle unseen scripts.
Those are not implemented here: they are the component that needs unlabelled MLM
pre-training to be useful, which is exactly what these approaches are meant to avoid.
Worth one sentence in Future Work.

### Adapter learning rate — the easiest thing to get wrong

SuPar assigns `args.lr` (5e-5) to every parameter whose name starts with `encoder`, and
`args.lr * lr_rate` (1e-3) to everything else. Adapters live under `encoder.bert.*`, so
they would inherit **5e-5** — far too low for randomly initialised LoRA/bottleneck weights,
which typically want 1e-4 … 1e-3. A run left at 5e-5 will look like "adapters don't help"
when the real story is an under-trained adapter.

`install_adapter_optimizer` intercepts the optimizer, gives adapter parameters
`--adapter-lr` (default 3e-4), leaves the head and scalar mix at SuPar's rates, and drops
frozen parameters from the optimizer state entirely. It works by rebinding
`transformers.AdamW` — which `train_parser.py` already does at line 9, so this extends an
existing seam rather than inventing one. **Sweep `--adapter-lr` over {1e-4, 3e-4, 1e-3}**
before drawing any conclusion.

### Memory on a 12GB GPU

LoRA `r=16` on query/value adds ~1.2M trainable parameters (~0.4% of the 278M backbone);
bottleneck at reduction 16 adds ~1.8M. With the backbone frozen there are no optimizer
moments for 278M parameters — that saving (roughly 2.2GB of Adam state) is what keeps the
run comfortable at `--batch-size 2000`. If it is still tight, add `--grad-checkpointing`
(~30% slower steps, large activation saving). It passes `use_reentrant=False` deliberately:
with frozen embeddings no *input* tensor requires grad, and the reentrant implementation
would silently drop the graph.

---

## 8. Things to be aware of when writing this up

**"scalar_mix" was never a real pooling mode.** In supar 1.1.4,
`TransformerEmbedding` *always* applies a `ScalarMix` over the top
`n_bert_layers`; the `bert_pooling` argument only chooses how subword pieces
are pooled into a word vector (`first`/`last`, anything else = mean). The
previous project's report says it switched `bert_pooling` to `'scalar_mix'` —
that value silently falls into the mean branch. The configuration difference
that actually mattered there was `n_bert_layers` (4 in the repo script vs 12
claimed in the report). `train_parser_peft.py` exposes both knobs
(`--bert-pooling`, `--n-bert-layers`); to chase the 83.81 configuration, use
`--n-bert-layers 12`.

**The parser is chart-based, not transition-based.** `CRFConstituencyParser` is the CRF
chart parser of Zhang et al. (2020): it scores all spans and decodes with CKY (`--mbr` runs
MBR decoding over marginals). There is no transition system, action sequence, or stack. Both
approaches here are agnostic to that — they touch the input segmentation and the encoder,
not the decoder — but the write-up should say "chart-based CRF" to match the code.

**Expect the fragmentation fix to be roughly F1-neutral, and treat that as the finding.**
Your existing result is that FOCUS+DAPT moved Labelled F1 by −0.79 (83.81 → 83.02) despite
raising injected-token coverage from 0% to 43.7%. The most defensible reading is that a
frozen encoder already extracts enough structural signal from fragmented subwords, so
orthographic repair adds little to *this* task. If so, these two approaches should also land
near 83.8 — and that is a result worth reporting, because it now rests on three
independent interventions (embedding initialization, continued pre-training, and
parameter-efficient adaptation) rather than one. The value of the PEFT framing is the cost
side: it reaches the same place for ~1-2M trainable parameters and no pre-training run.

The comparison that would most strengthen the argument is `--mode adapters` with the
backbone **unfrozen** at the top few layers — the "targeted unfreezing" already listed in
your Future Work. If adapters and unfreezing both stay flat, the resilience claim is solid.

---

## References

- Provost & Titov (2020). *BPE-Dropout: Simple and Effective Subword Regularization.* ACL.
- Kudo (2018). *Subword Regularization: Improving NMT Models with Multiple Subword Candidates.* ACL.
- Pfeiffer et al. (2020). *MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer.* EMNLP.
- Houlsby et al. (2019). *Parameter-Efficient Transfer Learning for NLP.* ICML.
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
- Zhang et al. (2020). *Fast and Accurate Neural CRF Constituency Parsing* (SuPar). ACL.
