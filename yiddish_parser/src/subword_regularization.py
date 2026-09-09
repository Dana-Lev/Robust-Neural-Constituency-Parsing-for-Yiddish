# -*- coding: utf-8 -*-
r"""
Approach 1: Subword Regularization / BPE-Dropout for the SuPar constituency parser.

WHY THIS FILE EXISTS
--------------------
SuPar (`supar.utils.data.Dataset.build`) numericalizes the *whole* dataset once,
before the first epoch, and its `collate_fn` merely stacks pre-computed tensors::

    fields = self.transform(self.sentences)          # tokenize + numericalize ONCE
    self.loader = DataLoader(..., collate_fn=lambda x: Batch(x))

That means a stochastic tokenizer plugged into the field alone would be sampled
exactly once per sentence, for the entire run -- i.e. no regularization at all.
This module therefore replaces the *training* loader's collate function so that
every word is re-segmented each time the sentence is drawn into a batch.

Dev/test loaders keep SuPar's default collator and stay fully deterministic.

WHAT "BPE-DROPOUT" MEANS FOR XLM-R
----------------------------------
XLM-RoBERTa's tokenizer is SentencePiece **Unigram**, not BPE. Provost & Titov's
BPE-Dropout (2020) drops merge operations, which requires a BPE merge table that
XLM-R does not have. The correct analogue for a Unigram model is Kudo's (2018)
subword regularization: sample a segmentation from the unigram lattice instead of
taking the Viterbi-best one. SentencePiece implements it natively::

    sp.encode(w, out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)

Two backends are provided:

``spm``
    Kudo sampling from the unigram lattice (default, principled, fast).
``maxmatch``
    Vocabulary-level stochastic longest-match: at each position, with
    probability ``p`` skip the longest matching piece and take the next-longest.
    This is the closer analogue to BPE-Dropout's "skip a merge" behaviour, and,
    unlike ``spm``, it CAN emit tokens that were added to the HF vocabulary after
    the SentencePiece model was trained -- i.e. the 1,875 FOCUS-injected Yiddish
    tokens. Use this backend on ``phase1_focus_model`` / ``phase2_trained`` if you
    want the injected tokens to be reachable stochastically.

Both backends only ever emit strings, never ids. That matters: SuPar's
``SubwordField.transform`` looks pieces up in ``tokenizer.get_vocab()`` itself,
so we never have to reason about XLM-R's fairseq id offset.
"""

import os
import random
from typing import Dict, List, Optional, Sequence

_SPM_FILENAMES = ("sentencepiece.bpe.model", "spiece.model", "sentencepiece.model")

# SentencePiece's word-boundary marker used by XLM-R.
WORD_PREFIX = "▁"  # ▁


def _find_spm_file(model_name_or_path: str) -> Optional[str]:
    if os.path.isdir(model_name_or_path):
        for name in _SPM_FILENAMES:
            candidate = os.path.join(model_name_or_path, name)
            if os.path.exists(candidate):
                return candidate
    return None


class StochasticSubwordTokenizer:
    r"""A drop-in replacement for ``tokenizer.tokenize`` with sampled segmentation.

    The object is *deterministic by default*. Sampling only happens inside the
    :meth:`sampling` context manager, which the training collator opens. This
    keeps dataset construction, dev/test evaluation and prediction unchanged.

    Args:
        model_name_or_path (str):
            Same path handed to SuPar's ``bert`` argument, e.g.
            ``./output/phase2_trained/checkpoint-6000``.
        alpha (float):
            SentencePiece sampling smoothing. ``0.0`` samples uniformly from the
            lattice, large values approach the Viterbi-best segmentation.
            Kudo (2018) reports 0.1-0.2 as the useful band. Default: 0.1.
        p (float):
            For ``maxmatch``: probability of dropping the longest match at a
            given position. Provost & Titov use 0.1 for high-resource and
            recommend larger values for small corpora. Default: 0.1.
        word_p (float):
            Probability that a given *word* is re-segmented at all. ``1.0``
            perturbs every multi-piece word. Lower it to soften the noise.
        backend (str):
            ``'spm'``, ``'maxmatch'`` or ``'auto'`` (spm if a SentencePiece model
            file can be located, otherwise maxmatch).
        seed (int):
            Seed for this object's private RNG, so segmentation noise is
            reproducible and independent of the global torch/random streams.
        max_pieces (int):
            Reject a sampled segmentation longer than this and fall back to the
            deterministic one. Guards against SuPar's ``fix_len`` truncation
            silently chopping the tail off a word. Default: 32.
        anchor (str):
            ``maxmatch`` only -- what to return when no dropout event fires.

            ``'hf'`` (default)
                The HF/Viterbi segmentation, i.e. the exact segmentation used at
                evaluation time. Dropout then *perturbs around* the canonical
                form, which is BPE-Dropout's own semantics and leaves zero
                train/test segmentation skew.
            ``'greedy'``
                The greedy longest-match segmentation. This deliberately shifts
                the base distribution towards whole-word pieces, which is how you
                would make the FOCUS-injected "zombie tokens" reachable during
                parser training -- but note the training-time canonical form then
                differs from the evaluation-time one.
    """

    def __init__(self,
                 model_name_or_path: str,
                 alpha: float = 0.1,
                 p: float = 0.1,
                 word_p: float = 1.0,
                 backend: str = "auto",
                 seed: int = 1,
                 max_pieces: int = 32,
                 anchor: str = "hf"):
        self.model_name_or_path = model_name_or_path
        self.alpha = alpha
        self.p = p
        self.word_p = word_p
        self.backend = backend
        self.seed = seed
        self.max_pieces = max_pieces
        self.anchor = anchor

        # Runtime-only state; rebuilt lazily so the object stays picklable
        # (SuPar dill-pickles the whole Transform, including field.tokenize).
        self.sample = False
        self._hf = None
        self._sp = None
        self._vocab = None
        self._pieces_by_len = None
        self._max_piece_len = 0
        self._cache: Dict[str, List[str]] = {}
        self._rng = None
        self._resolved_backend = None

    # ------------------------------------------------------------------ #
    # pickling: SuPar saves `transform` with dill inside the .pt file.
    # Drop the tokenizer/sp handles and rebuild them on load.
    # ------------------------------------------------------------------ #
    def __getstate__(self):
        state = dict(self.__dict__)
        for key in ("_hf", "_sp", "_vocab", "_pieces_by_len", "_cache", "_rng"):
            state[key] = None
        state["sample"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = {}

    def __repr__(self):
        return (f"{self.__class__.__name__}(backend={self._resolved_backend or self.backend}, "
                f"alpha={self.alpha}, p={self.p}, word_p={self.word_p})")

    # ------------------------------------------------------------------ #
    # lazy init
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self):
        if self._hf is not None:
            return
        from transformers import AutoTokenizer

        self._rng = random.Random(self.seed)
        self._hf = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self._vocab = self._hf.get_vocab()

        backend = self.backend
        if backend in ("auto", "spm"):
            try:
                import sentencepiece as spm

                spm_file = _find_spm_file(self.model_name_or_path)
                if spm_file is None:
                    # Fall back to the slow tokenizer, which exposes .sp_model.
                    slow = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)
                    self._sp = getattr(slow, "sp_model", None)
                else:
                    self._sp = spm.SentencePieceProcessor(model_file=spm_file)
            except Exception as exc:  # noqa: BLE001 - diagnostics matter more than the type
                if backend == "spm":
                    raise RuntimeError(
                        f"backend='spm' requested but no SentencePiece model could be loaded "
                        f"from {self.model_name_or_path}: {exc}"
                    ) from exc
                self._sp = None

        if self._sp is not None and self.backend != "maxmatch":
            self._resolved_backend = "spm"
        else:
            if self.backend == "spm":
                # Do NOT fall through to maxmatch here. Reporting a "Kudo
                # sampling" cell that actually ran stochastic longest-match
                # would duplicate the maxmatch cell under a different name.
                raise RuntimeError(
                    "backend='spm' requested but no SentencePiece model is reachable "
                    f"for {self.model_name_or_path}. Recent transformers releases drop "
                    "the slow tokenizer that exposes .sp_model, and the checkpoint "
                    "ships no sentencepiece.bpe.model. Either place that file next to "
                    "the model and pass a local path, or run --backend maxmatch and say "
                    "in the write-up that Kudo lattice sampling could not be evaluated."
                )
            self._resolved_backend = "maxmatch"
            self._build_piece_index()

    def _build_piece_index(self):
        by_len: Dict[int, set] = {}
        for piece in self._vocab:
            by_len.setdefault(len(piece), set()).add(piece)
        self._pieces_by_len = by_len
        self._max_piece_len = max(by_len) if by_len else 0

    # ------------------------------------------------------------------ #
    # sampling switch
    # ------------------------------------------------------------------ #
    class _SamplingContext:
        def __init__(self, owner):
            self.owner = owner
            self.previous = False

        def __enter__(self):
            self.previous = self.owner.sample
            self.owner.sample = True
            return self.owner

        def __exit__(self, *exc):
            self.owner.sample = self.previous
            return False

    def sampling(self):
        """Context manager enabling stochastic segmentation."""
        return self._SamplingContext(self)

    # ------------------------------------------------------------------ #
    # the tokenize entry point (this is what SuPar's Field.preprocess calls)
    # ------------------------------------------------------------------ #
    def __call__(self, word: str) -> List[str]:
        self._ensure_loaded()
        if not self.sample:
            return self.deterministic(word)

        pieces = self.deterministic(word)
        # Single-piece words have nothing to resegment; this also protects
        # FOCUS-injected whole-word tokens from being broken up under 'spm'.
        if len(pieces) < 2 or self._rng.random() >= self.word_p:
            return pieces

        sampled = (self._sample_spm(word) if self._resolved_backend == "spm"
                   else self._sample_maxmatch(word))
        if not sampled or len(sampled) > self.max_pieces:
            return pieces
        # Never emit an OOV piece: SubwordField would silently map it to <unk>.
        if any(piece not in self._vocab for piece in sampled):
            return pieces
        return sampled

    # `tokenize` alias so the object can also be passed where a bound
    # tokenizer method is expected.
    def tokenize(self, word: str) -> List[str]:
        return self(word)

    def deterministic(self, word: str) -> List[str]:
        self._ensure_loaded()
        cached = self._cache.get(word)
        if cached is None:
            cached = self._hf.tokenize(word)
            self._cache[word] = cached
        return cached

    # ------------------------------------------------------------------ #
    # backend: Kudo (2018) unigram lattice sampling
    # ------------------------------------------------------------------ #
    def _sample_spm(self, word: str) -> List[str]:
        try:
            return self._sp.encode(word, out_type=str, enable_sampling=True,
                                   alpha=self.alpha, nbest_size=-1)
        except TypeError:
            # Older sentencepiece python bindings use SampleEncodeAsPieces.
            return self._sp.SampleEncodeAsPieces(word, -1, self.alpha)

    # ------------------------------------------------------------------ #
    # backend: stochastic longest-match over the HF vocabulary
    # ------------------------------------------------------------------ #
    def _sample_maxmatch(self, word: str) -> List[str]:
        text = WORD_PREFIX + word
        pieces, start, n = [], 0, len(text)
        dropped = False
        while start < n:
            candidates = []
            upper = min(n - start, self._max_piece_len)
            for length in range(upper, 0, -1):
                candidate = text[start:start + length]
                bucket = self._pieces_by_len.get(len(candidate))
                if bucket is not None and candidate in bucket:
                    candidates.append(candidate)
                    if len(candidates) == 2:
                        break
            if not candidates:
                return []  # unsegmentable -> caller falls back to deterministic
            chosen = candidates[0]
            if len(candidates) > 1 and self._rng.random() < self.p:
                chosen = candidates[1]  # "drop" the longest match
                dropped = True
            pieces.append(chosen)
            start += len(chosen)
        if not dropped and self.anchor == "hf":
            # No merge was dropped, so this word is "unperturbed": return the
            # canonical segmentation the model also sees at evaluation time,
            # not the greedy one.
            return []
        return pieces


# ====================================================================== #
# The training collator
# ====================================================================== #
class StochasticBatchCollator:
    r"""Replacement for SuPar's ``collate_fn=lambda x: Batch(x)``.

    Re-runs ``SubwordField.transform`` on the raw words of every sentence in the
    batch, with sampling enabled, then defers to SuPar's ``Batch`` to pad and
    move the tensors to device exactly as before.

    Only the subword field is recomputed. Charts, trees and tags are untouched,
    so the supervision signal is bit-identical to the baseline -- only the
    *segmentation* of the input varies.
    """

    def __init__(self, tokenizer: StochasticSubwordTokenizer, field_names: Sequence[str] = ("words", "bert")):
        self.tokenizer = tokenizer
        self.field_names = tuple(field_names)
        self._fields = None

    def _resolve_fields(self, sentence):
        from supar.utils.field import SubwordField

        fields = []
        for field in sentence.transform.flattened_fields:
            if not isinstance(field, SubwordField):
                continue
            if field.name not in self.field_names:
                continue
            if getattr(field, "tokenize", None) is None:
                continue  # e.g. the CharLSTM field, which splits on characters
            # Route the field through our stochastic tokenizer.
            if field.tokenize is not self.tokenizer:
                field.tokenize = self.tokenizer
            fields.append(field)
        if not fields:
            raise RuntimeError(
                "Subword regularization is enabled but no tokenizing SubwordField was found. "
                f"Looked for fields named {self.field_names}; the transform exposes "
                f"{[f.name for f in sentence.transform.flattened_fields]}."
            )
        return fields

    def __call__(self, sentences):
        from supar.utils.transform import Batch

        if self._fields is None:
            self._fields = self._resolve_fields(sentences[0])

        with self.tokenizer.sampling():
            for sentence in sentences:
                for field in self._fields:
                    words = getattr(sentence, field.name)
                    sentence.transformed[field.name] = field.transform([words])[0]
        return Batch(sentences)


# ====================================================================== #
# Installation
# ====================================================================== #
_ORIGINAL_BUILD = None


def install_subword_regularization(model_name_or_path: str,
                                   alpha: float = 0.1,
                                   p: float = 0.1,
                                   word_p: float = 1.0,
                                   backend: str = "auto",
                                   seed: int = 1,
                                   max_pieces: int = 32,
                                   anchor: str = "hf",
                                   verbose: bool = True) -> StochasticSubwordTokenizer:
    r"""Patch ``supar.utils.data.Dataset.build`` so the *training* loader resamples.

    The split is identified by ``shuffle=True``, which SuPar passes only for the
    training set (``supar/parsers/parser.py``)::

        train = Dataset(...).build(batch_size, buckets, True, dist.is_initialized())
        dev   = Dataset(...).build(batch_size, buckets)     # shuffle=False
        test  = Dataset(...).build(batch_size, buckets)     # shuffle=False

    Call this once, *before* ``CRFConstituencyParser.build`` / ``parser.train``.
    Returns the tokenizer object so the caller can inspect or report on it.
    """
    global _ORIGINAL_BUILD

    from torch.utils.data import DataLoader
    from supar.utils.data import Dataset

    tokenizer = StochasticSubwordTokenizer(model_name_or_path, alpha=alpha, p=p, word_p=word_p,
                                           backend=backend, seed=seed, max_pieces=max_pieces,
                                           anchor=anchor)

    if _ORIGINAL_BUILD is None:
        _ORIGINAL_BUILD = Dataset.build

    original_build = _ORIGINAL_BUILD

    def build_with_regularization(self, batch_size, n_buckets=1, shuffle=False, distributed=False):
        # Deterministic pass first: builds the vocab-side tensors and the
        # length-based buckets from the canonical segmentation.
        dataset = original_build(self, batch_size, n_buckets, shuffle, distributed)
        if shuffle:  # training split only
            self.loader = DataLoader(dataset=self,
                                     batch_sampler=self.loader.batch_sampler,
                                     collate_fn=StochasticBatchCollator(tokenizer))
            if verbose:
                print(f"[subword-reg] training loader now resamples segmentation per batch "
                      f"({tokenizer})")
        return dataset

    Dataset.build = build_with_regularization
    if verbose:
        tokenizer._ensure_loaded()
        print(f"[subword-reg] installed: backend={tokenizer._resolved_backend}, "
              f"alpha={alpha}, p={p}, word_p={word_p}, anchor={anchor}, max_pieces={max_pieces}")
    return tokenizer


def uninstall_subword_regularization():
    """Restore SuPar's original ``Dataset.build`` (useful in notebooks/tests)."""
    if _ORIGINAL_BUILD is not None:
        from supar.utils.data import Dataset
        Dataset.build = _ORIGINAL_BUILD


# ====================================================================== #
# Diagnostics for the write-up
# ====================================================================== #
def report_fragmentation(tokenizer: StochasticSubwordTokenizer,
                         words: Sequence[str],
                         n_samples: int = 5,
                         n_examples: int = 8) -> dict:
    r"""Fertility statistics + example segmentations.

    ``fertility`` = mean subword pieces per word, the standard measure of the
    "tokenization tax" this project is about. Report the deterministic value and
    the sampled value; the sampled one should be modestly higher, and the *set*
    of distinct segmentations per word is the actual regularization signal.
    """
    tokenizer._ensure_loaded()
    words = [w for w in words if w]
    det = [tokenizer.deterministic(w) for w in words]
    det_fertility = sum(len(p) for p in det) / max(len(det), 1)

    sampled_lengths, distinct = [], []
    with tokenizer.sampling():
        for word in words:
            variants = {tuple(tokenizer(word)) for _ in range(n_samples)}
            distinct.append(len(variants))
            sampled_lengths.extend(len(v) for v in variants)

    examples = []
    with tokenizer.sampling():
        for word in words[:n_examples]:
            variants = sorted({tuple(tokenizer(word)) for _ in range(n_samples)})
            examples.append((word, tokenizer.deterministic(word), [list(v) for v in variants]))

    stats = {
        "n_words": len(words),
        "backend": tokenizer._resolved_backend,
        "deterministic_fertility": round(det_fertility, 3),
        "sampled_fertility": round(sum(sampled_lengths) / max(len(sampled_lengths), 1), 3),
        "mean_distinct_segmentations": round(sum(distinct) / max(len(distinct), 1), 3),
        "pct_words_with_variation": round(100.0 * sum(d > 1 for d in distinct) / max(len(distinct), 1), 2),
        "max_pieces_observed": max(sampled_lengths) if sampled_lengths else 0,
        "examples": examples,
    }
    return stats


def words_from_supar_file(path: str, limit: int = 5000) -> List[str]:
    """Pull terminals out of a SuPar-format (bracketed tree) file, for reporting."""
    import re

    words: List[str] = []
    token_re = re.compile(r"\(([^\s()]+)\s+([^\s()]+)\)")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            for _tag, word in token_re.findall(line):
                words.append(word)
                if len(words) >= limit:
                    return words
    return words
