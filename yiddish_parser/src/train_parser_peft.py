# -*- coding: utf-8 -*-
r"""
Drop-in replacement for ``src/train_parser.py`` adding two parameter-efficient
routes to the subword-fragmentation problem, selectable per run:

    python src/train_parser_peft.py --mode baseline
    python src/train_parser_peft.py --mode bpe-dropout   --backend maxmatch --p 0.1
    python src/train_parser_peft.py --mode adapters      --adapter-type lora
    python src/train_parser_peft.py --mode both          --adapter-type bottleneck

Neither route touches the physical vocabulary and neither requires continued
pre-training. ``--mode baseline`` reproduces the original script's behaviour
(full backbone frozen, parser head only), so all four cells of the ablation
come from one entrypoint with one seed handling path.

Run from the ``yiddish_parser/`` directory, exactly like the original script.
"""

import argparse
import os
import random
import sys

import torch
import transformers

# =========================================================
# PATCH: Inject AdamW  (unchanged from the original script -- recent
# transformers releases removed transformers.AdamW, which supar imports
# inside Parser.train)
# =========================================================
transformers.AdamW = torch.optim.AdamW
try:
    from transformers import optimization

    if not hasattr(optimization, "AdamW"):
        optimization.AdamW = torch.optim.AdamW
except ImportError:
    pass
# =========================================================

import supar  # noqa: E402

try:
    from supar.utils import Config
except ImportError:
    from supar.config import Config

from supar import CRFConstituencyParser  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import language_adapters as la  # noqa: E402
import subword_regularization as sr  # noqa: E402

# ================= DEFAULTS (match the original train_parser.py) =================
DEFAULT_DATA_DIR = "data/processed/supar_ready"
DEFAULT_ENCODER = "./output/phase2_trained/checkpoint-6000"
DEFAULT_OUTPUT = "./output/phase3_parser_model"
# ================================================================================


def parse_args():
    p = argparse.ArgumentParser(description="PEFT constituency parser for Yiddish (PPCHY)")

    p.add_argument("--mode", choices=["baseline", "bpe-dropout", "adapters", "both"],
                   default="both",
                   help="Which parameter-efficient approach(es) to enable.")

    # --- paths ---
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--encoder-path", default=DEFAULT_ENCODER,
                   help="HF model dir. Use the raw Yiddish XLM-R (skulick/xlmb-ybc-ck05) "
                        "to isolate the PEFT contribution from the FOCUS+DAPT pipeline.")
    p.add_argument("--output-dir", default=None,
                   help="Defaults to ./output/parser_<mode>[_<adapter-type>].")

    # --- Approach 1: subword regularization ---
    p.add_argument("--backend", choices=["auto", "spm", "maxmatch"], default="auto",
                   help="spm = Kudo unigram-lattice sampling; maxmatch = stochastic "
                        "longest-match, the only backend that can emit FOCUS-injected tokens.")
    p.add_argument("--alpha", type=float, default=0.1, help="SentencePiece sampling smoothing.")
    p.add_argument("--p", type=float, default=0.1, help="maxmatch: per-position dropout probability.")
    p.add_argument("--word-p", type=float, default=1.0,
                   help="Probability that a multi-piece word is resegmented at all.")
    p.add_argument("--anchor", choices=["hf", "greedy"], default="hf",
                   help="maxmatch only. 'hf': no-dropout words keep the evaluation-time "
                        "segmentation (no train/test skew). 'greedy': shifts the base towards "
                        "whole-word pieces, making FOCUS-injected tokens reachable.")
    p.add_argument("--fix-len", type=int, default=32,
                   help="Max subword pieces per word. The original uses SuPar's default of 20; "
                        "sampling makes words longer, and SubwordField TRUNCATES silently.")
    p.add_argument("--selftest", action="store_true",
                   help="Print fertility stats and example segmentations, then exit.")

    # --- Approach 2: language adapters ---
    p.add_argument("--adapter-type", choices=["lora", "bottleneck", "both"], default="lora")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-targets", nargs="+", default=["query", "value"])
    p.add_argument("--reduction-factor", type=int, default=16,
                   help="Bottleneck adapters: hidden_size // reduction_factor.")
    p.add_argument("--bottleneck-dropout", type=float, default=0.0)
    p.add_argument("--adapter-layers", nargs="+", type=int, default=None,
                   help="Restrict bottleneck adapters to these layer indices (default: all).")
    p.add_argument("--adapter-lr", type=float, default=3e-4,
                   help="Adapters need a higher LR than SuPar's encoder rate (5e-5). Sweep 1e-4/3e-4/1e-3.")
    p.add_argument("--grad-checkpointing", action="store_true",
                   help="Enable if 12GB is tight; ~30%% slower steps, much less activation memory.")

    # --- training ---
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2000, help="Token-level batch size.")
    p.add_argument("--update-steps", type=int, default=1, help="Gradient accumulation steps.")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-rate", type=float, default=20, help="Parser-head LR multiplier (SuPar).")
    p.add_argument("--n-bert-layers", type=int, default=4,
                   help="Top encoder layers fed to SuPar's ScalarMix (which is ALWAYS applied "
                        "in supar 1.1.4, whatever bert_pooling says). The repo script uses 4; "
                        "use 12 to match the previous project's reported configuration.")
    p.add_argument("--bert-pooling", choices=["mean", "first", "last"], default="mean",
                   help="How subword pieces are pooled into one word vector (supar 1.1.4 has "
                        "no 'scalar_mix' pooling: layer mixing is separate and always on). "
                        "'mean' matches the repo's train_parser.py.")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_output_dir(args) -> str:
    if args.output_dir:
        return args.output_dir
    suffix = args.mode
    if args.mode in ("adapters", "both"):
        suffix = f"{suffix}_{args.adapter_type}"
    return f"./output/parser_{suffix}"


def run_selftest(args):
    """Fertility / segmentation-variation report -- useful table for the write-up."""
    tokenizer = sr.StochasticSubwordTokenizer(
        args.encoder_path, alpha=args.alpha, p=args.p, word_p=args.word_p,
        backend=args.backend, seed=args.seed, max_pieces=args.fix_len, anchor=args.anchor)
    train_file = os.path.join(args.data_dir, "train.txt")
    words = sr.words_from_supar_file(train_file, limit=5000)
    if not words:
        raise SystemExit(f"No terminals parsed out of {train_file}")

    stats = sr.report_fragmentation(tokenizer, words, n_samples=5)
    print("\n" + "=" * 66)
    print("SUBWORD REGULARIZATION SELF-TEST")
    print("=" * 66)
    for key in ("n_words", "backend", "deterministic_fertility", "sampled_fertility",
                "mean_distinct_segmentations", "pct_words_with_variation", "max_pieces_observed"):
        print(f"  {key:<32} {stats[key]}")
    print("-" * 66)
    for word, deterministic, variants in stats["examples"]:
        print(f"  {word}")
        print(f"      deterministic: {deterministic}")
        for variant in variants:
            print(f"      sampled:       {variant}")
    print("=" * 66)
    if stats["max_pieces_observed"] > args.fix_len:
        print(f"\nWARNING: sampled segmentations reach {stats['max_pieces_observed']} pieces "
              f"but fix_len={args.fix_len}; raise --fix-len to avoid silent truncation.\n")


def train(args):
    output_dir = resolve_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    model_file = os.path.join(output_dir, "yiddish_parser.pt")

    use_subword_reg = args.mode in ("bpe-dropout", "both")
    use_adapters = args.mode in ("adapters", "both")

    print("=" * 66)
    print(f"MODE: {args.mode}")
    print(f"  subword regularization : {use_subword_reg}")
    print(f"  language adapters      : {use_adapters}"
          f"{' (' + args.adapter_type + ')' if use_adapters else ''}")
    print(f"  encoder                : {args.encoder_path}")
    print(f"  output                 : {model_file}")
    print("=" * 66)

    set_seed(args.seed)

    # --- STEP 1: Approach 1. Patch Dataset.build BEFORE any Dataset is created,
    #     so the training loader gets the stochastic collator.
    tokenizer = None
    if use_subword_reg:
        tokenizer = sr.install_subword_regularization(
            args.encoder_path, alpha=args.alpha, p=args.p, word_p=args.word_p,
            backend=args.backend, seed=args.seed, max_pieces=args.fix_len, anchor=args.anchor)

    # --- STEP 2: Approach 2. Patch CRFConstituencyParser.MODEL BEFORE build(),
    #     because build() instantiates cls.MODEL(**args).
    if use_adapters:
        la.patch_parser_with_adapters(
            CRFConstituencyParser,
            adapter_type=args.adapter_type,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            lora_targets=args.lora_targets,
            reduction_factor=args.reduction_factor,
            bottleneck_dropout=args.bottleneck_dropout,
            bottleneck_layers=args.adapter_layers,
            gradient_checkpointing=args.grad_checkpointing)
    else:
        # Baseline behaviour: freeze the whole backbone, no adapters. This
        # replaces the original apply_freeze_patch(), whose first check
        # ("'Embeddings' in name and 'Bert' in name") never matches
        # XLMRobertaEmbeddings -- only its generic-scan fallback fired.
        apply_full_freeze_patch()

    config = Config(**build_supar_args(args, model_file))

    print("   -> Building Parser instance...")
    parser = CRFConstituencyParser.build(**config)

    # --- STEP 3: adapters need their own LR. Must happen after the model
    #     exists (the factory maps parameter identity -> name) and before
    #     train() imports transformers.AdamW.
    if use_adapters:
        la.install_adapter_optimizer(parser.model, adapter_lr=args.adapter_lr)
    else:
        la.install_adapter_optimizer(parser.model, adapter_lr=args.lr, drop_frozen=True)

    if tokenizer is not None:
        words = sr.words_from_supar_file(os.path.join(args.data_dir, "train.txt"), limit=2000)
        stats = sr.report_fragmentation(tokenizer, words, n_samples=5, n_examples=3)
        print(f"[subword-reg] fertility {stats['deterministic_fertility']} (deterministic) -> "
              f"{stats['sampled_fertility']} (sampled); "
              f"{stats['pct_words_with_variation']}% of words vary; "
              f"max {stats['max_pieces_observed']} pieces (fix_len={args.fix_len})")

    print("   -> Starting Training Loop...")
    parser.train(**config)
    print("Training Complete.")
    print(f"Best model: {model_file}")


def build_supar_args(args, model_file):
    """The original script's config dict, with only the noted values parameterized."""
    return {
        # 1. Paths
        "train": os.path.join(args.data_dir, "train.txt"),
        "dev": os.path.join(args.data_dir, "dev.txt"),
        "test": os.path.join(args.data_dir, "test.txt"),
        "path": model_file,
        "mode": "train",
        "build": True,
        "checkpoint": False,

        # 2. Encoder
        "encoder": "bert",
        "bert": args.encoder_path,
        "finetune": False,
        "n_bert_layers": args.n_bert_layers,
        "bert_pooling": args.bert_pooling,
        "mix_dropout": 0.0,
        # CHANGED: sampled segmentations are longer than the Viterbi-best one,
        # and SubwordField.transform truncates to fix_len without warning.
        "fix_len": args.fix_len,

        # 3. Model Architecture
        "feat": ["char"],
        "n_embed": 100,
        "n_char_embed": 50,
        "n_char_hidden": 100,
        "n_feat_embed": 100,
        "n_encoder_hidden": 800,
        "n_encoder_layers": 3,
        "encoder_dropout": 0.33,
        "n_span_mlp": 500,
        "n_label_mlp": 100,
        "mlp_dropout": 0.33,
        "embed_dropout": 0.33,

        # 4. Training
        "lr": args.lr,
        "lr_rate": args.lr_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "update_steps": args.update_steps,
        "warmup": 0.1,
        "clip": 5.0,
        "decay": 0.75,
        "decay_steps": 5000,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "eps": 1e-8,
        "mu": 0.9,
        "nu": 0.9,

        # 5. Eval
        "structure": "joint",
        "mbr": True,
        "delete": {"TOP", "S1", "-NONE-", ",", ":", "``", "''", ".", "?", "!", ""},
        "equal": {"ADVP": "PRT"},

        # 6. System
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": args.seed,
        "amp": False,
        "cache": False,
        "verbose": True,
        "punct": False,
        "buckets": 32,
        "workers": 0,
    }


def apply_full_freeze_patch():
    """Baseline: freeze the entire backbone including the 192M embedding matrix."""
    OriginalModel = CRFConstituencyParser.MODEL

    class FrozenModel(OriginalModel):
        def __init__(self, *model_args, **model_kwargs):
            super().__init__(*model_args, **model_kwargs)
            bert = getattr(getattr(self, "encoder", None), "bert", None)
            if bert is None:
                raise RuntimeError("Expected parser.model.encoder.bert (encoder='bert').")
            la.freeze_backbone(bert)
            self.encoder.requires_grad = False
            la.describe_trainable(self)

    CRFConstituencyParser.MODEL = FrozenModel
    print("[baseline] full-backbone freeze patch applied.")


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.selftest:
        run_selftest(parsed)
    else:
        train(parsed)
