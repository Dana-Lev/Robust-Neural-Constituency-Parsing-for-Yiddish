# -*- coding: utf-8 -*-
"""
Dataset statistics for the SuPar-ready PPCHY splits.

Produces exactly the numbers the report's data table needs: sentence and token
counts per split, length distribution, subword fertility under the encoder's
tokenizer, and the most frequent constituent labels.

    python scripts/dataset_stats.py --data-dir yiddish_parser/data/processed/supar_ready
    python scripts/dataset_stats.py --data-dir ... --encoder skulick/xlmb-ybc-ck05
    python scripts/dataset_stats.py --data-dir ... --markdown > results/dataset_stats.md
"""

import argparse
import os
import re
from collections import Counter

from nltk.tree import Tree


def strip_coindex(label):
    """NP-SBJ-1 -> NP-SBJ. The scorer ignores co-indexation, so counting labels
    with it attached inflates the inventory (402 raw vs 226 normalised here)."""
    return re.sub(r"[-=]\d+$", "", label)


def read_trees(path):
    trees, malformed = [], 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                trees.append(Tree.fromstring(line))
            except Exception:
                malformed += 1
    return trees, malformed


def label_counts(tree, counter, raw_counter):
    for subtree in tree.subtrees():
        is_preterminal = len(subtree) == 1 and not isinstance(subtree[0], Tree)
        if not is_preterminal:
            raw_counter[subtree.label()] += 1
            counter[strip_coindex(subtree.label())] += 1


def split_stats(path):
    trees, malformed = read_trees(path)
    lengths = [len(t.leaves()) for t in trees]
    labels, raw_labels = Counter(), Counter()
    for tree in trees:
        label_counts(tree, labels, raw_labels)
    depths = [t.height() for t in trees]
    return {
        "file": os.path.basename(path),
        "sentences": len(trees),
        "tokens": sum(lengths),
        "mean_len": sum(lengths) / len(lengths) if lengths else 0.0,
        "max_len": max(lengths) if lengths else 0,
        "mean_depth": sum(depths) / len(depths) if depths else 0.0,
        "malformed": malformed,
        "labels": labels,
        "raw_labels": raw_labels,
        "trees": trees,
    }


def fertility(trees, encoder):
    """Mean subword pieces per word -- the 'tokenization tax' figure."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(encoder)
    words, pieces, unsplit = 0, 0, 0
    for tree in trees:
        for word in tree.leaves():
            n = len(tokenizer.tokenize(word)) or 1
            words += 1
            pieces += n
            unsplit += (n == 1)
    return {
        "words": words,
        "pieces": pieces,
        "fertility": pieces / words if words else 0.0,
        "pct_single_piece": 100.0 * unsplit / words if words else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="yiddish_parser/data/processed/supar_ready")
    ap.add_argument("--splits", nargs="+", default=["train.txt", "dev.txt", "test.txt"])
    ap.add_argument("--encoder", default=None,
                    help="HF model id/path; if given, also report subword fertility.")
    ap.add_argument("--top-labels", type=int, default=15)
    ap.add_argument("--markdown", action="store_true", help="Emit Markdown tables.")
    args = ap.parse_args()

    stats = []
    for name in args.splits:
        path = os.path.join(args.data_dir, name)
        if not os.path.exists(path):
            raise SystemExit(f"Missing split file: {path}")
        stats.append(split_stats(path))

    rows = [(s["file"], s["sentences"], s["tokens"], s["mean_len"], s["max_len"],
             s["mean_depth"], s["malformed"]) for s in stats]
    total_sent = sum(s["sentences"] for s in stats)
    total_tok = sum(s["tokens"] for s in stats)

    if args.markdown:
        print("| Split | Sentences | Tokens | Mean len | Max len | Mean depth | Malformed |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            print(f"| {r[0]} | {r[1]:,} | {r[2]:,} | {r[3]:.1f} | {r[4]} | {r[5]:.1f} | {r[6]} |")
        print(f"| **total** | **{total_sent:,}** | **{total_tok:,}** | | | | |")
    else:
        print(f"{'split':<12}{'sents':>9}{'tokens':>10}{'mean_len':>10}"
              f"{'max_len':>9}{'depth':>8}{'bad':>6}")
        print("-" * 64)
        for r in rows:
            print(f"{r[0]:<12}{r[1]:>9,}{r[2]:>10,}{r[3]:>10.1f}{r[4]:>9}{r[5]:>8.1f}{r[6]:>6}")
        print("-" * 64)
        print(f"{'total':<12}{total_sent:>9,}{total_tok:>10,}")

    all_labels, all_raw = Counter(), Counter()
    for s in stats:
        all_labels.update(s["labels"])
        all_raw.update(s["raw_labels"])
    print(f"\nDistinct constituent labels: {len(all_labels)} after stripping "
          f"co-indexation ({len(all_raw)} raw). The scorer ignores co-indexation, "
          f"so the first number is the one to quote.")
    print(f"Top {args.top_labels} labels:")
    for label, count in all_labels.most_common(args.top_labels):
        print(f"  {label:<16}{count:>8,}")

    if args.encoder:
        print(f"\nSubword fertility under {args.encoder}:")
        for s in stats:
            f = fertility(s["trees"], args.encoder)
            print(f"  {s['file']:<12} fertility={f['fertility']:.3f}  "
                  f"single-piece words={f['pct_single_piece']:.1f}%  "
                  f"({f['words']:,} words -> {f['pieces']:,} pieces)")


if __name__ == "__main__":
    main()
