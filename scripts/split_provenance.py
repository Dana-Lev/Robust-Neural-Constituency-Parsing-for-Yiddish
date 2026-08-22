# -*- coding: utf-8 -*-
"""
Recover which PPCHY component each tree in each split came from.

`split_supar_data.py` shuffles a flat file, so provenance is lost by the time
the splits exist. `build_final_trees.py` writes a `*.sources.tsv` manifest;
this maps the splits back onto it.

Two uses:

1. Report the composition of each split (does the test set actually contain the
   older, orthographically messy material you claim to be testing on?).
2. Write a label file aligned line-for-line with test.txt, so test F1 can be
   broken down per component -- which answers "does subword regularization help
   more where spelling is noisier?" with no extra training.

    python scripts/split_provenance.py \
        --manifest yiddish_parser/data/processed/ppchy_final_trees.sources.tsv \
        --data-dir yiddish_parser/data/processed/supar_ready \
        --write-labels
"""

import argparse
import os
from collections import Counter, defaultdict


def load_manifest(path):
    tree_to_source = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            source, tree = line.split("\t", 1)
            # First writer wins; identical trees from different files are rare
            # and it only matters for the composition percentages.
            tree_to_source.setdefault(tree.strip(), source)
    return tree_to_source


def component(filename):
    """Collapse a JSON filename to a readable component name."""
    return os.path.splitext(filename)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="yiddish_parser/data/processed/ppchy_final_trees.sources.tsv")
    ap.add_argument("--data-dir", default="yiddish_parser/data/processed/supar_ready")
    ap.add_argument("--splits", nargs="+", default=["train.txt", "dev.txt", "test.txt"])
    ap.add_argument("--write-labels", action="store_true",
                    help="Write <split>.sources.txt, one component name per line, "
                         "aligned with the split file.")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        raise SystemExit(
            f"Manifest not found: {args.manifest}\n"
            "Rebuild with the current build_final_trees.py -- older runs did not "
            "write one."
        )

    tree_to_source = load_manifest(args.manifest)
    print(f"Manifest: {len(tree_to_source):,} distinct trees\n")

    grand = defaultdict(Counter)
    for split in args.splits:
        path = os.path.join(args.data_dir, split)
        if not os.path.exists(path):
            print(f"  (skipping missing {split})")
            continue

        labels, counts, unmatched = [], Counter(), 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                tree = line.strip()
                if not tree:
                    continue
                # clean_tree_data.py / finalize may have wrapped in (TOP ...)
                source = (tree_to_source.get(tree)
                          or tree_to_source.get(tree[5:-1].strip() if tree.startswith("(TOP ") else tree))
                if source is None:
                    unmatched += 1
                    labels.append("UNKNOWN")
                    counts["UNKNOWN"] += 1
                else:
                    name = component(source)
                    labels.append(name)
                    counts[name] += 1
                    grand[split][name] += 1

        total = sum(counts.values())
        print(f"{split}  ({total:,} trees)")
        for name, n in counts.most_common():
            print(f"    {name:<40}{n:>7,}  {100.0 * n / max(total, 1):5.1f}%")
        if unmatched:
            print(f"    note: {unmatched:,} trees could not be matched to a source "
                  f"(normalization differences); they are labelled UNKNOWN")
        print()

        if args.write_labels:
            out = os.path.join(args.data_dir, split.replace(".txt", ".sources.txt"))
            with open(out, "w", encoding="utf-8") as fout:
                fout.write("\n".join(labels) + "\n")
            print(f"    -> wrote {out}\n")


if __name__ == "__main__":
    main()
