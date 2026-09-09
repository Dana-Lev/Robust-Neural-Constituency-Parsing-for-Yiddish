# Data

No corpus data is redistributed in this repository. This file documents how to
obtain the treebank and rebuild the exact split files the training scripts read.

## Expected layout after rebuilding

```
yiddish_parser/data/
├── raw/
│   ├── ppchyprep/                  # cloned tool + its output
│   └── ppchy/corpus_data/          # PPCHY source trees
└── processed/
    ├── ppchy_final_trees.txt       # Hebrew-script trees, one per line
    ├── supar_train_ready.txt       # after finalize_ppchy_for_supar.py
    └── supar_ready/
        ├── train.txt               # 15,394 trees
        ├── dev.txt                 #    855 trees
        └── test.txt                #    856 trees
```

## PPCHY — the treebank

The Penn Parsed Corpus of Historical Yiddish provides the gold constituency
trees. This project uses **all** PPCHY components, not only the two largest
20th-century texts: the older material carries the heaviest orthographic
variation, and therefore the heaviest subword fragmentation, which is the
phenomenon under test.

Romanized trees are converted to Hebrew script with Kulick's `ppchyprep`:

```bash
mkdir -p yiddish_parser/data/raw && cd yiddish_parser/data/raw
git clone https://github.com/skulick/ppchyprep
```

`ppchyprep` requires two non-PyPI packages (`yiddishycode`, `ppctree`) and the
corpus at a pinned commit; its `run.sh` also ships without an execute bit.

Then, from `yiddish_parser/`:

```bash
python src/ppchy_formatting/build_final_trees.py        # JSON -> Hebrew-script trees
python src/ppchy_formatting/finalize_ppchy_for_supar.py # (TOP ...) wrapping, cleanup
python src/split_supar_data.py                          # 90/5/5 split, seed 42
python src/clean_tree_data.py                           # drop trees SuPar would reject
```

`scripts/build_ppchy_data.sh` runs those four steps in order.
`scripts/dataset_stats.py` reproduces the statistics reported in
`results/dataset_stats.md`.

## Encoder checkpoint

The Yiddish-pretrained XLM-R checkpoint is downloaded from the Hugging Face Hub
automatically: [`skulick/xlmb-ybc-ck05`](https://huggingface.co/skulick/xlmb-ybc-ck05).
On a shared cluster, set `HF_HOME` to a storage directory outside your home
quota.

## Attribution

`ppchy_formatting/`, `split_supar_data.py` and `clean_tree_data.py` are reused,
with thanks, from the course project of Amit Halfon and Omri Boiman,
[Neural Constituency Parsing for Yiddish via Vocabulary Adaptation](https://github.com/omri-boiman/Neural-Constituency-Parsing-for-Yiddish-via-Vocabulary-Adaptation).
No result from that project is reported in this one.
