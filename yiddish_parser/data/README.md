# Data

No corpus data is redistributed in this repository. This file documents how to
obtain each corpus and rebuild the exact files the training scripts expect.

## Expected layout after rebuilding

```
yiddish_parser/data/
├── raw/
│   ├── ppchyprep/                  # cloned tool + its output
│   ├── ppchy/corpus_data/          # PPCHY source trees
│   └── jochre/                     # Jochre OCR text files
└── processed/
    ├── ppchy_final_trees.txt       # Hebrew-script trees, one per line
    ├── supar_train_ready.txt       # after finalize_ppchy_for_supar.py
    └── supar_ready/
        ├── train.txt
        ├── dev.txt
        └── test.txt
```

## PPCHY — the treebank (required)

The Penn Parsed Corpus of Historical Yiddish provides the gold constituency
trees. Following Kulick et al. (2022), this project uses the two largest
20th-century components (Hirshbein 1977 and Olsvanger 1947), whose romanization
aligns closely with Standard Yiddish Orthography.

Romanized trees are converted to Hebrew script with Kulick's `ppchyprep`:

```bash
cd yiddish_parser/data/raw
git clone https://github.com/skulick/ppchyprep
# follow ppchyprep's own instructions to produce out/data/json
```

Then, from `yiddish_parser/`:

```bash
python "src/ppchy_formatting/build_final_trees.py"          # JSON -> Hebrew-script trees
python "src/ppchy_formatting/finalize_ppchy_for_supar.py"   # (TOP ...) wrapping, cleanup
python src/split_supar_data.py                             # 90/5/5 split, seed 42
python src/clean_tree_data.py                              # drop trees SuPar would reject
```

Record the resulting sentence and token counts — the report's dataset table must
use your own numbers, not any inherited from prior work.

## Jochre — clean vocabulary source (optional)

Needed only to reproduce the *static* FOCUS vocabulary-injection branch
(`inject_vocab.py`), which this project uses as a reference point rather than a
method. Available from the [Jochre Yiddish Corpus](https://gitlab.com/jochre/corpora/jochre-yiddish-corpus)
(Urieli, 2025); place the extracted text files under `raw/jochre/` and run
`src/data_extraction/build_vocab_jochre.py`.

## YBC — pre-training text (optional)

Needed only to reproduce the DAPT branch (`run_mlm.py`). Derived from the
[Yiddish Book Center](https://www.yiddishbookcenter.org/) digitized library; see
`src/data_extraction/harvest_ybc.py` and `combine_ybc.py`.

## Encoder checkpoint

The Yiddish-pretrained XLM-R checkpoint is downloaded from the Hugging Face Hub
automatically: [`skulick/xlmb-ybc-ck05`](https://huggingface.co/skulick/xlmb-ybc-ck05).
On the cluster, set `HF_HOME` to your course storage directory so the cache does
not land in your home quota.
