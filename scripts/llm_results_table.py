# -*- coding: utf-8 -*-
"""
Assemble every LLM baseline run into one comparison table.

Reads the result JSONs written by `testing.py` and emits a Markdown table for
`results/`, plus LaTeX rows to paste into the report -- so no number is
transcribed by hand between the two.

    python scripts/llm_results_table.py results/llm_*.json
    python scripts/llm_results_table.py results/llm_*.json --latex
"""

import argparse
import glob
import json
import os


def micro_lf(records):
    """Corpus-level labeled F1 over a subset of records."""
    m = p = g = 0
    for r in records:
        c = r.get("counts")
        if c:
            m += c["labeled_match"]; p += c["pred_count"]; g += c["gold_count"]
    prec = m / p if p else 0.0
    rec = m / g if g else 0.0
    return round(100 * (2 * prec * rec / (prec + rec) if (prec + rec) else 0.0), 2)


def load(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    s = data.get("summary", {})
    records = data.get("results", [])
    models = sorted({r.get("model") for r in records if r.get("model")})
    got = [r for r in records if not r.get("error")]
    faithful = [r for r in got if r.get("tokens_match")]
    return {
        # LF restricted to answers that reproduced the given tokens. Without
        # this, LF conflates "cannot parse" with "did not copy the tokens":
        # a tree over different terminals cannot align to the gold spans at all.
        "lf_faithful": micro_lf(faithful),
        "n_faithful": len(faithful),
        "n_got": len(got),
        "file": os.path.basename(path),
        "model": s.get("model", models[0] if models else "?"),
        "mixed_models": len(models) > 1,
        "shots": s.get("shots", 0),
        "n": s.get("n_sentences_attempted", s.get("n_sentences", len(records))),
        "got": s.get("n_responses_received", "-"),
        "coverage": s.get("coverage_rate", "-"),
        "valid": s.get("valid_syntax_rate", "-"),
        "tok": s.get("token_fidelity_rate", "-"),
        "lf": s.get("labeled_f1", "-"),
        "uf": s.get("unlabeled_f1", "-"),
        "lp": s.get("labeled_precision", "-"),
        "lr": s.get("labeled_recall", "-"),
        "macro": s.get("macro_labeled_f1", "-"),
    }


def condition(row):
    return f"{row['model']} {row['shots']}-shot" if row["shots"] else f"{row['model']} zero-shot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Result JSONs (globs are fine).")
    ap.add_argument("--latex", action="store_true", help="Also emit LaTeX table rows.")
    args = ap.parse_args()

    paths = sorted({p for pattern in args.paths for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit("No result files matched.")

    rows = [load(p) for p in paths]
    rows.sort(key=lambda r: (r["model"], r["shots"]))

    print("| Condition | n | Returned | Coverage | Valid | Tok | LF | UF |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {condition(r)} | {r['n']} | {r['got']} | {r['coverage']} | "
              f"{r['valid']} | {r['tok']} | **{r['lf']}** | {r['uf']} |")

    print("\nLabeled F1 restricted to answers that reproduced the given tokens "
          "(separating parsing ability from instruction-following):\n")
    print("| Condition | Token-faithful | LF (all) | LF (faithful only) |")
    print("|---|---:|---:|---:|")
    for r in rows:
        print(f"| {condition(r)} | {r['n_faithful']}/{r['n_got']} | {r['lf']} | "
              f"**{r['lf_faithful']}** |")

    print("\nLabeled P/R and macro LF:\n")
    print("| Condition | LP | LR | Macro LF |")
    print("|---|---:|---:|---:|")
    for r in rows:
        print(f"| {condition(r)} | {r['lp']} | {r['lr']} | {r['macro']} |")

    warn = [r for r in rows if r["mixed_models"]]
    if warn:
        print("\n> WARNING: these files mix answers from more than one model, so "
              "their scores are not interpretable as a single condition: "
              + ", ".join(r["file"] for r in warn))

    low = [r for r in rows if isinstance(r["coverage"], (int, float)) and r["coverage"] < 100]
    if low:
        print("\n> Coverage below 100% (API quota, not model failure) in: "
              + ", ".join(f"{r['file']} ({r['coverage']}%)" for r in low)
              + ". State this in the report.")

    if args.latex:
        print("\n% ---- paste into Table 2 of report/main.tex ----")
        for r in rows:
            model = r["model"].replace("_", r"\_")
            shots = "zero-shot" if not r["shots"] else f"{r['shots']}-shot"
            print(f"\\texttt{{{model}}} {shots} & {r['valid']} & {r['tok']} "
                  f"& {r['lf']} & {r['uf']} \\\\")


if __name__ == "__main__":
    main()
