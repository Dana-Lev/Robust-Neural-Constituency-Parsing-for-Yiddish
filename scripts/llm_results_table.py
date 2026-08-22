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


def load(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    s = data.get("summary", {})
    records = data.get("results", [])
    models = sorted({r.get("model") for r in records if r.get("model")})
    return {
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
