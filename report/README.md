# Final report (ACL format)

- `main.tex` — the report skeleton. Every red `[TODO: ...]` is something you must
  write or a number you must fill in. Delete the `\todo` usages before submitting.
- `references.bib` — all citations already used by the skeleton (use `\citet` /
  `\citep` only — the guidelines grade citation format).
- `acl.sty`, `acl_natbib.bst` — official ACL style files (downloaded from
  github.com/acl-org/acl-style-files).

## Compiling

No LaTeX is installed on this machine. Easiest: create a new **Overleaf** project
and upload this whole folder (`main.tex` as the main document). Or locally:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Rules from the course guidelines to remember

- **8 pages max**, excluding references and appendix.
- Figures must be **PDF** (not PNG/JPEG), with large fonts.
- An **AI Disclosure and Reflection** section is mandatory (already stubbed).
- Yiddish (Hebrew-script) text does not render well under pdflatex + ACL style:
  use YIVO romanization in running text, or render trees as separate PDF figures.
