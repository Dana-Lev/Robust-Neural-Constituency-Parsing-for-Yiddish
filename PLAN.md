# Project plan — Yiddish parsing via subword regularization & PEFT

**Deadline: September 30, 2026** (today: Aug 18 → ~6 weeks).
Team: **Dana** (cluster + experiments) and **Ayala** (LLM baseline + writing lead).
Both: analysis, discussion, proofreading. Swap roles wherever it makes sense —
the split below balances the load, not ownership.

The scientific goal (from the proposal): test whether the subword-fragmentation
bottleneck can be fixed **dynamically** (BPE-dropout-style regularization) or
**parameter-efficiently** (MAD-X-style adapters), benchmarked against the frozen
baseline and the previous project's static FOCUS+DAPT result (83.81 / 83.02 LF).
Plus the instructor's addition: check whether **frontier LLMs already solve this**
zero-/few-shot (`testing.py`).

---

## Week 1 (Aug 18–24) — setup & sanity

**Dana**
- [ ] Fill the Slurm access form (guidelines §2.2, PI/lab = "NLP class 2025/2026"),
      confirm login on `studentkillable`.
- [ ] Set up storage under `/home/morg/NLP_2526b/<user>`: conda env `yiddish`
      (`supar==1.1.4`, `transformers`, `peft>=0.7.0`, `sentencepiece`, `nltk`),
      point `HF_HOME` there. The slurm script now reads `NLP_STORAGE`.
- [ ] Rebuild the data: clone `ppchyprep`, run the `ppchy_formatting` scripts →
      `split_supar_data.py` → `clean_tree_data.py`. Record the actual
      train/dev/test counts (the report table must use *your* numbers).
- [ ] Run `python src/train_parser_peft.py --selftest --encoder-path skulick/xlmb-ybc-ck05`
      for both backends (`--backend spm`, `--backend maxmatch`). Save the fertility
      output — it becomes a table in the report.
- [ ] **Sanity overfit** (rubric requirement): train on ~100 sentences, confirm
      train F1 ≈ 100. Keep the log.
- [ ] Launch the baseline run: `sbatch src/parser_train_peft.slurm baseline`.

**Ayala**
- [ ] Get a Gemini API key (if free-tier limits are too tight for ~70 requests/run,
      email the lecturer for API credits — guidelines §2.2 explicitly offers this;
      include group names/IDs, title, cost estimate, justification).
- [ ] Smoke-test `testing.py` with no `--data` (pilot mode) — checks the API,
      retry loop, and JSON output end to end.
- [ ] Create the Overleaf project from `report/` (main.tex compiles as-is).
- [ ] Read Kulick et al. 2022 + Provilkov 2020 + Pfeiffer 2020 closely; collect
      notes for Background (lit review is 20 pts).

## Week 2 (Aug 25–31) — main experiment grid

**Dana** — run and evaluate all cells (one code path, seed 1):
- [ ] `baseline` (done from week 1 — evaluate)
- [ ] `bpe-dropout --backend spm`
- [ ] `bpe-dropout --backend maxmatch`
- [ ] `adapters --adapter-type lora`
- [ ] `adapters --adapter-type bottleneck`
- [ ] `both --adapter-type lora`
- [ ] Evaluate each with `evaluate_peft.py` (never `Parser.load` directly for
      adapter checkpoints); log UF/LF/UCM/LCM into a shared results sheet.
- [ ] Save training curves (dev F1 per epoch) from the logs for figures.

**Ayala** — the LLM control experiment (the instructor's question):
- [ ] Copy `supar_ready/test.txt` + `train.txt` from the cluster to run locally.
- [ ] Zero-shot: `python testing.py --data .../test.txt --n 30`
- [ ] Few-shot: `python testing.py --data .../test.txt --n 30 --shots 3 --train-file .../train.txt`
- [ ] Same two runs with a second model if accessible (e.g. a GPT-4-class model —
      requires adding an OpenAI branch, or just note it as future work).
- [ ] Record exact model IDs + access dates (API models drift; the report must say).
- [ ] Draft **Introduction** and **Background** sections in Overleaf.

## Week 3 (Sept 1–7) — robustness & writing

**Dana**
- [ ] Adapter-LR sweep on the best adapter cell: `--adapter-lr 1e-4 / 3e-4 / 1e-3`
      (a flat adapter result at 5e-5-ish rates is an under-training artifact, not
      a finding).
- [ ] Seeds 2 and 3 for baseline + the two most interesting cells → mean±std
      (rubric: "always account for randomness").
- [ ] Optional stretch: `--n-bert-layers 12` baseline to connect to the previous
      project's 83.81 configuration.

**Ayala**
- [ ] Draft **Methodology** (incl. the LLM-control subsection) and **Experimental
      Setup**.
- [ ] Build figures as **PDF**: results bar chart, one training-curve plot,
      optionally one gold-vs-predicted tree; large fonts.
- [ ] Dataset statistics table from Dana's week-1 counts.

## Week 4 (Sept 8–14) — results & discussion

**Both**
- [ ] Freeze the numbers; fill Tables 1–2 in the report.
- [ ] Write **Results** (Dana leads parser tables, Ayala leads LLM table).
- [ ] Write **Discussion** together — this is where the grade lives. Whatever the
      outcome, the story is strong: either dynamic adaptation helps where static
      injection didn't, or three independent interventions all fail to move
      frozen-encoder parsing F1 → constituency parsing is resilient to subword
      fragmentation, and PEFT reaches the same F1 at ~1–2M trainable params with
      no pre-training. Relate the LLM numbers to the 83+ LF parser: does a
      generalist model make the custom parser unnecessary? Use token-fidelity and
      unlabeled-F1 to separate "doesn't know Yiddish" from "doesn't know PPCHY".

## Week 5 (Sept 15–21) — full draft

- [ ] Abstract + Conclusion + Limitations (both).
- [ ] **AI Disclosure and Reflection** — mandatory section; be concrete about
      what was AI-assisted (proposal brainstorming, PEFT integration code, the
      LLM-baseline script, LaTeX) and what you verified yourselves.
- [ ] Self-grade against the rubric: research question / ambition / lit review /
      methodology / results / presentation (each 10–20 pts).
- [ ] Check: ≤8 pages excl. references+appendix; every citation via \citet/\citep;
      figures are PDFs; no work-log style writing ("we tried X, it failed…").

## Week 6 (Sept 22–29) — buffer & submit

- [ ] Buffer for re-runs, proofread, final compile check.
- [ ] Push code to a clean GitHub repo (reproducibility is graded); README with
      exact commands. **Make sure no API key is anywhere in the history.**
- [ ] Submit by **Sept 30**.

---

## Risks / gotchas

- **Cluster queue**: `studentkillable` jobs get preempted near deadlines — start
  the grid in week 2, not week 5.
- **Storage quota**: don't checkpoint every 500 steps; SuPar saves best-only by
  default — keep it that way.
- **Gemini rate limits**: free tier ≈ 5 req/min; a 30-sentence run takes ~7 min +
  retries. If quota blocks few-shot runs, email the lecturer early (week 1).
- **Data provenance**: report numbers only from the real `supar_ready` splits.
  The 30 hardcoded sentences in `testing.py` are synthetic pilot data — never
  report them as PPCHY.
- **Comparability**: your baseline (4 layers, mean pooling) is not bit-identical
  to the previous project's published 83.81 (they report 12 layers). Compare
  against *your own* baseline in the main table; cite theirs as reference rows.
