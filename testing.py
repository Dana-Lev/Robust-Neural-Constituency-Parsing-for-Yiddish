# -*- coding: utf-8 -*-
"""
LLM baseline for Yiddish constituency parsing (PPCHY).

Answers the question raised at the proposal stage: the Yiddish parsing gap was
identified by Kulick et al. (2022) -- can a modern frontier LLM (e.g. Gemini)
already produce PPCHY-style constituency parses zero-shot, or is a dedicated
parser still needed?

The script sends each test sentence to the Gemini API, asks for a bracketed
S-expression tree, and scores the prediction against the gold tree with
EVALB-style Labeled/Unlabeled P/R/F1 (corpus-level micro + per-sentence macro).

USAGE
-----
    export GEMINI_API_KEY="..."          # never hardcode the key in this file

    # Real evaluation for the report: sample sentences from the PPCHY test split
    python testing.py --data Neural-Constituency-Parsing-for-Yiddish-via-Vocabulary-Adaptation-main/yiddish_parser/data/processed/supar_ready/test.txt --n 30

    # Few-shot variant (exemplars drawn from the training split)
    python testing.py --data .../test.txt --shots 3 --train-file .../train.txt

    # No --data: falls back to 30 SYNTHETIC pilot sentences (smoke test only --
    # these are hand-written, NOT from PPCHY, and must not be reported as such)
    python testing.py

Scoring conventions (kept consistent with the SuPar evaluation config used to
train the parser, so the numbers are comparable to its Labeled F1):
  * preterminals (POS tags over a single token) are not counted as constituents
  * labels in DELETE_LABELS (TOP, S1, -NONE-, punctuation) are skipped
  * co-indexation suffixes are stripped from labels (WNP-1 -> WNP)
  * ADVP/PRT are treated as equal, as in the SuPar 'equal' setting
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

from nltk.tree import Tree

# ---------------------------------------------------------------------------
# EVALB-STYLE SCORING
# ---------------------------------------------------------------------------

DELETE_LABELS = {"TOP", "S1", "-NONE-", ",", ":", "``", "''", ".", "?", "!", ""}
EQUAL_LABELS = {"ADVP": "PRT"}


def normalize_label(label):
    """Strip co-indexation (WNP-1 -> WNP, NP=2 -> NP) and apply the equal-map."""
    label = re.sub(r"[-=]\d+$", "", label)
    return EQUAL_LABELS.get(label, label)


def get_constituents(tree, start=0):
    """
    Recursively extract labeled spans (label, start, end) from an NLTK Tree.
    Preterminals (a POS tag over one leaf) are excluded, as in EVALB.
    Returns (list_of_spans, end_index).
    """
    spans = []
    end = start
    for child in tree:
        if isinstance(child, Tree):
            child_spans, end = get_constituents(child, end)
            spans.extend(child_spans)
        else:
            end += 1
    is_preterminal = len(tree) == 1 and not isinstance(tree[0], Tree)
    if not is_preterminal:
        label = normalize_label(tree.label())
        if label not in DELETE_LABELS:
            spans.append((label, start, end))
    return spans, end


def parse_tree_string(tree_str):
    """Parse an S-expression into an NLTK Tree; returns None on malformed input."""
    try:
        cleaned = re.sub(r"^```[a-zA-Z]*\s*|```\s*$", "", tree_str.strip(), flags=re.MULTILINE).strip()
        return Tree.fromstring(cleaned)
    except Exception:
        return None


def score_pair(gold_tree, pred_tree):
    """
    EVALB-style counts for one sentence. Uses multisets (Counter), like EVALB,
    rather than sets, so duplicate spans (e.g. unary chains) are counted fairly.
    Returns a dict of counts for labeled and unlabeled matching.
    """
    gold_spans, _ = get_constituents(gold_tree)
    pred_spans, _ = get_constituents(pred_tree)

    gold_l, pred_l = Counter(gold_spans), Counter(pred_spans)
    gold_u = Counter((s, e) for (_, s, e) in gold_spans)
    pred_u = Counter((s, e) for (_, s, e) in pred_spans)

    return {
        "labeled_match": sum((gold_l & pred_l).values()),
        "unlabeled_match": sum((gold_u & pred_u).values()),
        "gold_count": len(gold_spans),
        "pred_count": len(pred_spans),
    }


def prf(match, pred_count, gold_count):
    p = match / pred_count if pred_count else 0.0
    r = match / gold_count if gold_count else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_treebank(path, max_len=40):
    """Read a SuPar-format file (one bracketed tree per line) into items."""
    items = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            tree = parse_tree_string(line)
            if tree is None:
                continue
            leaves = tree.leaves()
            if not leaves or len(leaves) > max_len:
                continue
            items.append({
                "id": f"{os.path.basename(path)}:{line_no}",
                "tokens": leaves,
                "sentence": " ".join(leaves),
                "gold_tree": line,
            })
    return items


# 30 hand-written pilot sentences. These are SYNTHETIC examples in PPCHY-like
# style, useful only as a smoke test of the pipeline when the real treebank is
# not on this machine. Do NOT report scores on them as PPCHY results.
PILOT_DATA = [
    {"id": "pilot_01", "sentence": "דאס איז א פשוטע ביישפיל",
     "gold_tree": "(S (NP-SBJ (D דאס)) (VP (V איז) (NP-PRD (D א) (ADJ פשוטע) (N ביישפיל))))"},
    {"id": "pilot_02", "sentence": "און ער האט גערעדט צו אים",
     "gold_tree": "(S (CONJ און) (NP-SBJ (PRO ער)) (VP (H האט) (VBN גערעדט) (PP (P צו) (NP (PRO אים)))))"},
    {"id": "pilot_03", "sentence": "מיר ווילן גיין איצט",
     "gold_tree": "(S (NP-SBJ (PRO מיר)) (MD ווילן) (VP (VB גיין) (ADVP-TMP (ADV איצט))))"},
    {"id": "pilot_04", "sentence": "דער קליינער אינגל לויפט שנעל",
     "gold_tree": "(S (NP-SBJ (D דער) (ADJ קליינער) (N אינגל)) (VP (V לויפט) (ADVP-MNR (ADV שנעל))))"},
    {"id": "pilot_05", "sentence": "זי האט אים געגעבן דאס בוך",
     "gold_tree": "(S (NP-SBJ (PRO זי)) (VP (H האט) (NP-OB2 (PRO אים)) (VBN געגעבן) (NP-OB1 (D דאס) (N בוך))))"},
    {"id": "pilot_06", "sentence": "מען דארף לערנען די תורה",
     "gold_tree": "(S (NP-SBJ (MAN מען)) (MD דארף) (VP (VB לערנען) (NP-OB1 (D די) (NPR תורה))))"},
    {"id": "pilot_07", "sentence": "ווי אזוי איז עס געשען",
     "gold_tree": "(CP-QUE (WADVP-1 (WADV ווי) (ADV אזוי)) (IP-SUB (BE איז) (NP-SBJ (PRO עס)) (VP (VBN געשען))))"},
    {"id": "pilot_08", "sentence": "דאס קינד איז זייער שיין",
     "gold_tree": "(S (NP-SBJ (D דאס) (N קינד)) (VP (BE איז) (ADJP-PRD (ADV זייער) (ADJ שיין))))"},
    {"id": "pilot_09", "sentence": "איך ווייס נישט וואס צו טאן",
     "gold_tree": "(S (NP-SBJ (PRO איך)) (VP (V ווייס) (NEG נישט) (CP-QUE (WNP-1 (WPRO וואס)) (IP-INF (TO צו) (VB טאן)))))"},
    {"id": "pilot_10", "sentence": "דער שטאט איז גרויס און אלט",
     "gold_tree": "(S (NP-SBJ (D דער) (N שטאט)) (VP (BE איז) (ADJP-PRD (ADJ גרויס) (CONJ און) (ADJ אלט))))"},
    {"id": "pilot_11", "sentence": "האסטו געזען דעם שיינעם הויז",
     "gold_tree": "(CP-QUE (IP-SUB (H האסטו) (VP (VBN געזען) (NP-OB1 (D דעם) (ADJ שיינעם) (N הויז)))))"},
    {"id": "pilot_12", "sentence": "ער קען נישט קומען היינט",
     "gold_tree": "(S (NP-SBJ (PRO ער)) (MD קען) (NEG נישט) (VP (VB קומען) (ADVP-TMP (ADV היינט))))"},
    {"id": "pilot_13", "sentence": "די גרויסע פראגע בלייבט אפן",
     "gold_tree": "(S (NP-SBJ (D די) (ADJ גרויסע) (N פראגע)) (VP (V בלייבט) (ADJP-PRD (ADJ אפן))))"},
    {"id": "pilot_14", "sentence": "זיי האבן דאס זייער גוט געמאכט",
     "gold_tree": "(S (NP-SBJ (PRO זיי)) (VP (H האבן) (NP-OB1 (D דאס)) (ADVP-MNR (ADV זייער) (ADV גוט)) (VBN געמאכט)))"},
    {"id": "pilot_15", "sentence": "אדם איז געווען דער ערשטער מענטש",
     "gold_tree": "(S (NP-SBJ (NPR אדם)) (VP (BE איז) (VBN געווען) (NP-PRD (D דער) (ADJ ערשטער) (N מענטש))))"},
    {"id": "pilot_16", "sentence": "מיר שרייבן א בריוו צו אונדזער לערער",
     "gold_tree": "(S (NP-SBJ (PRO מיר)) (VP (V שרייבן) (NP-OB1 (D א) (N בריוו)) (PP (P צו) (NP (PRO$ אונדזער) (N לערער)))))"},
    {"id": "pilot_17", "sentence": "דער וואלד איז פול מיט ביימער",
     "gold_tree": "(S (NP-SBJ (D דער) (N וואלד)) (VP (BE איז) (ADJP-PRD (ADJ פול) (PP (P מיט) (NP (N ביימער))))))"},
    {"id": "pilot_18", "sentence": "איך האב געזען די לעוו אויפן בארג",
     "gold_tree": "(S (NP-SBJ (PRO איך)) (VP (H האב) (VBN געזען) (NP-OB1 (D די) (N לעוו)) (PP (P-D אויפן) (NP (N בארג)))))"},
    {"id": "pilot_19", "sentence": "איז דאס אמת אדער שקר",
     "gold_tree": "(CP-QUE (IP-SUB (BE איז) (NP-SBJ (D דאס)) (ADJP-PRD (ADJ אמת) (CONJ אדער) (ADJ שקר))))"},
    {"id": "pilot_20", "sentence": "די לערער שטיצט אונדזער פראיעקט",
     "gold_tree": "(S (NP-SBJ (D די) (N לערער)) (VP (V שטיצט) (NP-OB1 (PRO$ אונדזער) (N פראיעקט))))"},
    {"id": "pilot_21", "sentence": "מען האט אים גערופן ביים נאמען",
     "gold_tree": "(S (NP-SBJ (MAN מען)) (VP (H האט) (NP-OB1 (PRO אים)) (VBN גערופן) (PP (P-D ביים) (NP (N נאמען)))))"},
    {"id": "pilot_22", "sentence": "דער וועג איז לאנג און שווער",
     "gold_tree": "(S (NP-SBJ (D דער) (N וועג)) (VP (BE איז) (ADJP-PRD (ADJ לאנג) (CONJ און) (ADJ שווער))))"},
    {"id": "pilot_23", "sentence": "ער געדענקט וואס די חכמים האבן געזאגט",
     "gold_tree": "(S (NP-SBJ (PRO ער)) (VP (V געדענקט) (CP-QUE (WNP-1 (WPRO וואס)) (IP-SUB (NP-SBJ (D די) (NPR חכמים)) (VP (H האבן) (VBN געזאגט))))))"},
    {"id": "pilot_24", "sentence": "זייער אוצר ליגט אין טיפן ים",
     "gold_tree": "(S (NP-SBJ (PRO$ זייער) (N אוצר)) (VP (V ליגט) (PP (P אין) (NP (ADJ טיפן) (N ים)))))"},
    {"id": "pilot_25", "sentence": "די זון שיינט איבער די בערג",
     "gold_tree": "(S (NP-SBJ (D די) (N זון)) (VP (V שיינט) (PP (P איבער) (NP (D די) (N בערג)))))"},
    {"id": "pilot_26", "sentence": "איך לייען א ישן בוך אין שטוב",
     "gold_tree": "(S (NP-SBJ (PRO איך)) (VP (V לייען) (NP-OB1 (D א) (ADJ ישן) (N בוך)) (PP (P אין) (NP (N שטוב)))))"},
    {"id": "pilot_27", "sentence": "ער איז אוועקגעגאנגען פרי אינדערפרי",
     "gold_tree": "(S (NP-SBJ (PRO ער)) (VP (BE איז) (VBN אוועקגעגאנגען) (ADVP-TMP (ADV פרי) (ADV אינדערפרי))))"},
    {"id": "pilot_28", "sentence": "זיי האבן מיר געפילט דעם ענטפער",
     "gold_tree": "(S (NP-SBJ (PRO זיי)) (VP (H האבן) (NP-OB2 (PRO מיר)) (VBN געפילט) (NP-OB1 (D דעם) (N ענטפער))))"},
    {"id": "pilot_29", "sentence": "מיר וועלן בלייבן דא ביז מארגן",
     "gold_tree": "(S (NP-SBJ (PRO מיר)) (MD וועלן) (VP (VB בלייבן) (ADVP-LOC (ADV דא)) (PP (P ביז) (NP (N מארגן)))))"},
    {"id": "pilot_30", "sentence": "די פרייד פון לערנען איז גרויס",
     "gold_tree": "(S (NP-SBJ (D די) (N פרייד) (PP (P פון) (IP-INF (VB לערנען)))) (VP (BE איז) (ADJP-PRD (ADJ גרויס))))"},
]


def pilot_items():
    items = []
    for entry in PILOT_DATA:
        tree = parse_tree_string(entry["gold_tree"])
        items.append({
            "id": entry["id"],
            "tokens": tree.leaves(),
            "sentence": entry["sentence"],
            "gold_tree": entry["gold_tree"],
        })
    return items


# ---------------------------------------------------------------------------
# PROMPT & MODEL
# ---------------------------------------------------------------------------

SYSTEM_RULES = (
    "You are an expert syntactic parser for historical Yiddish, annotated in the "
    "style of the Penn Parsed Corpus of Historical Yiddish (PPCHY): clause labels "
    "such as IP-MAT / IP-SUB / CP-QUE, phrase labels with function tags such as "
    "NP-SBJ / NP-OB1 / NP-PRD / PP / ADVP-TMP, and a POS preterminal above every token.\n"
    "RULES:\n"
    "1. Output ONLY the raw bracketed S-expression tree, nothing else.\n"
    "2. Do NOT wrap the output in markdown code fences.\n"
    "3. Use EXACTLY the given tokens as the leaves, in the given order -- do not "
    "split, merge, add, or drop tokens. Put exactly one POS preterminal above each token.\n"
)


def build_prompt(item, exemplars):
    parts = [SYSTEM_RULES]
    if exemplars:
        parts.append("\nEXAMPLES:\n")
        for ex in exemplars:
            parts.append(f"Tokens: {' '.join(ex['tokens'])}\nTree: {ex['gold_tree']}\n")
    parts.append(f"\nNow parse this sentence.\nTokens: {' '.join(item['tokens'])}\nTree:")
    return "".join(parts)


def make_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set the GEMINI_API_KEY environment variable first, e.g.\n"
                 "  export GEMINI_API_KEY=\"your-key\"")
    from google import genai
    return genai.Client(api_key=api_key)


def predict_tree(client, model, prompt, max_attempts=8, backoff=20):
    """Call the API with retry on rate limits. Returns (text, error)."""
    from google.genai import types

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return (response.text or "").strip(), None
        except Exception as exc:  # noqa: BLE001 - we inspect and re-raise below
            message = str(exc)
            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                print(f"  rate limit hit (attempt {attempt}/{max_attempts}); "
                      f"sleeping {backoff}s ...")
                time.sleep(backoff)
                continue
            return None, message
    return None, f"gave up after {max_attempts} rate-limit retries"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def summarize(results, model, shots, source, seed):
    """
    Aggregate per-sentence records into reportable metrics.

    An API failure is NOT a model failure, and conflating the two understates the
    model: a run that loses sentences to free-tier rate limits would otherwise
    look like a model that emits malformed trees. Rates over the model's actual
    behaviour are therefore computed over *responses received*, and the requests
    that never returned are reported separately as coverage.
    """
    n = len(results)
    errored = [r for r in results if r.get("error")]
    got = [r for r in results if not r.get("error")]
    n_got = len(got)

    totals = {"labeled_match": 0, "unlabeled_match": 0, "gold_count": 0, "pred_count": 0}
    for record in results:
        counts = record.get("counts")
        if counts:
            for key in totals:
                totals[key] += counts[key]

    lp, lr, lf = prf(totals["labeled_match"], totals["pred_count"], totals["gold_count"])
    up, ur, uf = prf(totals["unlabeled_match"], totals["pred_count"], totals["gold_count"])
    macro_got = (sum(r["labeled_f1"] for r in got) / n_got) if n_got else 0.0

    return {
        "model": model, "shots": shots, "data": source, "seed": seed,
        # --- coverage: how much of the sample the API actually returned ---
        "n_sentences_attempted": n,
        "n_responses_received": n_got,
        "n_api_failures": len(errored),
        "coverage_rate": round(100.0 * n_got / n, 2) if n else 0.0,
        # --- model behaviour, over responses received ---
        "valid_syntax_rate": round(100.0 * sum(r["valid_syntax"] for r in got) / n_got, 2) if n_got else 0.0,
        "token_fidelity_rate": round(100.0 * sum(r["tokens_match"] for r in got) / n_got, 2) if n_got else 0.0,
        "macro_labeled_f1": round(100 * macro_got, 2),
        # --- corpus-level micro scores, comparable to the parser's Labeled F1 ---
        "labeled_precision": round(100 * lp, 2), "labeled_recall": round(100 * lr, 2),
        "labeled_f1": round(100 * lf, 2),
        "unlabeled_precision": round(100 * up, 2), "unlabeled_recall": round(100 * ur, 2),
        "unlabeled_f1": round(100 * uf, 2),
    }


def print_summary(summary):
    print("\n" + "=" * 26 + " FINAL REPORT " + "=" * 26)
    for key, value in summary.items():
        print(f"  {key:<24} {value}")
    print("=" * 66)
    print("labeled/unlabeled P/R/F1 are corpus-level micro scores over the responses")
    print("received, directly comparable to the SuPar parser's Labeled F1.")
    print("valid_syntax_rate / token_fidelity_rate / macro_labeled_f1 are also over")
    print("responses received -- API failures are reported as coverage, not as model")
    print("errors. If coverage_rate is below 100, say so in the report and consider")
    print("re-running with a larger --sleep.")


def main():
    ap = argparse.ArgumentParser(description="Frontier-LLM baseline for Yiddish constituency parsing")
    ap.add_argument("--data", default=None,
                    help="SuPar-format PPCHY test file (one tree per line). "
                         "Without it, 30 synthetic pilot sentences are used (smoke test only).")
    ap.add_argument("--train-file", default=None,
                    help="SuPar-format train file to draw few-shot exemplars from.")
    ap.add_argument("--n", type=int, default=30, help="Number of test sentences to sample.")
    ap.add_argument("--max-len", type=int, default=40, help="Skip sentences longer than this.")
    ap.add_argument("--shots", type=int, default=0, help="Number of few-shot exemplars (0 = zero-shot).")
    ap.add_argument("--model", default="gemini-3.6-flash",
                    help="Gemini model id (check availability for your key).")
    ap.add_argument("--seed", type=int, default=1, help="Sampling seed, for reproducibility.")
    ap.add_argument("--sleep", type=float, default=13.0,
                    help="Seconds between requests (13s stays under the 5 req/min free tier).")
    ap.add_argument("--out", default="llm_baseline_results.json", help="Where to write results.")
    ap.add_argument("--recompute", metavar="RESULTS_JSON", default=None,
                    help="Rebuild the summary from a saved results file, without "
                         "calling the API. Use after a metric fix, or to re-read an "
                         "older run. Writes back in place unless --out is given.")
    args = ap.parse_args()

    if args.recompute:
        with open(args.recompute, encoding="utf-8") as handle:
            saved = json.load(handle)
        old = saved.get("summary", {})
        records = saved["results"]
        for record in records:
            if record.get("counts") is None and not record.get("error"):
                # older runs did not store counts; recover them from the trees
                gold = parse_tree_string(record["gold_tree"])
                pred = parse_tree_string(record["predicted_tree"] or "")
                if gold is not None and pred is not None:
                    record["counts"] = score_pair(gold, pred)
                elif gold is not None:
                    spans, _ = get_constituents(gold)
                    record["counts"] = {"labeled_match": 0, "unlabeled_match": 0,
                                        "gold_count": len(spans), "pred_count": 0}
        summary = summarize(records,
                            model=old.get("model", "unknown"),
                            shots=old.get("shots", 0),
                            source=old.get("data", args.recompute),
                            seed=old.get("seed", 0))
        print_summary(summary)
        out = args.out if args.out != "llm_baseline_results.json" else args.recompute
        with open(out, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "results": records}, handle,
                      ensure_ascii=False, indent=2)
        print(f"\nRewrote {out}")
        return

    rng = random.Random(args.seed)

    if args.data:
        items = load_treebank(args.data, max_len=args.max_len)
        if not items:
            sys.exit(f"No parsable trees found in {args.data}")
        if len(items) > args.n:
            items = rng.sample(items, args.n)
        source = args.data
    else:
        print("WARNING: no --data given. Falling back to 30 SYNTHETIC pilot sentences.\n"
              "         These are NOT from PPCHY -- use them only to check the pipeline,\n"
              "         never as reported results.\n")
        items = pilot_items()[: args.n]
        source = "synthetic pilot set"

    exemplars = []
    if args.shots > 0:
        if args.train_file:
            pool = load_treebank(args.train_file, max_len=25)
            exemplars = rng.sample(pool, min(args.shots, len(pool)))
        else:
            sys.exit("--shots requires --train-file (exemplars must come from the "
                     "training split, never from the evaluated data).")

    client = make_client()
    print(f"Model: {args.model} | shots: {args.shots} | data: {source} | n: {len(items)}\n")

    results = []

    for index, item in enumerate(items, 1):
        prompt = build_prompt(item, exemplars)
        raw, error = predict_tree(client, args.model, prompt)

        record = {"id": item["id"], "sentence": item["sentence"],
                  "gold_tree": item["gold_tree"], "predicted_tree": raw,
                  "valid_syntax": False, "tokens_match": False,
                  "labeled_f1": 0.0, "unlabeled_f1": 0.0, "error": error,
                  # kept so --recompute can rebuild the summary without re-querying
                  "counts": None}

        if raw is not None:
            pred = parse_tree_string(raw)
            gold = parse_tree_string(item["gold_tree"])
            if pred is not None:
                record["valid_syntax"] = True
                record["tokens_match"] = pred.leaves() == gold.leaves()
                counts = score_pair(gold, pred)
                record["counts"] = counts
                _, _, record["labeled_f1"] = prf(counts["labeled_match"],
                                                 counts["pred_count"], counts["gold_count"])
                _, _, record["unlabeled_f1"] = prf(counts["unlabeled_match"],
                                                   counts["pred_count"], counts["gold_count"])
            else:
                # A malformed tree is a real model failure: it contributes its
                # gold constituents to recall (the model matched none of them).
                gold_spans, _ = get_constituents(parse_tree_string(item["gold_tree"]))
                record["counts"] = {"labeled_match": 0, "unlabeled_match": 0,
                                    "gold_count": len(gold_spans), "pred_count": 0}

        results.append(record)
        print(f"[{index}/{len(items)}] {item['id']}")
        print(f"  sentence : {item['sentence']}")
        print(f"  predicted: {raw if raw else '(API error: ' + str(error) + ')'}")
        print(f"  LF1={record['labeled_f1']:.3f} UF1={record['unlabeled_f1']:.3f} "
              f"valid={record['valid_syntax']} tokens_ok={record['tokens_match']}")
        print("-" * 70)
        if index < len(items):
            time.sleep(args.sleep)

    summary = summarize(results, model=args.model, shots=args.shots,
                        source=source, seed=args.seed)
    print_summary(summary)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": results}, handle, ensure_ascii=False, indent=2)
    print(f"\nSaved detailed results to {args.out}")


if __name__ == "__main__":
    main()
