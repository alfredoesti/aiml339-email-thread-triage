"""
bc3_rouge.py
Evaluates the generated summaries (Condition A and B) against the BC3
human reference summaries using ROUGE-1, ROUGE-2, and ROUGE-L (F1).

Multiple annotators per thread: ROUGE is computed against each annotator
separately and the maximum score across annotators is kept (standard
practice for multi-reference summarization evaluation, used in DUC/TAC
and the original BC3 paper).

Output: results/bc3_rouge_scores.csv
"""

from pathlib import Path
import pandas as pd
from rouge_score import rouge_scorer

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA      = REPO_ROOT / "data"
RESULTS   = REPO_ROOT / "results"

METRICS = ["rouge1", "rouge2", "rougeL"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
refs_df = pd.read_csv(DATA / "bc3_summaries.csv")
gen_a   = pd.read_csv(RESULTS / "bc3_summaries_condition_a.csv")
gen_b   = pd.read_csv(RESULTS / "bc3_summaries_condition_b.csv")

test_threads = gen_a["listno"].tolist()

scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=True)

# ---------------------------------------------------------------------------
# Compute ROUGE: max over annotators
# ---------------------------------------------------------------------------
records = []

for listno in test_threads:
    thread_name = gen_a.loc[gen_a["listno"] == listno, "thread_name"].iloc[0]
    n_emails    = int(gen_a.loc[gen_a["listno"] == listno, "n_emails"].iloc[0])
    summary_a   = gen_a.loc[gen_a["listno"] == listno, "summary_a"].iloc[0]
    summary_b   = gen_b.loc[gen_b["listno"] == listno, "summary_b"].iloc[0]

    # Build one reference string per annotator (join their sentences)
    thread_refs = refs_df[refs_df["listno"] == listno]
    references  = (
        thread_refs.groupby("annotator")["summary_text"]
        .apply(lambda s: " ".join(s.tolist()))
        .tolist()
    )

    row = {"listno": listno, "thread_name": thread_name, "n_emails": n_emails}

    for cond_label, summary in [("a", summary_a), ("b", summary_b)]:
        scores_per_ref = [scorer.score(ref, summary) for ref in references]
        for metric in METRICS:
            best_f1 = max(s[metric].fmeasure for s in scores_per_ref)
            row[f"{metric}_{cond_label}"] = round(best_f1, 4)

    records.append(row)
    print(f"[{listno}] done")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df = pd.DataFrame(records)
df.to_csv(RESULTS / "bc3_rouge_scores.csv", index=False)
print(f"\nSaved: {RESULTS / 'bc3_rouge_scores.csv'}")

# Quick summary
print("\n--- Global averages ---")
for metric in METRICS:
    avg_a = df[f"{metric}_a"].mean()
    avg_b = df[f"{metric}_b"].mean()
    delta = avg_b - avg_a
    sign  = "+" if delta >= 0 else ""
    print(f"  {metric.upper():8s}  A={avg_a:.4f}  B={avg_b:.4f}  d={sign}{delta:.4f}")
