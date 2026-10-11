"""
stats_tests.py
Statistical tests for the two comparisons of the project. It only reads
result files that are already in results/; no model is trained and no API
is called.

1. Classifier: BERT+LoRA (30 seeds) vs TF-IDF baseline.
   The baseline is deterministic (lbfgs logistic regression on a fixed
   TF-IDF matrix), so it gives one value per metric. For each metric, the
   30 per-seed differences (BERT - baseline) are tested against zero with a
   two-sided one-sample Wilcoxon signed-rank test. This tests whether
   BERT+LoRA beats the baseline across training randomness on this fixed
   test set; it does not account for the sampling of the 37 test emails.

2. Summarization: Condition B vs Condition A.
   Paired by test thread (n = 6), two-sided exact Wilcoxon signed-rank test
   on the per-thread ROUGE F1 scores.

Effect size: matched-pairs rank-biserial correlation of the differences (B - A);
positive means B is larger (better for F1 and ROUGE, worse for Hamming loss):
    r = (W+ - W-) / (W+ + W-), in [-1, 1].
Multiple comparisons: Holm correction within each family of three metrics.
Significance threshold: alpha = 0.05.

Output: results/stats_tests.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import f1_score, hamming_loss

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
ALPHA = 0.05
LABELS = ["Request", "Propose", "Commit", "Meeting", "Subjective", "Informative"]


def rank_biserial(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (zero differences dropped)."""
    d = diffs[diffs != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    w_pos, w_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return (w_pos - w_neg) / (w_pos + w_neg)


def holm(pvalues: list) -> list:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def classifier_tests() -> list:
    base = pd.read_csv(RESULTS / "bc3_baseline_test_predictions.csv")
    y_true = base[[f"true_{l}" for l in LABELS]].values
    y_base = base[[f"pred_{l}" for l in LABELS]].values
    baseline = {
        "test_micro_f1": f1_score(y_true, y_base, average="micro", zero_division=0),
        "test_macro_f1": f1_score(y_true, y_base, average="macro", zero_division=0),
        "test_hamming_loss": hamming_loss(y_true, y_base),
    }
    runs = pd.read_csv(RESULTS / "bc3_bert_lora_seed_runs.csv")

    rows = []
    for metric, higher_better in [("test_micro_f1", True), ("test_macro_f1", True),
                                  ("test_hamming_loss", False)]:
        values = runs[metric].to_numpy()
        diffs = values - baseline[metric]
        test = wilcoxon(diffs, alternative="two-sided")
        better = (diffs > 0) if higher_better else (diffs < 0)
        rows.append({
            "comparison": "BERT+LoRA (30 seeds) vs TF-IDF baseline",
            "metric": metric.replace("test_", ""),
            "higher_is_better": higher_better,
            "n": len(values),
            "a_name": "baseline", "a_value": baseline[metric], "a_std": 0.0,
            "b_name": "bert_lora", "b_value": values.mean(), "b_std": values.std(ddof=1),
            "mean_diff_b_minus_a": diffs.mean(),
            "n_b_better": int(better.sum()),
            "test": "one-sample Wilcoxon signed-rank (two-sided)",
            "statistic": test.statistic,
            "p_value": test.pvalue,
            "effect_size_rank_biserial": rank_biserial(diffs),
        })
    for row, p_adj in zip(rows, holm([r["p_value"] for r in rows])):
        row["p_holm"] = p_adj
    return rows


def rouge_tests() -> list:
    scores = pd.read_csv(RESULTS / "bc3_rouge_scores.csv")
    rows = []
    for metric in ["rouge1", "rouge2", "rougeL"]:
        a, b = scores[f"{metric}_a"].to_numpy(), scores[f"{metric}_b"].to_numpy()
        diffs = b - a
        test = wilcoxon(b, a, alternative="two-sided", method="exact")
        rows.append({
            "comparison": "Condition B vs Condition A (6 test threads)",
            "metric": metric,
            "higher_is_better": True,
            "n": len(diffs),
            "a_name": "condition_a", "a_value": a.mean(), "a_std": a.std(ddof=1),
            "b_name": "condition_b", "b_value": b.mean(), "b_std": b.std(ddof=1),
            "mean_diff_b_minus_a": diffs.mean(),
            "n_b_better": int((diffs > 0).sum()),
            "test": "paired Wilcoxon signed-rank, exact (two-sided)",
            "statistic": test.statistic,
            "p_value": test.pvalue,
            "effect_size_rank_biserial": rank_biserial(diffs),
        })
    for row, p_adj in zip(rows, holm([r["p_value"] for r in rows])):
        row["p_holm"] = p_adj
    return rows


def main():
    df = pd.DataFrame(classifier_tests() + rouge_tests())
    df["significant_holm"] = df["p_holm"] < ALPHA
    out = RESULTS / "stats_tests.csv"
    df.to_csv(out, index=False, float_format="%.6g")

    show = ["metric", "n", "a_value", "b_value", "b_std", "n_b_better",
            "p_value", "p_holm", "effect_size_rank_biserial", "significant_holm"]
    for comparison, group in df.groupby("comparison", sort=False):
        print(f"\n=== {comparison} (alpha = {ALPHA}, Holm-corrected) ===")
        print(group[show].round(4).to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
