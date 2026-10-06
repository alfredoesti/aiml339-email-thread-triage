"""
Baseline: TF-IDF + logistic regression (multi-label, OneVsRestClassifier)
Email Thread Triage - AIML339

Trains and evaluates the baseline classifier on the 5 speech-act
categories + Informative (multi-label), using the thread-level split
already generated in bc3_split.json / bc3_emails_labeled.csv.

This baseline is the comparison point for the BERT+LoRA classifier
(Task 3). It must use exactly the same label columns and the same
split so that the comparison is valid.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, hamming_loss
)

DATA_PATH = "../data/bc3_emails_labeled.csv"

ALL_LABELS = ["Request", "Propose", "Commit", "Meeting", "Subjective", "Informative"]
LABEL_COLS = [f"label_{l}" for l in ALL_LABELS]


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def main():
    df = load_data()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"Train: {len(train_df)} emails | Val: {len(val_df)} | Test: {len(test_df)}")
    print()

    # --- TF-IDF vectorization ---
    # Fitted ONLY on train, to avoid leaking val/test vocabulary.
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(train_df["body"].fillna(""))
    X_val = vectorizer.transform(val_df["body"].fillna(""))
    X_test = vectorizer.transform(test_df["body"].fillna(""))

    y_train = train_df[LABEL_COLS].values
    y_val = val_df[LABEL_COLS].values
    y_test = test_df[LABEL_COLS].values

    # --- Model: OneVsRestClassifier of logistic regression ---
    # class_weight='balanced' because the classes are imbalanced
    # (Subjective 194 vs Informative 27, see the Task 1 EDA).
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    )
    clf.fit(X_train, y_train)

    def evaluate(X, y_true, split_name):
        y_pred = clf.predict(X)
        print(f"=== {split_name} ===")
        print(f"Micro F1:  {f1_score(y_true, y_pred, average='micro', zero_division=0):.3f}")
        print(f"Macro F1:  {f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}")
        print(f"Micro Precision: {precision_score(y_true, y_pred, average='micro', zero_division=0):.3f}")
        print(f"Micro Recall:    {recall_score(y_true, y_pred, average='micro', zero_division=0):.3f}")
        print(f"Hamming loss: {hamming_loss(y_true, y_pred):.3f}")
        print()
        print("Per label:")
        print(classification_report(y_true, y_pred, target_names=ALL_LABELS, zero_division=0))
        print()
        return y_pred

    evaluate(X_train, y_train, "TRAIN (reference, not a real evaluation)")
    evaluate(X_val, y_val, "VALIDATION")
    y_pred_test = evaluate(X_test, y_test, "TEST")

    # Save test predictions to compare later with BERT+LoRA
    pred_df = test_df[["listno", "email_num"]].copy()
    for i, lbl in enumerate(ALL_LABELS):
        pred_df[f"pred_{lbl}"] = y_pred_test[:, i]
        pred_df[f"true_{lbl}"] = y_test[:, i]
    pred_df.to_csv("../results/bc3_baseline_test_predictions.csv", index=False)
    print("Saved: bc3_baseline_test_predictions.csv")


if __name__ == "__main__":
    main()
