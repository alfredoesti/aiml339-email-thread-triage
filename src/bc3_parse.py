"""
BC3 corpus parser and preparation (Email Thread Triage - AIML339).

v2: MULTI-LABEL classification at the email level (instead of a single
label by majority vote), plus a per-email inter-annotator agreement
score, as decided on 2026-09-01 after reviewing the pros/cons of the
initial single-label approach.

Reads corpus.xml (raw emails, with numbered sentences) and annotation.xml
(reference summaries + speech-act labels per sentence and annotator),
merges them, aggregates the labels at the email level as a multi-label
vector (the union of what any annotator marked), and generates a
train/val/test split at the level of the WHOLE thread.
"""
import xml.etree.ElementTree as ET
import pandas as pd
import random
import json
from collections import Counter
from itertools import combinations

CORPUS_XML = "../data/raw/corpus.xml"
ANNOTATION_XML = "../data/raw/annotation.xml"

SPEECH_ACT_TAGS = ["req", "prop", "cmt", "meet", "subj"]
LABEL_NAMES = {
    "req": "Request",
    "prop": "Propose",
    "cmt": "Commit",
    "meet": "Meeting",
    "subj": "Subjective",
}
ALL_LABELS = list(LABEL_NAMES.values()) + ["Informative"]

# "Importance" order used to pick a secondary primary label (auxiliary
# column only): it prioritises the speech acts that are more actionable
# for summarization, following the intuition of Carvalho & Cohen / the
# Ulrich thesis, not statistical frequency.
PRIORITY_ORDER = ["Commit", "Request", "Propose", "Meeting", "Subjective", "Informative"]


# ---------- 1. Parse corpus.xml: threads -> emails -> sentences ----------

def parse_corpus(path):
    tree = ET.parse(path)
    root = tree.getroot()
    emails = []
    sentences = []
    for thread in root.findall("thread"):
        listno = thread.findtext("listno")
        thread_name = thread.findtext("name")
        email_num = 0
        for doc in thread.findall("DOC"):
            email_num += 1
            received = doc.findtext("Received")
            frm = doc.findtext("From")
            to = doc.findtext("To")
            subject = doc.findtext("Subject")
            text_el = doc.find("Text")
            sent_texts = []
            if text_el is not None:
                for sent in text_el.findall("Sent"):
                    sid = sent.get("id")
                    stext = (sent.text or "").strip()
                    sent_texts.append((sid, stext))
                    sentences.append({
                        "listno": listno,
                        "email_num": email_num,
                        "sent_id": sid,
                        "text": stext,
                    })
            body = " ".join(t for _, t in sent_texts)
            emails.append({
                "listno": listno,
                "thread_name": thread_name,
                "email_num": email_num,
                "received": received,
                "from": frm,
                "to": to,
                "subject": subject,
                "body": body,
                "n_sentences": len(sent_texts),
            })
    return pd.DataFrame(emails), pd.DataFrame(sentences)


# ---------- 2. Parse annotation.xml: per-sentence speech-act labels (multi-annotator) ----------

def parse_annotations(path):
    tree = ET.parse(path)
    root = tree.getroot()
    label_rows = []
    summary_rows = []
    for thread in root.findall("thread"):
        listno = thread.findtext("listno")
        for ann in thread.findall("annotation"):
            annotator = ann.findtext("desc")
            summary_el = ann.find("summary")
            if summary_el is not None:
                for s in summary_el.findall("sent"):
                    summary_rows.append({
                        "listno": listno,
                        "annotator": annotator,
                        "link": s.get("link"),
                        "summary_text": (s.text or "").strip(),
                    })
            labels_el = ann.find("labels")
            if labels_el is not None:
                for child in labels_el:
                    tag = child.tag
                    if tag in SPEECH_ACT_TAGS:
                        sid = child.get("id")
                        email_num = int(sid.split(".")[0])
                        label_rows.append({
                            "listno": listno,
                            "annotator": annotator,
                            "email_num": email_num,
                            "sent_id": sid,
                            "speech_act": LABEL_NAMES[tag],
                        })
    return pd.DataFrame(label_rows), pd.DataFrame(summary_rows)


# ---------- 3. Multi-label per email (union of annotators) + inter-annotator agreement ----------

def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def aggregate_email_labels_multilabel(label_df):
    """
    For each (listno, email_num):
      - per annotator, the set of categories used in that email
      - the final email label = UNION of those sets across all
        annotators (if nobody marked anything -> {'Informative'})
      - inter-annotator agreement = mean Jaccard between each pair of
        annotators who annotated that email (NaN if only 1 annotator)
    """
    rows = []
    grouped = label_df.groupby(["listno", "email_num"])
    for (listno, email_num), g in grouped:
        per_annotator = g.groupby("annotator")["speech_act"].apply(set).to_dict()
        annotators = list(per_annotator.keys())
        union_labels = set()
        for s in per_annotator.values():
            union_labels |= s
        if not union_labels:
            union_labels = {"Informative"}

        if len(annotators) >= 2:
            pair_scores = [jaccard(per_annotator[a], per_annotator[b])
                           for a, b in combinations(annotators, 2)]
            agreement = sum(pair_scores) / len(pair_scores)
        else:
            agreement = float("nan")

        # secondary primary label (reference only), by actionability priority
        primary = next((lbl for lbl in PRIORITY_ORDER if lbl in union_labels), "Informative")

        rows.append({
            "listno": listno,
            "email_num": email_num,
            "labels": sorted(union_labels),
            "n_annotators": len(annotators),
            "annotator_agreement": agreement,
            "primary_label": primary,
        })
    return pd.DataFrame(rows)


# ---------- 4. Thread-level train/val/test split ----------

def thread_level_split(listnos, seed=42, train_frac=0.7, val_frac=0.15):
    listnos = sorted(set(listnos))
    rng = random.Random(seed)
    rng.shuffle(listnos)
    n = len(listnos)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = listnos[:n_train]
    val = listnos[n_train:n_train + n_val]
    test = listnos[n_train + n_val:]
    return train, val, test


def main():
    email_df, sentence_df = parse_corpus(CORPUS_XML)
    label_df, summary_df = parse_annotations(ANNOTATION_XML)

    print("=== Basic EDA ===")
    print(f"Threads: {email_df['listno'].nunique()}")
    print(f"Total emails: {len(email_df)}")
    print(f"Total sentences: {len(sentence_df)}")
    print()

    email_labels = aggregate_email_labels_multilabel(label_df)

    merged = email_df.merge(
        email_labels[["listno", "email_num", "labels", "n_annotators", "annotator_agreement", "primary_label"]],
        on=["listno", "email_num"], how="left"
    )
    # Emails with no row in label_df (no annotator marked anything) -> Informative, 0 annotators with an act
    merged["labels"] = merged["labels"].apply(lambda x: x if isinstance(x, list) else ["Informative"])
    merged["primary_label"] = merged["primary_label"].fillna("Informative")
    merged["n_annotators"] = merged["n_annotators"].fillna(0).astype(int)

    # multi-label binary columns
    for lbl in ALL_LABELS:
        merged[f"label_{lbl}"] = merged["labels"].apply(lambda ls, lbl=lbl: int(lbl in ls))

    merged["label_set_str"] = merged["labels"].apply(lambda ls: ",".join(ls))
    merged["n_labels"] = merged["labels"].apply(len)

    print("=== Multi-label distribution (one email can count in several rows) ===")
    for lbl in ALL_LABELS:
        print(f"{lbl}: {merged[f'label_{lbl}'].sum()}")
    print()
    print(f"Emails with more than one active label: {(merged['n_labels'] > 1).sum()} / {len(merged)}")
    print()

    print("=== Inter-annotator agreement (mean Jaccard per email, only emails with >=2 annotators) ===")
    valid_agreement = merged["annotator_agreement"].dropna()
    print(f"Emails with >=2 annotators: {len(valid_agreement)} / {len(merged)}")
    print(f"Mean agreement: {valid_agreement.mean():.3f}")
    print(f"Median agreement: {valid_agreement.median():.3f}")
    print(f"Emails with perfect agreement (=1.0): {(valid_agreement == 1.0).sum()}")
    print(f"Emails with zero agreement (=0.0): {(valid_agreement == 0.0).sum()}")
    print()

    train_ids, val_ids, test_ids = thread_level_split(email_df["listno"].unique())
    print(f"Threads -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    def tag_split(listno):
        if listno in train_ids:
            return "train"
        if listno in val_ids:
            return "val"
        return "test"

    merged["split"] = merged["listno"].apply(tag_split)
    print()
    print("=== Emails per split ===")
    print(merged["split"].value_counts())
    print()
    print("=== Primary label (secondary, by actionability priority) x split ===")
    print(pd.crosstab(merged["primary_label"], merged["split"]))

    # Save results
    out_cols = ["listno", "thread_name", "email_num", "received", "from", "to", "subject",
                "body", "n_sentences", "split", "n_annotators", "annotator_agreement",
                "primary_label", "label_set_str", "n_labels"] + [f"label_{l}" for l in ALL_LABELS]
    merged[out_cols].to_csv("../data/bc3_emails_labeled.csv", index=False)
    sentence_df.to_csv("../data/bc3_sentences.csv", index=False)
    summary_df.to_csv("../data/bc3_summaries.csv", index=False)
    with open("../data/bc3_split.json", "w") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)

    print("\nSaved: bc3_emails_labeled.csv (multi-label), bc3_sentences.csv, bc3_summaries.csv, bc3_split.json")


if __name__ == "__main__":
    main()
