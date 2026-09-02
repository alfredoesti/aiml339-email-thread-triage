"""
Parser y preparacion del corpus BC3 (Email Thread Triage - AIML339).

v2: clasificacion MULTI-ETIQUETA a nivel de email (en vez de una sola
etiqueta por mayoria de voto), mas calculo de acuerdo entre anotadores
por email, tal y como se decidio el 1 sep 2026 tras revisar los pros/
contras del enfoque single-label inicial.

Lee corpus.xml (emails crudos, con frases numeradas) y annotation.xml
(resumenes de referencia + etiquetas de acto de habla por frase y
anotador), los combina, agrega las etiquetas a nivel de email como un
vector multi-etiqueta (union de lo que marco cualquier anotador), y
genera un split train/val/test a nivel de HILO completo.
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

# Orden de "importancia" para elegir una etiqueta principal secundaria
# (solo como columna auxiliar): prioriza actos de habla mas accionables
# para el resumen, segun la intuicion de Carvalho&Cohen / tesis de Ulrich,
# no por frecuencia estadistica.
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


# ---------- 3. Multi-etiqueta por email (union de anotadores) + acuerdo entre anotadores ----------

def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def aggregate_email_labels_multilabel(label_df):
    """
    Para cada (listno, email_num):
      - por anotador, el conjunto de categorias que uso en ese email
      - la etiqueta final del email = UNION de esos conjuntos entre
        todos los anotadores (si nadie marco nada -> {'Informative'})
      - acuerdo entre anotadores = Jaccard medio entre cada par de
        anotadores que anotaron ese email (NaN si solo hay 1 anotador)
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

        # etiqueta principal secundaria (solo de referencia), por prioridad de accionabilidad
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

    print("=== EDA basica ===")
    print(f"Hilos (threads): {email_df['listno'].nunique()}")
    print(f"Emails totales: {len(email_df)}")
    print(f"Frases totales: {len(sentence_df)}")
    print()

    email_labels = aggregate_email_labels_multilabel(label_df)

    merged = email_df.merge(
        email_labels[["listno", "email_num", "labels", "n_annotators", "annotator_agreement", "primary_label"]],
        on=["listno", "email_num"], how="left"
    )
    # Emails sin ninguna fila en label_df (ningun anotador marco nada) -> Informative, 0 anotadores con acto
    merged["labels"] = merged["labels"].apply(lambda x: x if isinstance(x, list) else ["Informative"])
    merged["primary_label"] = merged["primary_label"].fillna("Informative")
    merged["n_annotators"] = merged["n_annotators"].fillna(0).astype(int)

    # columnas binarias multi-etiqueta
    for lbl in ALL_LABELS:
        merged[f"label_{lbl}"] = merged["labels"].apply(lambda ls, lbl=lbl: int(lbl in ls))

    merged["label_set_str"] = merged["labels"].apply(lambda ls: ",".join(ls))
    merged["n_labels"] = merged["labels"].apply(len)

    print("=== Distribucion multi-etiqueta (un email puede contar en varias filas) ===")
    for lbl in ALL_LABELS:
        print(f"{lbl}: {merged[f'label_{lbl}'].sum()}")
    print()
    print(f"Emails con mas de una etiqueta activa: {(merged['n_labels'] > 1).sum()} / {len(merged)}")
    print()

    print("=== Acuerdo entre anotadores (Jaccard medio por email, solo emails con >=2 anotadores) ===")
    valid_agreement = merged["annotator_agreement"].dropna()
    print(f"Emails con >=2 anotadores: {len(valid_agreement)} / {len(merged)}")
    print(f"Acuerdo medio: {valid_agreement.mean():.3f}")
    print(f"Acuerdo mediana: {valid_agreement.median():.3f}")
    print(f"Emails con acuerdo perfecto (=1.0): {(valid_agreement == 1.0).sum()}")
    print(f"Emails con acuerdo nulo (=0.0): {(valid_agreement == 0.0).sum()}")
    print()

    train_ids, val_ids, test_ids = thread_level_split(email_df["listno"].unique())
    print(f"Hilos -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    def tag_split(listno):
        if listno in train_ids:
            return "train"
        if listno in val_ids:
            return "val"
        return "test"

    merged["split"] = merged["listno"].apply(tag_split)
    print()
    print("=== Emails por split ===")
    print(merged["split"].value_counts())
    print()
    print("=== Etiqueta principal (secundaria, por prioridad de accionabilidad) x split ===")
    print(pd.crosstab(merged["primary_label"], merged["split"]))

    # Guardar resultados
    out_cols = ["listno", "thread_name", "email_num", "received", "from", "to", "subject",
                "body", "n_sentences", "split", "n_annotators", "annotator_agreement",
                "primary_label", "label_set_str", "n_labels"] + [f"label_{l}" for l in ALL_LABELS]
    merged[out_cols].to_csv("../data/bc3_emails_labeled.csv", index=False)
    sentence_df.to_csv("../data/bc3_sentences.csv", index=False)
    summary_df.to_csv("../data/bc3_summaries.csv", index=False)
    with open("../data/bc3_split.json", "w") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)

    print("\nGuardado: bc3_emails_labeled.csv (multi-etiqueta), bc3_sentences.csv, bc3_summaries.csv, bc3_split.json")


if __name__ == "__main__":
    main()
