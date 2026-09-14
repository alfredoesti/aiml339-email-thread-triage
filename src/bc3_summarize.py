"""
bc3_summarize.py
Generates LLM summaries for the 6 BC3 test threads under two conditions:
  Condition A — raw email text only
  Condition B — email text + BERT+LoRA predicted speech act labels

Outputs:
  results/bc3_summaries_condition_a.csv
  results/bc3_summaries_condition_b.csv

Requires GEMINI_API_KEY in a .env file at the repo root.
"""

import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import os
from google import genai

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_PATH  = REPO_ROOT / "data"
RESULTS    = REPO_ROOT / "results"

LABEL_COLS = ["Request", "Propose", "Commit", "Meeting", "Subjective", "Informative"]

MODEL      = "gemini-3.6-flash"
PAUSE_SEC  = 5   # seconds between API calls to stay within free-tier rate limits

SYSTEM_INSTRUCTION = (
    "You are an assistant that summarizes email threads for a research project. "
    "Write a concise summary of 150-200 words. Cover: the main topic, key discussion "
    "points, any decisions reached, and any action items. Use plain prose, no bullet "
    "points or markdown headers."
)

# ---------------------------------------------------------------------------
# Block 1 — Load data
# ---------------------------------------------------------------------------
load_dotenv(REPO_ROOT / ".env")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

with open(DATA_PATH / "bc3_split.json") as f:
    split = json.load(f)
test_threads = split["test"]

emails_df = pd.read_csv(DATA_PATH / "bc3_emails_labeled.csv")
test_emails = emails_df[emails_df["listno"].isin(test_threads)].copy()

preds_df = pd.read_csv(RESULTS / "bc3_bert_lora_test_predictions.csv")

# ---------------------------------------------------------------------------
# Block 2 — Prompt builders
# ---------------------------------------------------------------------------

def build_raw_prompt(thread_df: pd.DataFrame) -> str:
    parts = []
    for _, row in thread_df.iterrows():
        parts.append(
            f"--- Email {int(row['email_num'])} "
            f"(From: {row['from']} | Subject: {row['subject']}) ---\n"
            f"{str(row['body']).strip()}"
        )
    return "Email thread:\n\n" + "\n\n".join(parts)


def build_labeled_prompt(thread_df: pd.DataFrame, thread_preds: pd.DataFrame) -> str:
    label_intro = (
        "Each email below has been automatically annotated with speech act labels "
        "(Request / Propose / Commit / Meeting / Subjective / Informative). "
        "Use these labels to understand each email's communicative role when writing the summary.\n\n"
        "Email thread:\n\n"
    )
    parts = []
    for _, row in thread_df.iterrows():
        email_num = int(row["email_num"])
        pred_row = thread_preds[thread_preds["email_num"] == email_num]
        if not pred_row.empty:
            active = [l for l in LABEL_COLS if pred_row.iloc[0][f"pred_{l}"] == 1]
            label_str = ", ".join(active) if active else "None"
        else:
            label_str = "Unknown"
        parts.append(
            f"--- Email {email_num} "
            f"(From: {row['from']} | Subject: {row['subject']} | Labels: {label_str}) ---\n"
            f"{str(row['body']).strip()}"
        )
    return label_intro + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Block 3 — Call Gemini for both conditions
# ---------------------------------------------------------------------------

def summarize(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"system_instruction": SYSTEM_INSTRUCTION},
    )
    return response.text.strip()


records_a = []
records_b = []

for listno in test_threads:
    thread_df    = test_emails[test_emails["listno"] == listno].sort_values("email_num")
    thread_preds = preds_df[preds_df["listno"] == listno]
    thread_name  = thread_df["thread_name"].iloc[0]
    n_emails     = len(thread_df)

    print(f"\n[{listno}] {thread_name} ({n_emails} emails)")

    # --- Condition A ---
    print("  Condition A ...", end=" ", flush=True)
    prompt_a = build_raw_prompt(thread_df)
    summary_a = summarize(prompt_a)
    records_a.append({
        "listno":      listno,
        "thread_name": thread_name,
        "n_emails":    n_emails,
        "summary_a":   summary_a,
    })
    print("done")
    time.sleep(PAUSE_SEC)

    # --- Condition B ---
    print("  Condition B ...", end=" ", flush=True)
    prompt_b = build_labeled_prompt(thread_df, thread_preds)
    summary_b = summarize(prompt_b)
    records_b.append({
        "listno":      listno,
        "thread_name": thread_name,
        "n_emails":    n_emails,
        "summary_b":   summary_b,
    })
    print("done")
    time.sleep(PAUSE_SEC)

# ---------------------------------------------------------------------------
# Block 4 — Save results
# ---------------------------------------------------------------------------
RESULTS.mkdir(exist_ok=True)

pd.DataFrame(records_a).to_csv(RESULTS / "bc3_summaries_condition_a.csv", index=False)
pd.DataFrame(records_b).to_csv(RESULTS / "bc3_summaries_condition_b.csv", index=False)

print("\nSaved:")
print(f"  {RESULTS / 'bc3_summaries_condition_a.csv'}")
print(f"  {RESULTS / 'bc3_summaries_condition_b.csv'}")
