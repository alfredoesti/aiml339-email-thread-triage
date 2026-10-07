"""
bc3_summarize.py
Generates LLM summaries for the 6 BC3 test threads under two conditions:
  Condition A — raw email text only
  Condition B — email text + BERT+LoRA predicted speech act labels

Outputs:
  results/bc3_summaries_condition_a.csv
  results/bc3_summaries_condition_b.csv

Requires GEMINI_API_KEY in a .env file at the repo root.
Resume-safe: if a CSV already exists, threads already present are skipped.
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

MODEL      = "gemini-3.8-flash"
PAUSE_SEC  = 30  # gemini-3.8-flash free tier is stricter — 2 RPM safe margin

SYSTEM_INSTRUCTION = (
    "You are an expert at summarizing professional email threads for a research project.\n\n"
    "Write a factual summary of 150-200 words in plain prose (no bullet points, no markdown headers).\n\n"
    "Follow this priority order:\n"
    "1. State the main topic of the thread and its outcome or current status.\n"
    "2. Include every concrete decision reached and every commitment made, using the exact "
    "names of the people involved (e.g. \"Charles agreed to...\", \"the group decided to...\").\n"
    "3. List all action items explicitly, naming who is responsible and any deadline mentioned.\n"
    "4. Include other discussion points only if they directly influenced a decision or action item.\n\n"
    "Omit opinions, greetings, and off-topic remarks. Be specific: prefer "
    "\"October 7-8 at the Royal Sonesta Hotel\" over \"a future date and location\". "
    "Base the summary only on the content of the thread below — do not invent names, "
    "dates, or details that are not present in it."
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

# Load existing results if present (for resume)
RESULTS.mkdir(exist_ok=True)
path_a = RESULTS / "bc3_summaries_condition_a.csv"
path_b = RESULTS / "bc3_summaries_condition_b.csv"

done_a = set(pd.read_csv(path_a)["listno"].tolist()) if path_a.exists() else set()
done_b = set(pd.read_csv(path_b)["listno"].tolist()) if path_b.exists() else set()

if done_a or done_b:
    print(f"Resuming — already done: A={len(done_a)} threads, B={len(done_b)} threads")

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
        "Each email below is annotated with one or more speech-act labels assigned by a classifier "
        "(Request / Propose / Commit / Meeting / Subjective / Informative). "
        "Use them as signals for what to prioritize, not as a script to follow literally:\n"
        "- Commit: a firm commitment was made — report it among the decisions/action items.\n"
        "- Request: something is being asked for — include it if it drives the thread toward a decision or action.\n"
        "- Propose: a proposal was made — state whether it was accepted, rejected, or left open.\n"
        "- Meeting: scheduling or logistics — include concrete dates, times, and locations.\n"
        "- Subjective: personal opinion — include only if it changed the outcome of the discussion.\n"
        "- Informative: background information — include only if necessary to understand the decision or outcome.\n\n"
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
    for attempt in range(6):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={"system_instruction": SYSTEM_INSTRUCTION},
            )
            return response.text.strip()
        except Exception as e:
            if "503" in str(e):
                wait = 60 * (attempt + 1)  # 60s, 120s, 180s… backoff
                print(f" [503, retrying in {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Failed after 6 attempts")


def append_row(path: Path, row: dict) -> None:
    """Append a single row to a CSV, creating it with headers if needed."""
    df_new = pd.DataFrame([row])
    if path.exists():
        df_new.to_csv(path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(path, index=False)


for listno in test_threads:
    thread_df    = test_emails[test_emails["listno"] == listno].sort_values("email_num")
    thread_preds = preds_df[preds_df["listno"] == listno]
    thread_name  = thread_df["thread_name"].iloc[0]
    n_emails     = len(thread_df)

    print(f"\n[{listno}] {thread_name} ({n_emails} emails)")

    # --- Condition A ---
    if listno in done_a:
        print("  Condition A ... skipped (already done)")
    else:
        print("  Condition A ...", end=" ", flush=True)
        summary_a = summarize(build_raw_prompt(thread_df))
        append_row(path_a, {
            "listno": listno, "thread_name": thread_name,
            "n_emails": n_emails, "summary_a": summary_a,
        })
        print("done")
        time.sleep(PAUSE_SEC)

    # --- Condition B ---
    if listno in done_b:
        print("  Condition B ... skipped (already done)")
    else:
        print("  Condition B ...", end=" ", flush=True)
        summary_b = summarize(build_labeled_prompt(thread_df, thread_preds))
        append_row(path_b, {
            "listno": listno, "thread_name": thread_name,
            "n_emails": n_emails, "summary_b": summary_b,
        })
        print("done")
        time.sleep(PAUSE_SEC)

print(f"\nSaved:\n  {path_a}\n  {path_b}")
