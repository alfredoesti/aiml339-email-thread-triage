"""
gmail_triage_agent.py
---------------------
Core logic for the Gmail Triage Agent.

For each unprocessed thread in the Inbox, uses Gemini to classify it into
one of six categories and applies the corresponding Gmail label:

    Triage/Universidad  — academic emails (university, courses, professors)
    Triage/Trabajo      — work or internship-related emails
    Triage/Personal     — friends, family, personal matters
    Triage/Newsletters  — newsletters, subscriptions, digests
    Triage/Spam         — unwanted email that isn't worth blocking
    Triage/Otro         — anything that doesn't fit the above

Results are appended to results/gmail_triage_log.csv so you can review
what the agent did and spot any misclassifications.
"""

import os
import time
import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai
from googleapiclient.errors import HttpError

from gmail_auth import get_gmail_service

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "gmail_triage"
LOG_FILE    = RESULTS_DIR / "triage_log.csv"

GEMINI_MODEL      = "gemini-1.5-flash"  # 1500 req/day free tier (vs 20 of gemini-3.6-flash)
PAUSE_SEC         = 13                   # free tier allows 5 RPM → need ≥12s between calls
FIRST_RUN_LIMIT   = 10                  # threads to process on the very first run
STATE_FILE        = RESULTS_DIR / "last_run.txt"  # stores the timestamp of the last run

# The six categories the agent can assign.
CATEGORIES = [
    "Universidad",
    "Trabajo",
    "Personal",
    "Spam",
    "Otro",
]

LABEL_PREFIX = "Triage"

# One-line description of each category, included in the Gemini prompt.
CATEGORY_DESCRIPTIONS = {
    "Universidad": "university, courses, professors, academic deadlines, student admin",
    "Trabajo":     "job, internship, work projects, colleagues, recruiters",
    "Personal":    "friends, family, personal matters, or important news from explicitly "
                   "subscribed sources such as Bloomberg financial newsletters",
    "Spam":        "any unsolicited or promotional email: marketing, discounts, food delivery "
                   "(Uber Eats, DoorDash), restaurants (Chipotle, Subway, Shake Shack), "
                   "entertainment/events (Ticketmaster, Hot Wheels), retail offers, or any "
                   "email the user did not explicitly request",
    "Otro":        "transactional emails (bank fee updates, exchange delistings, account "
                   "notices) or anything that does not clearly fit the above categories",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Block 1 — Label management
# ---------------------------------------------------------------------------

def get_or_create_triage_labels(service) -> dict:
    """Return a mapping {category_name: gmail_label_id} for all Triage/* labels.

    Creates any label that does not yet exist in the account.
    """
    existing = service.users().labels().list(userId="me").execute().get("labels", [])
    existing_by_name = {lbl["name"]: lbl["id"] for lbl in existing}

    label_ids = {}
    for category in CATEGORIES:
        full_name = f"{LABEL_PREFIX}/{category}"
        if full_name in existing_by_name:
            label_ids[category] = existing_by_name[full_name]
            log.info(f"Label already exists: {full_name}")
        else:
            created = service.users().labels().create(
                userId="me",
                body={"name": full_name, "labelListVisibility": "labelShow",
                      "messageListVisibility": "show"},
            ).execute()
            label_ids[category] = created["id"]
            log.info(f"Created label: {full_name}")

    return label_ids


# ---------------------------------------------------------------------------
# Block 2 — Fetch unprocessed threads
# ---------------------------------------------------------------------------

def _read_last_run_timestamp() -> Optional[int]:
    """Return the Unix timestamp (seconds) of the last run, or None if first run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        return int(STATE_FILE.read_text().strip())
    return None


def _save_last_run_timestamp(ts: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(str(ts))


def fetch_untriaged_threads(service, label_ids: dict) -> list:
    """Return threads in the Inbox that have no Triage/* label yet.

    First run: returns the 10 most recent threads so we don't flood the
    inbox on the first execution.

    Subsequent runs: returns all threads newer than the last run timestamp,
    so the agent only ever processes new arrivals.
    """
    exclude_query = " ".join(f"-label:{LABEL_PREFIX}/{cat}" for cat in CATEGORIES)
    last_ts = _read_last_run_timestamp()

    if last_ts is None:
        # First run — grab only the most recent 10 threads.
        query = f"in:inbox {exclude_query}"
        log.info(f"First run — fetching last {FIRST_RUN_LIMIT} threads")
        results = service.users().threads().list(
            userId="me", q=query, maxResults=FIRST_RUN_LIMIT
        ).execute()
    else:
        # Subsequent runs — only threads that arrived after the last run.
        query = f"in:inbox {exclude_query} after:{last_ts}"
        log.info(f"Incremental run — fetching threads since {datetime.fromtimestamp(last_ts)}")
        results = service.users().threads().list(
            userId="me", q=query
        ).execute()

    threads = results.get("threads", [])
    log.info(f"Found {len(threads)} untriaged thread(s)")
    return threads


# ---------------------------------------------------------------------------
# Block 3 — Build context for Gemini
# ---------------------------------------------------------------------------

def _decode_body(part: dict) -> str:
    """Decode a base64url-encoded email body part into plain text."""
    data = part.get("body", {}).get("data", "")
    if not data:
        return ""
    return base64.urlsafe_b64decode(data.encode()).decode("utf-8", errors="replace")


def build_classification_context(service, thread_id: str) -> dict:
    """Extract subject, sender, and a short snippet from the first email.

    We deliberately keep this lightweight: subject + sender + ~300 chars of
    body is enough for topic classification and avoids sending sensitive
    content to the API unnecessarily.
    """
    thread = service.users().threads().get(
        userId="me", id=thread_id, format="full"
    ).execute()

    first_msg = thread["messages"][0]
    headers   = {h["name"]: h["value"] for h in first_msg["payload"]["headers"]}

    subject = headers.get("Subject", "(no subject)")
    sender  = headers.get("From", "(unknown sender)")
    n_msgs  = len(thread["messages"])

    # Try to grab the plaintext body for a short snippet.
    snippet = first_msg.get("snippet", "")
    payload = first_msg["payload"]

    # Walk MIME parts looking for text/plain.
    body_text = ""
    parts = payload.get("parts", [payload])
    for part in parts:
        if part.get("mimeType") == "text/plain":
            body_text = _decode_body(part)
            break

    # Use whichever is longer: decoded body or the Gmail snippet.
    preview = body_text[:300].strip() if body_text else snippet[:300].strip()

    return {
        "thread_id": thread_id,
        "subject":   subject,
        "sender":    sender,
        "n_msgs":    n_msgs,
        "preview":   preview,
    }


# ---------------------------------------------------------------------------
# Block 4 — Gemini classification
# ---------------------------------------------------------------------------

def _build_prompt(ctx: dict) -> str:
    category_lines = "\n".join(
        f"  - {cat}: {desc}" for cat, desc in CATEGORY_DESCRIPTIONS.items()
    )
    return (
        "You are an email triage assistant. Classify the following email thread "
        "into exactly ONE of these categories:\n"
        f"{category_lines}\n\n"
        "Reply with ONLY the category name, nothing else.\n\n"
        f"From: {ctx['sender']}\n"
        f"Subject: {ctx['subject']}\n"
        f"Preview: {ctx['preview']}\n"
        f"Emails in thread: {ctx['n_msgs']}"
    )


def classify_thread(gemini_client, ctx: dict) -> str:
    """Ask Gemini to classify the thread. Returns the category name.

    If Gemini returns something unexpected, defaults to 'Otro' so the
    agent never crashes and always applies some label.
    """
    prompt = _build_prompt(ctx)
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    raw = response.text.strip().strip("\"'").strip()

    # Validate — accept case-insensitive match.
    for cat in CATEGORIES:
        if cat.lower() == raw.lower():
            return cat

    log.warning(f"Unexpected Gemini response '{raw}' — defaulting to 'Otro'")
    return "Otro"


# ---------------------------------------------------------------------------
# Block 5 — Apply label to thread
# ---------------------------------------------------------------------------

def apply_label(service, thread_id: str, label_id: str) -> None:
    """Add the Triage label to every message in the thread."""
    service.users().threads().modify(
        userId="me",
        id=thread_id,
        body={"addLabelIds": [label_id]},
    ).execute()


# ---------------------------------------------------------------------------
# Block 6 — Main orchestrator
# ---------------------------------------------------------------------------

def run_triage() -> pd.DataFrame:
    """Run a full triage pass. Returns a DataFrame with this run's results."""
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in .env")

    gemini_client = genai.Client(api_key=api_key)
    service       = get_gmail_service()
    label_ids     = get_or_create_triage_labels(service)
    threads       = fetch_untriaged_threads(service, label_ids)

    # Save timestamp BEFORE processing so any emails arriving during the run
    # are caught in the next execution (no gap, no duplicates).
    import time as _time
    _save_last_run_timestamp(int(_time.time()))

    if not threads:
        log.info("Nothing to triage — inbox is up to date.")
        return pd.DataFrame()

    records = []
    for i, thread_stub in enumerate(threads, 1):
        thread_id = thread_stub["id"]
        log.info(f"[{i}/{len(threads)}] Processing thread {thread_id}")

        try:
            ctx      = build_classification_context(service, thread_id)
            category = classify_thread(gemini_client, ctx)
            apply_label(service, thread_id, label_ids[category])

            log.info(f"  → {category:12s}  |  {ctx['subject'][:60]}")
            records.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "thread_id": thread_id,
                "subject":   ctx["subject"],
                "sender":    ctx["sender"],
                "n_msgs":    ctx["n_msgs"],
                "category":  category,
            })

        except HttpError as exc:
            log.error(f"Gmail API error on thread {thread_id}: {exc}")

        time.sleep(PAUSE_SEC)

    df = pd.DataFrame(records)

    # Append to the running log file.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

    log.info(f"Triage complete. {len(df)} thread(s) classified. Log: {LOG_FILE}")
    return df


if __name__ == "__main__":
    run_triage()
