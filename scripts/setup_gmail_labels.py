"""
setup_gmail_labels.py
---------------------
ONE-TIME setup script. Run this once to create the Triage/* labels in
your Gmail account. After that, the triage agent manages them automatically.

Usage
-----
    python scripts/setup_gmail_labels.py

Expected output
---------------
    Created  Triage/Universidad
    Created  Triage/Trabajo
    Created  Triage/Personal
    Created  Triage/Newsletters
    Created  Triage/Spam
    Created  Triage/Otro
    Done — 6 label(s) ready.
"""

import sys
from pathlib import Path

# Allow imports from src/ without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gmail_auth import get_gmail_service

LABEL_PREFIX = "Triage"
CATEGORIES   = ["Universidad", "Trabajo", "Personal", "Spam", "Otro"]


def main():
    service  = get_gmail_service()
    existing = service.users().labels().list(userId="me").execute().get("labels", [])
    existing_names = {lbl["name"] for lbl in existing}

    created = 0
    for category in CATEGORIES:
        full_name = f"{LABEL_PREFIX}/{category}"
        if full_name in existing_names:
            print(f"  Already exists  {full_name}")
        else:
            service.users().labels().create(
                userId="me",
                body={
                    "name": full_name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                },
            ).execute()
            print(f"  Created         {full_name}")
            created += 1

    total = len(CATEGORIES)
    print(f"\nDone — {total} label(s) ready ({created} newly created).")


if __name__ == "__main__":
    main()
