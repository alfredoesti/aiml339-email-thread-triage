"""
run_triage.py
-------------
Entry point for the Gmail Triage Agent. This is the script that Windows
Task Scheduler calls on every scheduled run.

It does three things only:
  1. Adds src/ to the Python path so the agent module can be imported.
  2. Runs the triage agent.
  3. Catches and logs any unexpected error so a silent failure is impossible.

Usage
-----
    python scripts/run_triage.py

    # or, to reset the state and reprocess the last 10 threads again:
    python scripts/run_triage.py --reset
"""

import sys
import logging
import argparse
from pathlib import Path

# --- Path setup (must come before any local imports) -----------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# --- Persistent log file — Task Scheduler has no visible terminal ----------
LOG_DIR  = REPO_ROOT / "results" / "gmail_triage"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),                            # visible when run manually
        logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8"), # always written to disk
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Gmail Triage Agent runner")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the last-run timestamp so the agent re-fetches the last 10 threads.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.reset:
        state_file = LOG_DIR / "last_run.txt"
        if state_file.exists():
            state_file.unlink()
            log.info("State reset — next run will fetch the last 10 threads.")
        else:
            log.info("No state file found — already in first-run mode.")
        return

    log.info("=" * 60)
    log.info("Gmail Triage Agent — starting run")
    log.info("=" * 60)

    try:
        from gmail_triage_agent import run_triage
        df = run_triage()

        if df.empty:
            log.info("Run complete — no new threads to classify.")
        else:
            summary = df["category"].value_counts().to_dict()
            log.info(f"Run complete — {len(df)} thread(s) classified: {summary}")

    except Exception:
        # Log the full traceback so you can diagnose it by reading agent.log.
        log.exception("Unhandled error during triage run — see traceback above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
