"""
gmail_auth.py
-------------
OAuth2 authentication helper for the Gmail API.

On the first run it opens a browser window so you can authorise the app.
After that the token is stored in token.json (gitignored) and refreshed
automatically — no browser needed again.

Usage
-----
    from src.gmail_auth import get_gmail_service
    service = get_gmail_service()

Requirements
------------
Place credentials.json (downloaded from Google Cloud Console) at the repo root.
See README or docs/bitacora-proyecto.md for setup instructions.
"""

from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# We only need read + label access — no send, no delete.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/gmail.modify",
]

REPO_ROOT        = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = REPO_ROOT / "credentials.json"
TOKEN_FILE       = REPO_ROOT / "token.json"


def get_gmail_service():
    """Return an authenticated Gmail API service object.

    Reads token.json if it exists and is valid. Refreshes automatically
    when it expires. Opens a browser for the first-time authorisation flow
    only if no valid token is found.
    """
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    # Refresh expired token silently, or start the browser flow if none exists.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDENTIALS_FILE}.\n"
                    "Download it from Google Cloud Console → APIs & Services → "
                    "Credentials → your OAuth 2.0 Client ID → Download JSON."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as fh:
            fh.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)
