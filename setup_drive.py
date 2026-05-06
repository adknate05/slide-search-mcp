"""
One-time setup: authenticates with Google Drive, uploads all slides,
and saves drive_links.json mapping filenames to Drive file IDs.

Steps:
    1. Follow the instructions printed below to get credentials.json
    2. Run: python3 setup_drive.py
    3. A browser window will open — sign in and allow access
    4. All slides are uploaded to a "Exam Slides" folder in your Drive
    5. drive_links.json is saved — mcp_server.py will use it automatically

After initial upload, re-run any time you add new slides.
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SLIDES_DIR = os.path.join(BASE_DIR, "slides")
CREDENTIALS_FILE = os.path.join(BASE_DIR, "credentials.json")
TOKEN_FILE = os.path.join(BASE_DIR, "token.json")
DRIVE_LINKS_FILE = os.path.join(BASE_DIR, "drive_links.json")
DRIVE_FOLDER_NAME = "Exam Slides"

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

SETUP_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════╗
║         Google Drive Setup — One-time credential setup           ║
╚══════════════════════════════════════════════════════════════════╝

credentials.json not found. Do this once:

1. Go to: https://console.cloud.google.com/
2. Create a new project (name it anything, e.g. "Slide Search")
3. In the left menu → APIs & Services → Library
4. Search "Google Drive API" → Enable it
5. Go to APIs & Services → Credentials
6. Click "+ Create Credentials" → OAuth client ID
7. Application type: Desktop app → Name it anything → Create
8. Click the download icon (↓) next to your new credential
9. Rename the downloaded file to: credentials.json
10. Move it to: {dest}

Then re-run: python3 setup_drive.py

(You may need to set the OAuth consent screen to "External" and add
your own Google account as a test user under the consent screen settings.)
""".format(dest=CREDENTIALS_FILE)


def get_drive_service():
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def get_or_create_folder(service, folder_name: str) -> str:
    """Return the Drive folder ID, creating it if it doesn't exist."""
    query = (
        f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        " and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        folder_id = files[0]["id"]
        print(f"Found existing Drive folder '{folder_name}' (id: {folder_id})")
        return folder_id

    folder_meta = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    folder = service.files().create(body=folder_meta, fields="id").execute()
    folder_id = folder["id"]
    print(f"Created Drive folder '{folder_name}' (id: {folder_id})")
    return folder_id


def list_existing_files(service, folder_id: str) -> dict[str, str]:
    """Return {filename: file_id} for files already in the folder."""
    existing = {}
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            existing[f["name"]] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return existing


def upload_file(service, local_path: str, filename: str, folder_id: str) -> str:
    """Upload a PDF and return its Drive file ID."""
    from googleapiclient.http import MediaFileUpload

    file_meta = {"name": filename, "parents": [folder_id]}
    media = MediaFileUpload(local_path, mimetype="application/pdf", resumable=True)
    uploaded = (
        service.files()
        .create(body=file_meta, media_body=media, fields="id")
        .execute()
    )
    return uploaded["id"]


def main():
    if not os.path.exists(CREDENTIALS_FILE):
        print(SETUP_INSTRUCTIONS)
        sys.exit(1)

    print("Authenticating with Google Drive...")
    service = get_drive_service()
    print("Authenticated.")

    folder_id = get_or_create_folder(service, DRIVE_FOLDER_NAME)
    existing = list_existing_files(service, folder_id)
    print(f"Files already in Drive folder: {len(existing)}")

    pdfs = [f for f in os.listdir(SLIDES_DIR) if f.lower().endswith(".pdf")]
    drive_links = dict(existing)  # start with what's already there

    to_upload = [f for f in pdfs if f not in existing]
    print(f"PDFs to upload: {len(to_upload)} (skipping {len(pdfs) - len(to_upload)} already uploaded)")

    for i, filename in enumerate(to_upload, 1):
        local_path = os.path.join(SLIDES_DIR, filename)
        print(f"  [{i}/{len(to_upload)}] Uploading {filename}...")
        file_id = upload_file(service, local_path, filename, folder_id)
        drive_links[filename] = file_id

    with open(DRIVE_LINKS_FILE, "w", encoding="utf-8") as f:
        json.dump(drive_links, f, indent=2)

    print(f"\nDone. {len(drive_links)} files mapped in drive_links.json")
    print(f"Drive folder: https://drive.google.com/drive/folders/{folder_id}")

    # Print a sample link so you can verify page anchors work
    sample_name, sample_id = next(iter(drive_links.items()))
    print(f"\nSample link (page 5): https://drive.google.com/file/d/{sample_id}/view#page=5")
    print("Open that link to verify the PDF viewer jumps to page 5.")


if __name__ == "__main__":
    main()
