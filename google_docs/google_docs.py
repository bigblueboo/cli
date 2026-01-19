#!/usr/bin/env python3
"""
google_docs - Interact with Google Docs and Sheets

A CLI tool for AI agents to read, create, and modify Google Docs and Sheets.
Supports common operations like reading content, appending text, and
manipulating spreadsheet data.

Environment Variables:
    GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON file (required)

Authentication:
    Uses a Google Cloud service account. The service account must have
    appropriate permissions on the documents/sheets you want to access.
    Share documents with the service account email address.

Usage:
    google_docs doc read <doc_id>
    google_docs doc append <doc_id> "Text to append"
    google_docs doc create "Document Title"
    google_docs sheet read <sheet_id> [--range "A1:D10"]
    google_docs sheet append <sheet_id> --data '[["a","b"],["c","d"]]'
    google_docs sheet write <sheet_id> --range "A1" --data '[["value"]]'

Exit Codes:
    0: Success
    1: Authentication error
    2: Document/Sheet not found or permission denied
    3: API error
    4: Invalid arguments
"""

import argparse
import json
import os
import sys
from typing import Any, Optional

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    print("Error: Google API packages not installed.", file=sys.stderr)
    print("Run: pip install google-auth google-api-python-client", file=sys.stderr)
    sys.exit(1)


# API scopes needed for Docs and Sheets
SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file"
]


def get_credentials():
    """
    Get Google API credentials from service account.

    Returns:
        Credentials object

    Raises:
        SystemExit: If credentials file not found or invalid
    """
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if not creds_path:
        print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set", file=sys.stderr)
        print("\nSet it to the path of your service account JSON file:", file=sys.stderr)
        print("  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(creds_path):
        print(f"Error: Credentials file not found: {creds_path}", file=sys.stderr)
        sys.exit(1)

    try:
        credentials = service_account.Credentials.from_service_account_file(
            creds_path, scopes=SCOPES
        )
        return credentials
    except Exception as e:
        print(f"Error loading credentials: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# Google Docs Operations
# =============================================================================

def docs_read(doc_id: str, credentials) -> str:
    """
    Read the full text content of a Google Doc.

    Args:
        doc_id: The document ID from the URL
        credentials: Google API credentials

    Returns:
        Document text content
    """
    service = build("docs", "v1", credentials=credentials)
    document = service.documents().get(documentId=doc_id).execute()

    content = document.get("body", {}).get("content", [])
    text_parts = []

    for element in content:
        if "paragraph" in element:
            for para_element in element["paragraph"].get("elements", []):
                if "textRun" in para_element:
                    text_parts.append(para_element["textRun"].get("content", ""))

    return "".join(text_parts)


def docs_append(doc_id: str, text: str, credentials) -> dict:
    """
    Append text to the end of a Google Doc.

    Args:
        doc_id: The document ID
        text: Text to append
        credentials: Google API credentials

    Returns:
        API response
    """
    service = build("docs", "v1", credentials=credentials)

    # Get document to find the end index
    document = service.documents().get(documentId=doc_id).execute()
    end_index = document["body"]["content"][-1]["endIndex"] - 1

    requests = [
        {
            "insertText": {
                "location": {"index": end_index},
                "text": text
            }
        }
    ]

    result = service.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": requests}
    ).execute()

    return result


def docs_create(title: str, credentials, initial_content: Optional[str] = None) -> dict:
    """
    Create a new Google Doc.

    Args:
        title: Document title
        credentials: Google API credentials
        initial_content: Optional initial text content

    Returns:
        Dict with documentId and URL
    """
    service = build("docs", "v1", credentials=credentials)

    document = service.documents().create(body={"title": title}).execute()
    doc_id = document["documentId"]

    if initial_content:
        docs_append(doc_id, initial_content, credentials)

    return {
        "documentId": doc_id,
        "url": f"https://docs.google.com/document/d/{doc_id}/edit"
    }


# =============================================================================
# Google Sheets Operations
# =============================================================================

def sheets_read(
    sheet_id: str,
    credentials,
    range_notation: str = "A:ZZ"
) -> list:
    """
    Read data from a Google Sheet.

    Args:
        sheet_id: The spreadsheet ID
        credentials: Google API credentials
        range_notation: A1 notation range (e.g., "Sheet1!A1:D10")

    Returns:
        2D list of cell values
    """
    service = build("sheets", "v4", credentials=credentials)

    result = service.spreadsheets().values().get(
        spreadsheetId=sheet_id,
        range=range_notation
    ).execute()

    return result.get("values", [])


def sheets_append(sheet_id: str, data: list, credentials, range_notation: str = "A:A") -> dict:
    """
    Append rows to a Google Sheet.

    Args:
        sheet_id: The spreadsheet ID
        data: 2D list of values to append
        credentials: Google API credentials
        range_notation: Range to append to

    Returns:
        API response with updates info
    """
    service = build("sheets", "v4", credentials=credentials)

    body = {"values": data}

    result = service.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range=range_notation,
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body
    ).execute()

    return {
        "updatedRange": result.get("updates", {}).get("updatedRange"),
        "updatedRows": result.get("updates", {}).get("updatedRows"),
        "updatedCells": result.get("updates", {}).get("updatedCells")
    }


def sheets_write(
    sheet_id: str,
    data: list,
    credentials,
    range_notation: str = "A1"
) -> dict:
    """
    Write data to specific cells in a Google Sheet.

    Args:
        sheet_id: The spreadsheet ID
        data: 2D list of values to write
        credentials: Google API credentials
        range_notation: Starting cell in A1 notation

    Returns:
        API response with updates info
    """
    service = build("sheets", "v4", credentials=credentials)

    body = {"values": data}

    result = service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=range_notation,
        valueInputOption="USER_ENTERED",
        body=body
    ).execute()

    return {
        "updatedRange": result.get("updatedRange"),
        "updatedRows": result.get("updatedRows"),
        "updatedCells": result.get("updatedCells")
    }


def sheets_create(title: str, credentials, sheet_names: Optional[list] = None) -> dict:
    """
    Create a new Google Sheet.

    Args:
        title: Spreadsheet title
        credentials: Google API credentials
        sheet_names: Optional list of sheet/tab names to create

    Returns:
        Dict with spreadsheetId and URL
    """
    service = build("sheets", "v4", credentials=credentials)

    body = {"properties": {"title": title}}

    if sheet_names:
        body["sheets"] = [
            {"properties": {"title": name}} for name in sheet_names
        ]

    spreadsheet = service.spreadsheets().create(body=body).execute()

    return {
        "spreadsheetId": spreadsheet["spreadsheetId"],
        "url": spreadsheet["spreadsheetUrl"]
    }


def sheets_clear(sheet_id: str, credentials, range_notation: str = "A:ZZ") -> dict:
    """
    Clear cells in a Google Sheet.

    Args:
        sheet_id: The spreadsheet ID
        credentials: Google API credentials
        range_notation: Range to clear

    Returns:
        API response
    """
    service = build("sheets", "v4", credentials=credentials)

    result = service.spreadsheets().values().clear(
        spreadsheetId=sheet_id,
        range=range_notation
    ).execute()

    return {"clearedRange": result.get("clearedRange")}


# =============================================================================
# CLI Interface
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="google_docs",
        description="Interact with Google Docs and Sheets. "
                    "Designed for AI agents to read and write documents.",
        epilog="""
Examples:
  # Google Docs
  %(prog)s doc read 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms
  %(prog)s doc append DOC_ID "New paragraph text"
  %(prog)s doc create "My New Document"
  %(prog)s doc create "Report" --content "Initial content here"

  # Google Sheets
  %(prog)s sheet read SHEET_ID
  %(prog)s sheet read SHEET_ID --range "Sheet1!A1:D10"
  %(prog)s sheet append SHEET_ID --data '[["row1col1","row1col2"],["row2col1","row2col2"]]'
  %(prog)s sheet write SHEET_ID --range "B2" --data '[["value"]]'
  %(prog)s sheet create "My Spreadsheet"
  %(prog)s sheet clear SHEET_ID --range "A1:Z100"

Environment Variables:
  GOOGLE_APPLICATION_CREDENTIALS  Path to service account JSON file
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="service", help="Google service to use")

    # =================
    # Docs subcommands
    # =================
    doc_parser = subparsers.add_parser("doc", help="Google Docs operations")
    doc_subparsers = doc_parser.add_subparsers(dest="action", help="Action to perform")

    # doc read
    doc_read = doc_subparsers.add_parser("read", help="Read document content")
    doc_read.add_argument("doc_id", help="Document ID from the URL")

    # doc append
    doc_append = doc_subparsers.add_parser("append", help="Append text to document")
    doc_append.add_argument("doc_id", help="Document ID")
    doc_append.add_argument("text", help="Text to append")

    # doc create
    doc_create = doc_subparsers.add_parser("create", help="Create new document")
    doc_create.add_argument("title", help="Document title")
    doc_create.add_argument("--content", help="Initial content")

    # =================
    # Sheets subcommands
    # =================
    sheet_parser = subparsers.add_parser("sheet", help="Google Sheets operations")
    sheet_subparsers = sheet_parser.add_subparsers(dest="action", help="Action to perform")

    # sheet read
    sheet_read = sheet_subparsers.add_parser("read", help="Read spreadsheet data")
    sheet_read.add_argument("sheet_id", help="Spreadsheet ID from the URL")
    sheet_read.add_argument("--range", default="A:ZZ", help="A1 notation range (default: A:ZZ)")
    sheet_read.add_argument("--json", action="store_true", help="Output as JSON")

    # sheet append
    sheet_append = sheet_subparsers.add_parser("append", help="Append rows to spreadsheet")
    sheet_append.add_argument("sheet_id", help="Spreadsheet ID")
    sheet_append.add_argument("--data", required=True, help="JSON array of rows to append")
    sheet_append.add_argument("--range", default="A:A", help="Range to append to")

    # sheet write
    sheet_write = sheet_subparsers.add_parser("write", help="Write to specific cells")
    sheet_write.add_argument("sheet_id", help="Spreadsheet ID")
    sheet_write.add_argument("--range", required=True, help="Starting cell (A1 notation)")
    sheet_write.add_argument("--data", required=True, help="JSON array of data to write")

    # sheet create
    sheet_create = sheet_subparsers.add_parser("create", help="Create new spreadsheet")
    sheet_create.add_argument("title", help="Spreadsheet title")
    sheet_create.add_argument("--sheets", help="Comma-separated list of sheet names")

    # sheet clear
    sheet_clear = sheet_subparsers.add_parser("clear", help="Clear cells in spreadsheet")
    sheet_clear.add_argument("sheet_id", help="Spreadsheet ID")
    sheet_clear.add_argument("--range", default="A:ZZ", help="Range to clear")

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.service:
        parser.print_help()
        return 4

    if not args.action:
        parser.parse_args([args.service, "--help"])
        return 4

    credentials = get_credentials()

    try:
        # Google Docs commands
        if args.service == "doc":
            if args.action == "read":
                content = docs_read(args.doc_id, credentials)
                print(content)

            elif args.action == "append":
                result = docs_append(args.doc_id, args.text, credentials)
                print(f"Text appended successfully")

            elif args.action == "create":
                result = docs_create(args.title, credentials, args.content)
                print(json.dumps(result, indent=2))

        # Google Sheets commands
        elif args.service == "sheet":
            if args.action == "read":
                data = sheets_read(args.sheet_id, credentials, args.range)
                if args.json:
                    print(json.dumps(data, indent=2))
                else:
                    for row in data:
                        print("\t".join(str(cell) for cell in row))

            elif args.action == "append":
                data = json.loads(args.data)
                result = sheets_append(args.sheet_id, data, credentials, args.range)
                print(json.dumps(result, indent=2))

            elif args.action == "write":
                data = json.loads(args.data)
                result = sheets_write(args.sheet_id, data, credentials, args.range)
                print(json.dumps(result, indent=2))

            elif args.action == "create":
                sheet_names = args.sheets.split(",") if args.sheets else None
                result = sheets_create(args.title, credentials, sheet_names)
                print(json.dumps(result, indent=2))

            elif args.action == "clear":
                result = sheets_clear(args.sheet_id, credentials, args.range)
                print(json.dumps(result, indent=2))

        return 0

    except HttpError as e:
        if e.resp.status == 404:
            print(f"Error: Document/Sheet not found: {e}", file=sys.stderr)
            return 2
        elif e.resp.status == 403:
            print(f"Error: Permission denied. Share the document with your service account.", file=sys.stderr)
            return 2
        else:
            print(f"API Error: {e}", file=sys.stderr)
            return 3
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in --data argument: {e}", file=sys.stderr)
        return 4
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
