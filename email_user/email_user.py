#!/usr/bin/env python3
"""
email_user - Send notification emails via Resend API

A CLI tool for AI agents to notify users of completed tasks, routine process
results, or other important events. Sends from charlie@charliedeck.com to
charliedeck@gmail.com.

Environment Variables:
    RESEND_API_KEY: Your Resend API key (required)

Usage:
    email_user --subject "Task Complete" --body "Your report is ready"
    email_user -s "Alert" -b "Process finished" --priority high
    echo "Message body" | email_user -s "Subject"

Exit Codes:
    0: Email sent successfully
    1: Missing required environment variables or configuration error
    2: Resend API error (authentication, rate limit, server error)
    3: Invalid arguments or input validation error
    4: Network error (connection failed, timeout)
"""

import argparse
import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

# Hardcoded email addresses
SENDER_EMAIL = "charlie@charliedeck.com"
RECIPIENT_EMAIL = "charliedeck@gmail.com"

# Import handling for testability
try:
    import resend
    RESEND_AVAILABLE = True
except ImportError:
    resend = None
    RESEND_AVAILABLE = False


class ExitCode(IntEnum):
    """Exit codes for the CLI."""
    SUCCESS = 0
    CONFIG_ERROR = 1
    API_ERROR = 2
    INVALID_INPUT = 3
    NETWORK_ERROR = 4


class EmailError(Exception):
    """Base exception for email errors."""
    def __init__(self, message: str, exit_code: ExitCode):
        super().__init__(message)
        self.exit_code = exit_code


class ConfigError(EmailError):
    """Configuration or environment error."""
    def __init__(self, message: str):
        super().__init__(message, ExitCode.CONFIG_ERROR)


class APIError(EmailError):
    """Resend API error."""
    def __init__(self, message: str):
        super().__init__(message, ExitCode.API_ERROR)


class InputError(EmailError):
    """Input validation error."""
    def __init__(self, message: str):
        super().__init__(message, ExitCode.INVALID_INPUT)


class NetworkError(EmailError):
    """Network connectivity error."""
    def __init__(self, message: str):
        super().__init__(message, ExitCode.NETWORK_ERROR)


@dataclass
class EmailResult:
    """Result of sending an email."""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None


def get_api_key() -> str:
    """
    Get the Resend API key from environment.

    Returns:
        The API key string

    Raises:
        ConfigError: If RESEND_API_KEY is not set
    """
    import os
    api_key = os.environ.get("RESEND_API_KEY")

    if not api_key:
        raise ConfigError(
            "Missing RESEND_API_KEY environment variable.\n"
            "Get your API key from https://resend.com/api-keys"
        )

    if not api_key.startswith("re_"):
        raise ConfigError(
            "Invalid RESEND_API_KEY format. Keys should start with 're_'"
        )

    return api_key


def validate_input(subject: str, body: str) -> None:
    """
    Validate email input parameters.

    Args:
        subject: Email subject line
        body: Email body content

    Raises:
        InputError: If validation fails
    """
    if not subject or not subject.strip():
        raise InputError("Email subject cannot be empty")

    if not body or not body.strip():
        raise InputError("Email body cannot be empty")

    # Reasonable limits
    if len(subject) > 998:  # RFC 5322 line length limit
        raise InputError("Email subject exceeds maximum length (998 characters)")

    if len(body) > 10_000_000:  # 10MB limit
        raise InputError("Email body exceeds maximum size (10MB)")


def send_email(
    api_key: str,
    subject: str,
    body: str,
    html: bool = False,
    priority: str = "normal",
    tags: Optional[list] = None
) -> EmailResult:
    """
    Send an email via Resend API.

    Args:
        api_key: Resend API key
        subject: Email subject line
        body: Email body content
        html: If True, send as HTML content; otherwise plain text
        priority: Email priority (low, normal, high)
        tags: Optional list of tags for categorization

    Returns:
        EmailResult with success status and message_id or error

    Raises:
        APIError: On Resend API errors
        NetworkError: On connection failures
    """
    import os

    # Configure the API key
    resend.api_key = api_key

    # Build parameters
    params: dict = {
        "from": f"Charlie Deck <{SENDER_EMAIL}>",
        "to": [RECIPIENT_EMAIL],
        "subject": subject,
    }

    # Set content type
    if html:
        params["html"] = body
    else:
        params["text"] = body

    # Add priority headers
    headers = {}
    if priority == "high":
        headers["X-Priority"] = "1"
        headers["Importance"] = "high"
    elif priority == "low":
        headers["X-Priority"] = "5"
        headers["Importance"] = "low"

    if headers:
        params["headers"] = headers

    # Add tags if provided
    if tags:
        params["tags"] = [{"name": "category", "value": t} for t in tags]

    try:
        response = resend.Emails.send(params)

        # Extract message ID from response
        message_id = None
        if hasattr(response, 'id'):
            message_id = response.id
        elif isinstance(response, dict) and 'id' in response:
            message_id = response['id']

        return EmailResult(success=True, message_id=message_id)

    except resend.exceptions.ResendError as e:
        # Handle specific Resend errors
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            raise APIError(f"Authentication failed: Invalid API key")
        elif "403" in error_msg or "forbidden" in error_msg.lower():
            raise APIError(f"Permission denied: {error_msg}")
        elif "429" in error_msg or "rate" in error_msg.lower():
            raise APIError(f"Rate limit exceeded. Try again later.")
        elif "5" in error_msg[:1] if error_msg else False:
            raise APIError(f"Resend server error: {error_msg}")
        else:
            raise APIError(f"Resend API error: {error_msg}")

    except (ConnectionError, TimeoutError, OSError) as e:
        raise NetworkError(f"Network error: {e}")

    except Exception as e:
        # Catch-all for unexpected errors
        error_type = type(e).__name__
        raise APIError(f"Unexpected error ({error_type}): {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="email_user",
        description="Send notification emails via Resend API. "
                    f"Sends from {SENDER_EMAIL} to {RECIPIENT_EMAIL}.",
        epilog=f"""
Examples:
  %(prog)s -s "Build Complete" -b "All tests passed"
  %(prog)s --subject "Report Ready" --body "$(cat report.txt)"
  %(prog)s -s "Alert" -b "<h1>Warning</h1>" --html
  echo "Process finished" | %(prog)s -s "Status Update"
  %(prog)s -s "Deploy" -b "Done" --tag deploy --tag production

Sender:    {SENDER_EMAIL}
Recipient: {RECIPIENT_EMAIL}

Environment Variables:
  RESEND_API_KEY  Your Resend API key (get from https://resend.com/api-keys)

Exit Codes:
  0  Success
  1  Configuration error (missing API key)
  2  API error (auth, rate limit, server error)
  3  Invalid input (bad arguments)
  4  Network error (connection failed)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-s", "--subject",
        required=True,
        help="Email subject line"
    )

    parser.add_argument(
        "-b", "--body",
        help="Email body content. If not provided, reads from stdin"
    )

    parser.add_argument(
        "--html",
        action="store_true",
        help="Send body as HTML content instead of plain text"
    )

    parser.add_argument(
        "--priority",
        choices=["low", "normal", "high"],
        default="normal",
        help="Email priority level (default: normal)"
    )

    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        metavar="TAG",
        help="Add a tag for categorization (can be repeated)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print email details without sending"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output on success (exit code still indicates status)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output including API response"
    )

    return parser


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point.

    Args:
        argv: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code indicating success or failure type
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Check for resend package
    if not RESEND_AVAILABLE:
        print("Error: resend package not installed. Run: pip install resend", file=sys.stderr)
        return ExitCode.CONFIG_ERROR

    # Get body from argument or stdin
    body = args.body
    if body is None:
        if sys.stdin.isatty():
            print("Error: --body required or pipe content to stdin", file=sys.stderr)
            return ExitCode.INVALID_INPUT
        body = sys.stdin.read()

    try:
        # Validate inputs
        validate_input(args.subject, body)

        # Get API key
        api_key = get_api_key()

        # Dry run mode
        if args.dry_run:
            print("=== DRY RUN ===")
            print(f"To: {RECIPIENT_EMAIL}")
            print(f"From: {SENDER_EMAIL}")
            print(f"Subject: {args.subject}")
            print(f"Priority: {args.priority}")
            print(f"Content-Type: {'text/html' if args.html else 'text/plain'}")
            if args.tags:
                print(f"Tags: {', '.join(args.tags)}")
            print(f"Body:\n{body}")
            return ExitCode.SUCCESS

        # Send the email
        result = send_email(
            api_key=api_key,
            subject=args.subject,
            body=body,
            html=args.html,
            priority=args.priority,
            tags=args.tags
        )

        # Output based on verbosity
        if not args.quiet:
            print(f"Email sent successfully to {RECIPIENT_EMAIL}")
            if args.verbose and result.message_id:
                print(f"Message ID: {result.message_id}")

        return ExitCode.SUCCESS

    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return e.exit_code

    except InputError as e:
        print(f"Input error: {e}", file=sys.stderr)
        return e.exit_code

    except APIError as e:
        print(f"API error: {e}", file=sys.stderr)
        return e.exit_code

    except NetworkError as e:
        print(f"Network error: {e}", file=sys.stderr)
        return e.exit_code

    except KeyboardInterrupt:
        print("\nAborted by user", file=sys.stderr)
        return ExitCode.INVALID_INPUT


if __name__ == "__main__":
    sys.exit(main())
