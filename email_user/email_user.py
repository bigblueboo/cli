#!/usr/bin/env python3
"""
email_user - Send notification emails via SendGrid API

A CLI tool for AI agents to notify users of completed tasks, routine process
results, or other important events. Sends to a hardcoded recipient address.

Environment Variables:
    SENDGRID_API_KEY: Your SendGrid API key (required)
    EMAIL_RECIPIENT: Recipient email address (required)
    EMAIL_SENDER: Sender email address (required)

Usage:
    email_user --subject "Task Complete" --body "Your report is ready"
    email_user -s "Alert" -b "Process finished" --priority high
    echo "Message body" | email_user -s "Subject"

Exit Codes:
    0: Email sent successfully
    1: Missing required environment variables
    2: SendGrid API error
    3: Invalid arguments
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content, Header
except ImportError:
    print("Error: sendgrid package not installed. Run: pip install sendgrid", file=sys.stderr)
    sys.exit(1)


@dataclass
class EmailConfig:
    """Configuration for email sending."""
    api_key: str
    sender: str
    recipient: str


def get_config() -> EmailConfig:
    """
    Load configuration from environment variables.

    Returns:
        EmailConfig with validated settings

    Raises:
        SystemExit: If required environment variables are missing
    """
    api_key = os.environ.get("SENDGRID_API_KEY")
    recipient = os.environ.get("EMAIL_RECIPIENT")
    sender = os.environ.get("EMAIL_SENDER")

    missing = []
    if not api_key:
        missing.append("SENDGRID_API_KEY")
    if not recipient:
        missing.append("EMAIL_RECIPIENT")
    if not sender:
        missing.append("EMAIL_SENDER")

    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        print("\nRequired environment variables:", file=sys.stderr)
        print("  SENDGRID_API_KEY - Your SendGrid API key", file=sys.stderr)
        print("  EMAIL_RECIPIENT  - Recipient email address", file=sys.stderr)
        print("  EMAIL_SENDER     - Sender email address", file=sys.stderr)
        sys.exit(1)

    return EmailConfig(api_key=api_key, sender=sender, recipient=recipient)


def send_email(
    config: EmailConfig,
    subject: str,
    body: str,
    html: bool = False,
    priority: str = "normal"
) -> dict:
    """
    Send an email via SendGrid.

    Args:
        config: Email configuration with API key and addresses
        subject: Email subject line
        body: Email body content
        html: If True, send as HTML content; otherwise plain text
        priority: Email priority (low, normal, high)

    Returns:
        dict with status_code and message_id on success

    Raises:
        Exception: On SendGrid API errors
    """
    content_type = "text/html" if html else "text/plain"

    message = Mail(
        from_email=Email(config.sender),
        to_emails=To(config.recipient),
        subject=subject,
        plain_text_content=None if html else body,
        html_content=body if html else None
    )

    # Set priority headers
    if priority == "high":
        message.header = Header("X-Priority", "1")
        message.header = Header("X-MSMail-Priority", "High")
    elif priority == "low":
        message.header = Header("X-Priority", "5")
        message.header = Header("X-MSMail-Priority", "Low")

    sg = SendGridAPIClient(config.api_key)
    response = sg.send(message)

    return {
        "status_code": response.status_code,
        "message_id": response.headers.get("X-Message-Id", "unknown"),
        "recipient": config.recipient
    }


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="email_user",
        description="Send notification emails via SendGrid API. "
                    "Designed for AI agents to notify users of task completion.",
        epilog="""
Examples:
  %(prog)s -s "Build Complete" -b "All tests passed"
  %(prog)s --subject "Report Ready" --body "$(cat report.txt)"
  %(prog)s -s "Alert" -b "<h1>Warning</h1>" --html
  echo "Process finished" | %(prog)s -s "Status Update"

Environment Variables:
  SENDGRID_API_KEY  Your SendGrid API key
  EMAIL_RECIPIENT   Recipient email address
  EMAIL_SENDER      Sender email address
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
        "--dry-run",
        action="store_true",
        help="Print email details without sending"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output on success"
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Get body from argument or stdin
    body = args.body
    if body is None:
        if sys.stdin.isatty():
            print("Error: --body required or pipe content to stdin", file=sys.stderr)
            return 3
        body = sys.stdin.read()

    if not body.strip():
        print("Error: Email body cannot be empty", file=sys.stderr)
        return 3

    config = get_config()

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"To: {config.recipient}")
        print(f"From: {config.sender}")
        print(f"Subject: {args.subject}")
        print(f"Priority: {args.priority}")
        print(f"Content-Type: {'text/html' if args.html else 'text/plain'}")
        print(f"Body:\n{body}")
        return 0

    try:
        result = send_email(
            config=config,
            subject=args.subject,
            body=body,
            html=args.html,
            priority=args.priority
        )

        if not args.quiet:
            print(f"Email sent successfully to {result['recipient']}")
            print(f"Status: {result['status_code']}")
            print(f"Message ID: {result['message_id']}")

        return 0

    except Exception as e:
        print(f"Error sending email: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
