#!/usr/bin/env python3
"""
slack_notify - A self-documenting CLI tool for sending Slack messages.

This tool supports both Slack Webhooks (simple) and the Bot API (rich formatting).
It provides comprehensive options for sending messages to channels or users,
with support for attachments, blocks, and threading.

Environment Variables:
    SLACK_BOT_TOKEN: Bot token for Web API (starts with xoxb-)
    SLACK_WEBHOOK_URL: Incoming webhook URL for simple messages

Exit Codes:
    0 - Success
    1 - Missing credentials
    2 - API error
    3 - Invalid arguments

Examples:
    # Send a simple message to a channel using Bot API
    slack_notify -c "#deploys" -m "Build deployed successfully"

    # Send via webhook (simple, predefined channel)
    slack_notify --webhook -m "Alert: CPU usage high"

    # Reply in a thread
    slack_notify -c "#team" -m "Update on the issue" --thread-ts "1234567890.123456"

    # Send with custom username and emoji
    slack_notify -c "#alerts" -m "Server down!" --username "AlertBot" --icon-emoji ":warning:"

    # Send to a user directly (DM)
    slack_notify -u "@john.doe" -m "Hey, check this out!"

    # Send with blocks (JSON file)
    slack_notify -c "#general" --blocks-file blocks.json

    # Send with inline blocks
    slack_notify -c "#general" --blocks '[{"type":"section","text":{"type":"mrkdwn","text":"*Bold* text"}}]'

Author: CLI Tools
License: MIT
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List, Union

# Exit codes
EXIT_SUCCESS = 0
EXIT_MISSING_CREDENTIALS = 1
EXIT_API_ERROR = 2
EXIT_INVALID_ARGS = 3

# API endpoints
SLACK_API_BASE = "https://slack.com/api"
CHAT_POST_MESSAGE = f"{SLACK_API_BASE}/chat.postMessage"


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser with comprehensive help.

    Returns:
        argparse.ArgumentParser: Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        prog="slack_notify",
        description="""
Send messages to Slack channels or users via Webhooks or Bot API.

This tool supports two modes:
  1. Bot API (default): Uses SLACK_BOT_TOKEN for rich formatting and full control
  2. Webhook mode: Uses SLACK_WEBHOOK_URL for simple, predefined channel posting
        """,
        epilog="""
ENVIRONMENT VARIABLES:
  SLACK_BOT_TOKEN     Bot token (xoxb-...) for Web API access
  SLACK_WEBHOOK_URL   Incoming webhook URL for simple posting

EXIT CODES:
  0  Success
  1  Missing credentials (no token or webhook URL)
  2  API error (Slack returned an error)
  3  Invalid arguments

EXAMPLES:
  # Basic message to channel
  %(prog)s -c "#deploys" -m "Build v1.2.3 deployed"

  # Webhook mode (uses predefined channel from webhook)
  %(prog)s --webhook -m "Alert: High memory usage"

  # Thread reply
  %(prog)s -c "#incidents" -m "Update: Issue resolved" --thread-ts "1234567890.123456"

  # Custom bot appearance
  %(prog)s -c "#alerts" -m "Warning!" --username "AlertBot" --icon-emoji ":rotating_light:"

  # Direct message to user
  %(prog)s -u "U1234567890" -m "Please review PR #42"

  # Rich formatting with blocks
  %(prog)s -c "#general" --blocks '[{"type":"section","text":{"type":"mrkdwn","text":"*Hello* _world_"}}]'

  # Blocks from file
  %(prog)s -c "#updates" --blocks-file ./message_blocks.json -m "Fallback text"

For more information on Slack message formatting:
  https://api.slack.com/reference/surfaces/formatting
  https://api.slack.com/block-kit
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Target group (channel or user)
    target_group = parser.add_argument_group("Target Options")
    target_mutex = target_group.add_mutually_exclusive_group()
    target_mutex.add_argument(
        "-c", "--channel",
        metavar="CHANNEL",
        help="Channel to send message to (e.g., '#general', 'C1234567890')"
    )
    target_mutex.add_argument(
        "-u", "--user",
        metavar="USER",
        help="User to send direct message to (e.g., 'U1234567890', '@username')"
    )

    # Message content group
    content_group = parser.add_argument_group("Message Content")
    content_group.add_argument(
        "-m", "--message",
        metavar="TEXT",
        help="Message text (required for webhook, fallback text for blocks)"
    )
    content_group.add_argument(
        "--blocks",
        metavar="JSON",
        help="Block Kit blocks as JSON string"
    )
    content_group.add_argument(
        "--blocks-file",
        metavar="FILE",
        help="Path to JSON file containing Block Kit blocks"
    )
    content_group.add_argument(
        "--attachments",
        metavar="JSON",
        help="Message attachments as JSON string"
    )
    content_group.add_argument(
        "--attachments-file",
        metavar="FILE",
        help="Path to JSON file containing attachments"
    )

    # Threading options
    thread_group = parser.add_argument_group("Threading")
    thread_group.add_argument(
        "--thread-ts",
        metavar="TIMESTAMP",
        help="Thread timestamp to reply to (e.g., '1234567890.123456')"
    )
    thread_group.add_argument(
        "--reply-broadcast",
        action="store_true",
        help="Also post reply to channel (only with --thread-ts)"
    )

    # Customization options
    custom_group = parser.add_argument_group("Customization (Bot API only)")
    custom_group.add_argument(
        "--username",
        metavar="NAME",
        help="Custom username for the message"
    )
    custom_group.add_argument(
        "--icon-emoji",
        metavar="EMOJI",
        help="Custom emoji icon (e.g., ':robot_face:')"
    )
    custom_group.add_argument(
        "--icon-url",
        metavar="URL",
        help="Custom icon URL"
    )

    # Formatting options
    format_group = parser.add_argument_group("Formatting Options")
    format_group.add_argument(
        "--unfurl-links",
        action="store_true",
        default=None,
        help="Enable link unfurling"
    )
    format_group.add_argument(
        "--no-unfurl-links",
        action="store_true",
        help="Disable link unfurling"
    )
    format_group.add_argument(
        "--unfurl-media",
        action="store_true",
        default=None,
        help="Enable media unfurling"
    )
    format_group.add_argument(
        "--no-unfurl-media",
        action="store_true",
        help="Disable media unfurling"
    )
    format_group.add_argument(
        "--mrkdwn",
        action="store_true",
        default=True,
        help="Enable markdown formatting (default)"
    )
    format_group.add_argument(
        "--no-mrkdwn",
        action="store_true",
        help="Disable markdown formatting"
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode Selection")
    mode_group.add_argument(
        "--webhook",
        action="store_true",
        help="Use webhook URL instead of Bot API"
    )
    mode_group.add_argument(
        "--token",
        metavar="TOKEN",
        help="Override SLACK_BOT_TOKEN environment variable"
    )
    mode_group.add_argument(
        "--webhook-url",
        metavar="URL",
        help="Override SLACK_WEBHOOK_URL environment variable"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output including API response"
    )
    output_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors"
    )
    output_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payload without sending (for debugging)"
    )

    return parser


def load_json_file(filepath: str) -> Any:
    """
    Load and parse a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file contains invalid JSON.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_json_string(json_str: str, field_name: str) -> Any:
    """
    Parse a JSON string with helpful error messages.

    Args:
        json_str: JSON string to parse.
        field_name: Name of the field for error messages.

    Returns:
        Parsed JSON content.

    Raises:
        SystemExit: If JSON is invalid.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in --{field_name}: {e}", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)


def get_credentials(args: argparse.Namespace) -> tuple[Optional[str], Optional[str]]:
    """
    Get Slack credentials from arguments or environment.

    Args:
        args: Parsed command line arguments.

    Returns:
        Tuple of (bot_token, webhook_url).
    """
    bot_token = args.token or os.environ.get("SLACK_BOT_TOKEN")
    webhook_url = args.webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    return bot_token, webhook_url


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.

    Args:
        args: Parsed command line arguments.

    Raises:
        SystemExit: If validation fails.
    """
    # Check for message content
    has_content = args.message or args.blocks or args.blocks_file
    if not has_content:
        print("Error: Message content required. Use -m/--message, --blocks, or --blocks-file",
              file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)

    # Webhook mode validation
    if args.webhook:
        if args.channel or args.user:
            print("Warning: --channel and --user are ignored in webhook mode",
                  file=sys.stderr)
        if args.username or args.icon_emoji or args.icon_url:
            print("Warning: Customization options are ignored in webhook mode",
                  file=sys.stderr)
    else:
        # Bot API requires a target
        if not args.channel and not args.user:
            print("Error: Target required. Use -c/--channel or -u/--user (or --webhook mode)",
                  file=sys.stderr)
            sys.exit(EXIT_INVALID_ARGS)

    # Thread broadcast requires thread_ts
    if args.reply_broadcast and not args.thread_ts:
        print("Error: --reply-broadcast requires --thread-ts", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)

    # Conflicting options
    if args.verbose and args.quiet:
        print("Error: --verbose and --quiet are mutually exclusive", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build the message payload from arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Dictionary containing the message payload.
    """
    payload: Dict[str, Any] = {}

    # Channel/user target (Bot API only)
    if not args.webhook:
        if args.channel:
            # Strip # prefix if present for channel names
            channel = args.channel
            if channel.startswith('#'):
                channel = channel[1:]
            payload["channel"] = channel
        elif args.user:
            # Strip @ prefix if present for usernames
            user = args.user
            if user.startswith('@'):
                user = user[1:]
            payload["channel"] = user

    # Message text
    if args.message:
        payload["text"] = args.message

    # Blocks
    if args.blocks_file:
        try:
            payload["blocks"] = load_json_file(args.blocks_file)
        except FileNotFoundError:
            print(f"Error: Blocks file not found: {args.blocks_file}", file=sys.stderr)
            sys.exit(EXIT_INVALID_ARGS)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in blocks file: {e}", file=sys.stderr)
            sys.exit(EXIT_INVALID_ARGS)
    elif args.blocks:
        payload["blocks"] = parse_json_string(args.blocks, "blocks")

    # Attachments
    if args.attachments_file:
        try:
            payload["attachments"] = load_json_file(args.attachments_file)
        except FileNotFoundError:
            print(f"Error: Attachments file not found: {args.attachments_file}",
                  file=sys.stderr)
            sys.exit(EXIT_INVALID_ARGS)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in attachments file: {e}", file=sys.stderr)
            sys.exit(EXIT_INVALID_ARGS)
    elif args.attachments:
        payload["attachments"] = parse_json_string(args.attachments, "attachments")

    # Threading
    if args.thread_ts:
        payload["thread_ts"] = args.thread_ts
        if args.reply_broadcast:
            payload["reply_broadcast"] = True

    # Customization (Bot API only)
    if not args.webhook:
        if args.username:
            payload["username"] = args.username
        if args.icon_emoji:
            payload["icon_emoji"] = args.icon_emoji
        if args.icon_url:
            payload["icon_url"] = args.icon_url

    # Formatting options
    if args.no_unfurl_links:
        payload["unfurl_links"] = False
    elif args.unfurl_links:
        payload["unfurl_links"] = True

    if args.no_unfurl_media:
        payload["unfurl_media"] = False
    elif args.unfurl_media:
        payload["unfurl_media"] = True

    if args.no_mrkdwn:
        payload["mrkdwn"] = False

    return payload


def send_webhook_message(
    webhook_url: str,
    payload: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Send a message using an incoming webhook.

    Args:
        webhook_url: The webhook URL.
        payload: Message payload dictionary.
        verbose: Whether to print verbose output.

    Returns:
        Response dictionary with 'ok' key.

    Raises:
        SystemExit: On API error.
    """
    data = json.dumps(payload).encode('utf-8')
    headers = {
        "Content-Type": "application/json",
    }

    if verbose:
        print(f"Sending to webhook: {webhook_url[:50]}...", file=sys.stderr)

    try:
        req = urllib.request.Request(webhook_url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            response_text = response.read().decode('utf-8')

            # Webhooks return "ok" on success, not JSON
            if response_text == "ok":
                return {"ok": True}
            else:
                # Try to parse as JSON error response
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    return {"ok": False, "error": response_text}

    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else str(e)
        print(f"Error: HTTP {e.code}: {error_body}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)
    except urllib.error.URLError as e:
        print(f"Error: Network error: {e.reason}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)


def send_api_message(
    token: str,
    payload: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Send a message using the Slack Web API (chat.postMessage).

    Args:
        token: Bot token for authentication.
        payload: Message payload dictionary.
        verbose: Whether to print verbose output.

    Returns:
        API response dictionary.

    Raises:
        SystemExit: On API error.
    """
    data = json.dumps(payload).encode('utf-8')
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {token}",
    }

    if verbose:
        print(f"Sending to Slack API: {CHAT_POST_MESSAGE}", file=sys.stderr)

    try:
        req = urllib.request.Request(CHAT_POST_MESSAGE, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            response_text = response.read().decode('utf-8')
            return json.loads(response_text)

    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else str(e)
        try:
            error_json = json.loads(error_body)
            return error_json
        except json.JSONDecodeError:
            print(f"Error: HTTP {e.code}: {error_body}", file=sys.stderr)
            sys.exit(EXIT_API_ERROR)
    except urllib.error.URLError as e:
        print(f"Error: Network error: {e.reason}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)


def format_response(response: Dict[str, Any], verbose: bool = False) -> str:
    """
    Format the API response for output.

    Args:
        response: API response dictionary.
        verbose: Whether to include full response details.

    Returns:
        Formatted response string.
    """
    if verbose:
        return json.dumps(response, indent=2)

    if response.get("ok"):
        ts = response.get("ts", "")
        channel = response.get("channel", "")
        if ts and channel:
            return f"Message sent successfully (ts: {ts}, channel: {channel})"
        return "Message sent successfully"
    else:
        error = response.get("error", "unknown error")
        return f"Failed to send message: {error}"


def main() -> int:
    """
    Main entry point for the slack_notify CLI.

    Returns:
        Exit code (0=success, 1=missing creds, 2=API error, 3=invalid args).
    """
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    # Get credentials
    bot_token, webhook_url = get_credentials(args)

    # Check for required credentials
    if args.webhook:
        if not webhook_url:
            print("Error: Webhook URL required. Set SLACK_WEBHOOK_URL or use --webhook-url",
                  file=sys.stderr)
            return EXIT_MISSING_CREDENTIALS
    else:
        if not bot_token:
            print("Error: Bot token required. Set SLACK_BOT_TOKEN or use --token",
                  file=sys.stderr)
            return EXIT_MISSING_CREDENTIALS

    # Build payload
    payload = build_payload(args)

    # Dry run mode
    if args.dry_run:
        print("Dry run - payload would be:")
        print(json.dumps(payload, indent=2))
        return EXIT_SUCCESS

    # Send message
    if args.webhook:
        response = send_webhook_message(webhook_url, payload, verbose=args.verbose)
    else:
        response = send_api_message(bot_token, payload, verbose=args.verbose)

    # Handle response
    if response.get("ok"):
        if not args.quiet:
            print(format_response(response, verbose=args.verbose))
        return EXIT_SUCCESS
    else:
        error = response.get("error", "unknown error")
        print(f"Error: {error}", file=sys.stderr)

        # Provide helpful hints for common errors
        error_hints = {
            "channel_not_found": "Check that the channel exists and the bot has access",
            "not_in_channel": "The bot needs to be invited to the channel first",
            "invalid_auth": "Check your SLACK_BOT_TOKEN is valid",
            "token_revoked": "Your token has been revoked, generate a new one",
            "no_text": "Message text is required (use -m or ensure blocks have fallback text)",
            "msg_too_long": "Message exceeds 4000 character limit",
            "rate_limited": "Too many requests, wait and try again",
        }
        hint = error_hints.get(error)
        if hint:
            print(f"Hint: {hint}", file=sys.stderr)

        return EXIT_API_ERROR


if __name__ == "__main__":
    sys.exit(main())
