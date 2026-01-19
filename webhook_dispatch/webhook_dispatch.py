#!/usr/bin/env python3
"""
webhook_dispatch - A versatile CLI tool for sending HTTP webhook requests.

This tool provides a simple yet powerful interface for dispatching HTTP requests
to webhook endpoints with support for various authentication methods, retry logic,
and multiple content types.

Exit Codes:
    0 - Success (HTTP 2xx response)
    1 - Client error (HTTP 4xx response)
    2 - Server error (HTTP 5xx response)
    3 - Connection error (network issues, DNS failure, timeout)
    4 - Invalid arguments (bad CLI input)

Environment Variables:
    WEBHOOK_AUTH_TOKEN - Default Bearer token for authentication (optional)

Examples:
    # Simple GET request
    webhook_dispatch GET https://api.example.com/status

    # POST with JSON body
    webhook_dispatch POST https://api.example.com/events --json '{"event": "done"}'

    # POST with JSON from file
    webhook_dispatch POST https://api.example.com/events --json @payload.json

    # POST with form data
    webhook_dispatch POST https://hooks.zapier.com/xxx --form "key=value" --form "other=data"

    # PUT with custom headers
    webhook_dispatch PUT https://api.example.com/resource/1 --json '{"status": "active"}' \\
        -H "X-Custom-Header: value"

    # Request with Bearer auth
    webhook_dispatch GET https://api.example.com/secure -H "Authorization: Bearer $TOKEN"

    # Request with Basic auth
    webhook_dispatch GET https://api.example.com/secure --auth-basic "user:password"

    # Request with retry on failure
    webhook_dispatch POST https://api.example.com/webhook --json '{"data": 1}' --retry 3

    # DELETE request with verbose output
    webhook_dispatch DELETE https://api.example.com/resource/123 --verbose

    # Request with timeout
    webhook_dispatch GET https://slow-api.example.com/data --timeout 60
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

try:
    import requests
    from requests.exceptions import (
        ConnectionError,
        HTTPError,
        RequestException,
        Timeout,
    )
except ImportError:
    print("Error: 'requests' library is required. Install with: pip install requests", file=sys.stderr)
    sys.exit(4)


# Exit codes
EXIT_SUCCESS = 0
EXIT_CLIENT_ERROR = 1
EXIT_SERVER_ERROR = 2
EXIT_CONNECTION_ERROR = 3
EXIT_INVALID_ARGS = 4

# Supported HTTP methods
SUPPORTED_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]

# Default configuration
DEFAULT_TIMEOUT = 30
DEFAULT_RETRY_COUNT = 0
DEFAULT_RETRY_BASE_DELAY = 1.0
DEFAULT_RETRY_MAX_DELAY = 60.0


def setup_logging(verbose: bool = False, debug: bool = False) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        verbose: Enable verbose output (INFO level)
        debug: Enable debug output (DEBUG level, overrides verbose)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("webhook_dispatch")

    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    if debug:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = logging.Formatter("%(levelname)s: %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def parse_header(header_str: str) -> Tuple[str, str]:
    """Parse a header string in 'Name: Value' format.

    Args:
        header_str: Header string to parse

    Returns:
        Tuple of (header_name, header_value)

    Raises:
        ValueError: If header format is invalid
    """
    if ":" not in header_str:
        raise ValueError(f"Invalid header format: '{header_str}'. Expected 'Name: Value'")

    name, value = header_str.split(":", 1)
    return name.strip(), value.strip()


def parse_form_field(field_str: str) -> Tuple[str, str]:
    """Parse a form field string in 'key=value' format.

    Args:
        field_str: Form field string to parse

    Returns:
        Tuple of (key, value)

    Raises:
        ValueError: If field format is invalid
    """
    if "=" not in field_str:
        raise ValueError(f"Invalid form field format: '{field_str}'. Expected 'key=value'")

    key, value = field_str.split("=", 1)
    return key.strip(), value


def load_json_body(json_arg: str) -> Dict[str, Any]:
    """Load JSON body from string or file reference.

    Args:
        json_arg: JSON string or '@filename' reference

    Returns:
        Parsed JSON as dictionary

    Raises:
        ValueError: If JSON is invalid or file cannot be read
    """
    if json_arg.startswith("@"):
        filename = json_arg[1:]
        try:
            with open(filename, "r") as f:
                content = f.read()
        except FileNotFoundError:
            raise ValueError(f"JSON file not found: {filename}")
        except IOError as e:
            raise ValueError(f"Error reading JSON file '{filename}': {e}")
    else:
        content = json_arg

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def build_auth_header(
    auth_bearer: Optional[str] = None,
    auth_basic: Optional[str] = None,
    env_token: Optional[str] = None,
) -> Optional[str]:
    """Build Authorization header from various auth options.

    Args:
        auth_bearer: Bearer token (explicit)
        auth_basic: Basic auth in 'user:password' format
        env_token: Token from environment variable (fallback)

    Returns:
        Authorization header value or None

    Raises:
        ValueError: If basic auth format is invalid
    """
    if auth_bearer:
        return f"Bearer {auth_bearer}"

    if auth_basic:
        if ":" not in auth_basic:
            raise ValueError("Basic auth must be in 'user:password' format")
        encoded = base64.b64encode(auth_basic.encode()).decode()
        return f"Basic {encoded}"

    if env_token:
        return f"Bearer {env_token}"

    return None


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    max_delay: float = DEFAULT_RETRY_MAX_DELAY,
) -> float:
    """Calculate exponential backoff delay.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds

    Returns:
        Delay in seconds
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


def send_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    form_data: Optional[Dict[str, str]] = None,
    raw_body: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    logger: Optional[logging.Logger] = None,
) -> requests.Response:
    """Send an HTTP request.

    Args:
        method: HTTP method
        url: Target URL
        headers: Request headers
        json_body: JSON body (dict)
        form_data: Form data (dict)
        raw_body: Raw body string
        timeout: Request timeout in seconds
        logger: Logger instance

    Returns:
        Response object

    Raises:
        requests.RequestException: On request failure
    """
    if logger:
        logger.info(f"Sending {method} request to {url}")
        logger.debug(f"Headers: {headers}")
        if json_body:
            logger.debug(f"JSON body: {json.dumps(json_body, indent=2)}")
        if form_data:
            logger.debug(f"Form data: {form_data}")
        if raw_body:
            logger.debug(f"Raw body: {raw_body[:200]}...")

    kwargs: Dict[str, Any] = {
        "method": method,
        "url": url,
        "headers": headers or {},
        "timeout": timeout,
    }

    if json_body:
        kwargs["json"] = json_body
    elif form_data:
        kwargs["data"] = form_data
    elif raw_body:
        kwargs["data"] = raw_body

    response = requests.request(**kwargs)

    if logger:
        logger.info(f"Response status: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        logger.debug(f"Response body: {response.text[:500]}...")

    return response


def dispatch_webhook(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    form_data: Optional[Dict[str, str]] = None,
    raw_body: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    retry_count: int = DEFAULT_RETRY_COUNT,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[requests.Response], int]:
    """Dispatch a webhook request with retry logic.

    Args:
        method: HTTP method
        url: Target URL
        headers: Request headers
        json_body: JSON body
        form_data: Form data
        raw_body: Raw body string
        timeout: Request timeout
        retry_count: Number of retries on failure
        retry_base_delay: Base delay for exponential backoff
        retry_max_delay: Maximum delay cap
        logger: Logger instance

    Returns:
        Tuple of (response, exit_code)
    """
    attempts = retry_count + 1
    last_exception = None

    for attempt in range(attempts):
        if attempt > 0:
            delay = calculate_backoff_delay(
                attempt - 1, retry_base_delay, retry_max_delay
            )
            if logger:
                logger.info(f"Retry attempt {attempt}/{retry_count} after {delay:.1f}s delay")
            time.sleep(delay)

        try:
            response = send_request(
                method=method,
                url=url,
                headers=headers,
                json_body=json_body,
                form_data=form_data,
                raw_body=raw_body,
                timeout=timeout,
                logger=logger,
            )

            # Determine exit code based on status
            if 200 <= response.status_code < 300:
                return response, EXIT_SUCCESS
            elif 400 <= response.status_code < 500:
                # Client errors are not retryable
                if logger:
                    logger.warning(f"Client error: {response.status_code}")
                return response, EXIT_CLIENT_ERROR
            elif 500 <= response.status_code < 600:
                # Server errors might be retryable
                if logger:
                    logger.warning(f"Server error: {response.status_code}")
                if attempt < attempts - 1:
                    continue
                return response, EXIT_SERVER_ERROR
            else:
                # Unexpected status code
                return response, EXIT_SUCCESS if response.ok else EXIT_SERVER_ERROR

        except (ConnectionError, Timeout) as e:
            last_exception = e
            if logger:
                logger.warning(f"Connection error: {e}")
            if attempt < attempts - 1:
                continue
            return None, EXIT_CONNECTION_ERROR

        except RequestException as e:
            last_exception = e
            if logger:
                logger.error(f"Request error: {e}")
            return None, EXIT_CONNECTION_ERROR

    # Should not reach here, but handle edge case
    if logger and last_exception:
        logger.error(f"All retry attempts failed: {last_exception}")
    return None, EXIT_CONNECTION_ERROR


def format_response_output(
    response: Optional[requests.Response],
    show_headers: bool = False,
    output_format: str = "auto",
) -> str:
    """Format response for output.

    Args:
        response: Response object (can be None)
        show_headers: Include response headers
        output_format: Output format ('auto', 'json', 'raw')

    Returns:
        Formatted output string
    """
    if response is None:
        return ""

    output_parts = []

    if show_headers:
        output_parts.append(f"HTTP {response.status_code} {response.reason}")
        for name, value in response.headers.items():
            output_parts.append(f"{name}: {value}")
        output_parts.append("")

    # Format body
    body = response.text
    if output_format == "auto":
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                parsed = json.loads(body)
                body = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                pass
    elif output_format == "json":
        try:
            parsed = json.loads(body)
            body = json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            pass

    output_parts.append(body)
    return "\n".join(output_parts)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="webhook_dispatch",
        description="""
A versatile CLI tool for sending HTTP webhook requests.

Supports multiple HTTP methods, authentication schemes, retry logic,
and various body formats (JSON, form data, raw).
        """,
        epilog="""
Exit Codes:
  0  Success (HTTP 2xx response)
  1  Client error (HTTP 4xx response)
  2  Server error (HTTP 5xx response)
  3  Connection error (network issues, timeout)
  4  Invalid arguments

Environment Variables:
  WEBHOOK_AUTH_TOKEN  Default Bearer token for authentication

Examples:
  # Simple GET request
  %(prog)s GET https://api.example.com/status

  # POST with JSON body
  %(prog)s POST https://api.example.com/events --json '{"event": "done"}'

  # POST with JSON from file
  %(prog)s POST https://api.example.com/events --json @payload.json

  # POST with form data
  %(prog)s POST https://hooks.zapier.com/xxx --form "key=value" --form "other=data"

  # PUT with custom headers
  %(prog)s PUT https://api.example.com/resource/1 --json '{"status": "active"}' \\
      -H "X-Custom-Header: value"

  # Request with Bearer token authentication
  %(prog)s GET https://api.example.com/secure --auth-bearer "$TOKEN"

  # Request with Basic authentication
  %(prog)s GET https://api.example.com/secure --auth-basic "user:password"

  # Request with retry on failure (exponential backoff)
  %(prog)s POST https://api.example.com/webhook --json '{"data": 1}' --retry 3

  # DELETE request with verbose logging
  %(prog)s DELETE https://api.example.com/resource/123 --verbose

  # Request with custom timeout
  %(prog)s GET https://slow-api.example.com/data --timeout 60

  # Show response headers
  %(prog)s GET https://api.example.com/info --show-headers
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments
    parser.add_argument(
        "method",
        metavar="METHOD",
        choices=SUPPORTED_METHODS,
        help=f"HTTP method ({', '.join(SUPPORTED_METHODS)})",
    )
    parser.add_argument(
        "url",
        metavar="URL",
        help="Target URL for the webhook request",
    )

    # Body options (mutually exclusive)
    body_group = parser.add_mutually_exclusive_group()
    body_group.add_argument(
        "--json", "-j",
        dest="json_body",
        metavar="DATA",
        help="JSON body (string or @filename to read from file)",
    )
    body_group.add_argument(
        "--form", "-f",
        dest="form_fields",
        action="append",
        metavar="KEY=VALUE",
        help="Form field (can be specified multiple times)",
    )
    body_group.add_argument(
        "--data", "-d",
        dest="raw_body",
        metavar="DATA",
        help="Raw body data",
    )

    # Headers
    parser.add_argument(
        "--header", "-H",
        dest="headers",
        action="append",
        metavar="NAME:VALUE",
        help="HTTP header (can be specified multiple times)",
    )

    # Authentication
    auth_group = parser.add_mutually_exclusive_group()
    auth_group.add_argument(
        "--auth-bearer",
        metavar="TOKEN",
        help="Bearer token for authentication",
    )
    auth_group.add_argument(
        "--auth-basic",
        metavar="USER:PASS",
        help="Basic authentication credentials",
    )

    # Retry options
    parser.add_argument(
        "--retry",
        type=int,
        default=DEFAULT_RETRY_COUNT,
        metavar="N",
        help=f"Number of retry attempts (default: {DEFAULT_RETRY_COUNT})",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_BASE_DELAY,
        metavar="SECONDS",
        help=f"Base delay for exponential backoff (default: {DEFAULT_RETRY_BASE_DELAY})",
    )
    parser.add_argument(
        "--retry-max-delay",
        type=float,
        default=DEFAULT_RETRY_MAX_DELAY,
        metavar="SECONDS",
        help=f"Maximum delay cap for retries (default: {DEFAULT_RETRY_MAX_DELAY})",
    )

    # Request options
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )

    # Output options
    parser.add_argument(
        "--show-headers",
        action="store_true",
        help="Include response headers in output",
    )
    parser.add_argument(
        "--output-format",
        choices=["auto", "json", "raw"],
        default="auto",
        help="Output format (default: auto)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress response body output",
    )

    # Logging options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (includes timestamps)",
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code
    """
    parser = create_argument_parser()
    parsed_args = parser.parse_args(args)

    # Setup logging
    logger = setup_logging(
        verbose=parsed_args.verbose,
        debug=parsed_args.debug,
    )

    # Validate and process arguments
    try:
        # Parse headers
        headers: Dict[str, str] = {}
        if parsed_args.headers:
            for header_str in parsed_args.headers:
                name, value = parse_header(header_str)
                headers[name] = value

        # Parse JSON body
        json_body = None
        if parsed_args.json_body:
            json_body = load_json_body(parsed_args.json_body)

        # Parse form fields
        form_data = None
        if parsed_args.form_fields:
            form_data = {}
            for field_str in parsed_args.form_fields:
                key, value = parse_form_field(field_str)
                form_data[key] = value

        # Build auth header
        env_token = os.environ.get("WEBHOOK_AUTH_TOKEN")
        auth_header = build_auth_header(
            auth_bearer=parsed_args.auth_bearer,
            auth_basic=parsed_args.auth_basic,
            env_token=env_token if not parsed_args.auth_bearer and not parsed_args.auth_basic else None,
        )
        if auth_header and "Authorization" not in headers:
            headers["Authorization"] = auth_header

    except ValueError as e:
        logger.error(str(e))
        return EXIT_INVALID_ARGS

    # Send request
    response, exit_code = dispatch_webhook(
        method=parsed_args.method,
        url=parsed_args.url,
        headers=headers if headers else None,
        json_body=json_body,
        form_data=form_data,
        raw_body=parsed_args.raw_body,
        timeout=parsed_args.timeout,
        retry_count=parsed_args.retry,
        retry_base_delay=parsed_args.retry_delay,
        retry_max_delay=parsed_args.retry_max_delay,
        logger=logger,
    )

    # Output response
    if not parsed_args.quiet and response is not None:
        output = format_response_output(
            response,
            show_headers=parsed_args.show_headers,
            output_format=parsed_args.output_format,
        )
        if output:
            print(output)

    # Log final status
    if exit_code == EXIT_SUCCESS:
        logger.info("Request completed successfully")
    elif exit_code == EXIT_CLIENT_ERROR:
        logger.warning(f"Request failed with client error: {response.status_code if response is not None else 'unknown'}")
    elif exit_code == EXIT_SERVER_ERROR:
        logger.warning(f"Request failed with server error: {response.status_code if response is not None else 'unknown'}")
    elif exit_code == EXIT_CONNECTION_ERROR:
        logger.error("Request failed due to connection error")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
