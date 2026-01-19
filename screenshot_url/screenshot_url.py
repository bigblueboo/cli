#!/usr/bin/env python3
"""
screenshot_url - A CLI tool for capturing screenshots of web pages using Playwright.

This tool provides flexible screenshot capabilities including full-page capture,
viewport customization, mobile device emulation, and multiple output formats.

Exit Codes:
    0 - Success
    1 - Browser error (failed to launch or connect)
    2 - Navigation error (failed to load URL)
    3 - Timeout error (page load or screenshot timed out)
    4 - Invalid arguments

Environment Variables:
    PLAYWRIGHT_BROWSER - Browser to use (chromium, firefox, webkit). Default: chromium

Examples:
    # Basic screenshot
    screenshot_url https://example.com -o page.png

    # Full page screenshot
    screenshot_url https://example.com -o full.png --full-page

    # Wait for element before capture
    screenshot_url https://app.com --wait-for ".loaded" --viewport 1920x1080

    # Mobile device emulation
    screenshot_url https://app.com -o mobile.png --device "iPhone 14"

    # PDF output
    screenshot_url https://example.com -o doc.pdf --format pdf

    # JPEG with quality setting
    screenshot_url https://example.com -o photo.jpg --format jpeg --quality 85

    # Custom viewport with timeout
    screenshot_url https://example.com -o custom.png --viewport 1440x900 --timeout 60

    # Wait for network to be idle
    screenshot_url https://example.com -o loaded.png --wait-until networkidle
"""

import argparse
import os
import sys
import textwrap
from typing import Optional, Dict, Any, Tuple

# Exit codes
EXIT_SUCCESS = 0
EXIT_BROWSER_ERROR = 1
EXIT_NAVIGATION_ERROR = 2
EXIT_TIMEOUT_ERROR = 3
EXIT_INVALID_ARGS = 4


def parse_viewport(viewport_str: str) -> Tuple[int, int]:
    """
    Parse viewport string in format WIDTHxHEIGHT.

    Args:
        viewport_str: Viewport dimensions as "WIDTHxHEIGHT" (e.g., "1920x1080")

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If format is invalid
    """
    try:
        parts = viewport_str.lower().split('x')
        if len(parts) != 2:
            raise ValueError("Invalid format")
        width = int(parts[0])
        height = int(parts[1])
        if width <= 0 or height <= 0:
            raise ValueError("Dimensions must be positive")
        return width, height
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid viewport format '{viewport_str}'. "
            "Use WIDTHxHEIGHT (e.g., 1920x1080)"
        ) from e


def get_available_devices(playwright) -> list:
    """
    Get list of available device names from Playwright.

    Args:
        playwright: Playwright instance

    Returns:
        Sorted list of device names
    """
    return sorted(playwright.devices.keys())


def list_devices():
    """Print all available device names and exit."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Error: playwright is not installed. Run: pip install playwright", file=sys.stderr)
        sys.exit(EXIT_BROWSER_ERROR)

    with sync_playwright() as p:
        devices = get_available_devices(p)
        print(f"Available devices ({len(devices)}):")
        for device in devices:
            print(f"  {device}")
    sys.exit(EXIT_SUCCESS)


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    epilog = textwrap.dedent("""
    Examples:
      Basic screenshot:
        %(prog)s https://example.com -o page.png

      Full page screenshot:
        %(prog)s https://example.com -o full.png --full-page

      Wait for element and custom viewport:
        %(prog)s https://app.com --wait-for ".loaded" --viewport 1920x1080

      Mobile device emulation (use --list-devices to see all):
        %(prog)s https://app.com -o mobile.png --device "iPhone 14"

      PDF output:
        %(prog)s https://example.com -o doc.pdf --format pdf

      JPEG with quality:
        %(prog)s https://example.com -o photo.jpg --format jpeg --quality 85

      Wait for network idle:
        %(prog)s https://example.com -o loaded.png --wait-until networkidle

    Exit Codes:
      0 - Success
      1 - Browser error (failed to launch)
      2 - Navigation error (failed to load URL)
      3 - Timeout error
      4 - Invalid arguments

    Environment Variables:
      PLAYWRIGHT_BROWSER  Browser to use: chromium, firefox, webkit (default: chromium)
    """)

    parser = argparse.ArgumentParser(
        prog='screenshot_url',
        description='Capture screenshots of web pages using Playwright.',
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        'url',
        nargs='?',
        help='URL to capture (required unless using --list-devices)'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '-o', '--output',
        metavar='FILE',
        help='Output file path (default: screenshot.png or screenshot.pdf)'
    )
    output_group.add_argument(
        '-f', '--format',
        choices=['png', 'jpeg', 'pdf'],
        help='Output format (default: inferred from output file, or png)'
    )
    output_group.add_argument(
        '-q', '--quality',
        type=int,
        metavar='N',
        help='JPEG quality 0-100 (only for jpeg format, default: 80)'
    )

    # Capture options
    capture_group = parser.add_argument_group('Capture Options')
    capture_group.add_argument(
        '--full-page',
        action='store_true',
        help='Capture full scrollable page instead of viewport'
    )
    capture_group.add_argument(
        '--viewport',
        metavar='WxH',
        help='Viewport size as WIDTHxHEIGHT (e.g., 1920x1080)'
    )
    capture_group.add_argument(
        '--device',
        metavar='NAME',
        help='Emulate mobile device (use --list-devices to see options)'
    )
    capture_group.add_argument(
        '--scale',
        choices=['css', 'device'],
        default='device',
        help='Screenshot scale: css (1:1 CSS pixels) or device (default: device)'
    )
    capture_group.add_argument(
        '--omit-background',
        action='store_true',
        help='Make background transparent (PNG only)'
    )

    # Wait options
    wait_group = parser.add_argument_group('Wait Options')
    wait_group.add_argument(
        '--wait-for',
        metavar='SELECTOR',
        help='Wait for CSS selector to appear before capture'
    )
    wait_group.add_argument(
        '--wait-until',
        choices=['load', 'domcontentloaded', 'networkidle', 'commit'],
        default='load',
        help='Wait until event before capture (default: load)'
    )
    wait_group.add_argument(
        '-t', '--timeout',
        type=int,
        default=30,
        metavar='SECONDS',
        help='Timeout in seconds (default: 30, 0 to disable)'
    )
    wait_group.add_argument(
        '--delay',
        type=float,
        default=0,
        metavar='SECONDS',
        help='Additional delay after page load before capture (default: 0)'
    )

    # PDF-specific options
    pdf_group = parser.add_argument_group('PDF Options (only with --format pdf)')
    pdf_group.add_argument(
        '--pdf-landscape',
        action='store_true',
        help='PDF in landscape orientation'
    )
    pdf_group.add_argument(
        '--pdf-format',
        choices=['Letter', 'Legal', 'Tabloid', 'Ledger', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
        default='Letter',
        help='PDF page format (default: Letter)'
    )
    pdf_group.add_argument(
        '--pdf-print-background',
        action='store_true',
        help='Print background graphics in PDF'
    )

    # Browser options
    browser_group = parser.add_argument_group('Browser Options')
    browser_group.add_argument(
        '--browser',
        choices=['chromium', 'firefox', 'webkit'],
        help='Browser to use (default: from PLAYWRIGHT_BROWSER env or chromium)'
    )
    browser_group.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Run browser in headless mode (default: True)'
    )
    browser_group.add_argument(
        '--no-headless',
        action='store_true',
        help='Run browser with visible window (for debugging)'
    )

    # Utility options
    utility_group = parser.add_argument_group('Utility Options')
    utility_group.add_argument(
        '--list-devices',
        action='store_true',
        help='List all available device names and exit'
    )
    utility_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    utility_group.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    return parser


def validate_args(args: argparse.Namespace) -> Optional[str]:
    """
    Validate parsed arguments.

    Args:
        args: Parsed arguments namespace

    Returns:
        Error message if validation fails, None otherwise
    """
    # URL is required unless listing devices
    if not args.list_devices and not args.url:
        return "URL is required"

    # Quality only valid for JPEG
    if args.quality is not None:
        if args.format and args.format != 'jpeg':
            return "--quality can only be used with --format jpeg"
        if not (0 <= args.quality <= 100):
            return "--quality must be between 0 and 100"

    # Omit background only for PNG
    if args.omit_background:
        if args.format and args.format not in ('png', None):
            return "--omit-background can only be used with PNG format"

    # Viewport validation
    if args.viewport:
        try:
            parse_viewport(args.viewport)
        except ValueError as e:
            return str(e)

    # Device and viewport are mutually exclusive
    if args.device and args.viewport:
        return "--device and --viewport cannot be used together"

    # PDF options only valid with PDF format
    if args.format != 'pdf':
        if args.pdf_landscape or args.pdf_print_background:
            if args.format is not None:  # Only warn if format explicitly set
                return "PDF options require --format pdf"

    return None


def infer_format(output_path: Optional[str], explicit_format: Optional[str]) -> str:
    """
    Infer output format from file extension or explicit format.

    Args:
        output_path: Output file path
        explicit_format: Explicitly specified format

    Returns:
        Format string: 'png', 'jpeg', or 'pdf'
    """
    if explicit_format:
        return explicit_format

    if output_path:
        ext = os.path.splitext(output_path)[1].lower()
        format_map = {
            '.png': 'png',
            '.jpg': 'jpeg',
            '.jpeg': 'jpeg',
            '.pdf': 'pdf',
        }
        if ext in format_map:
            return format_map[ext]

    return 'png'


def get_default_output(format_type: str) -> str:
    """
    Get default output filename for format.

    Args:
        format_type: Output format

    Returns:
        Default filename
    """
    extensions = {
        'png': 'screenshot.png',
        'jpeg': 'screenshot.jpg',
        'pdf': 'screenshot.pdf',
    }
    return extensions.get(format_type, 'screenshot.png')


def capture_screenshot(
    url: str,
    output: str,
    format_type: str,
    full_page: bool = False,
    viewport: Optional[Tuple[int, int]] = None,
    device: Optional[str] = None,
    wait_for: Optional[str] = None,
    wait_until: str = 'load',
    timeout: int = 30,
    delay: float = 0,
    quality: Optional[int] = None,
    scale: str = 'device',
    omit_background: bool = False,
    pdf_landscape: bool = False,
    pdf_format: str = 'Letter',
    pdf_print_background: bool = False,
    browser_name: Optional[str] = None,
    headless: bool = True,
    verbose: bool = False,
) -> int:
    """
    Capture a screenshot of a web page.

    Args:
        url: URL to capture
        output: Output file path
        format_type: Output format (png, jpeg, pdf)
        full_page: Capture full scrollable page
        viewport: Custom viewport (width, height)
        device: Device name for mobile emulation
        wait_for: CSS selector to wait for
        wait_until: Page load event to wait for
        timeout: Timeout in seconds
        delay: Additional delay after load
        quality: JPEG quality (0-100)
        scale: Screenshot scale (css or device)
        omit_background: Transparent background (PNG only)
        pdf_landscape: PDF landscape orientation
        pdf_format: PDF page format
        pdf_print_background: Print background in PDF
        browser_name: Browser to use
        headless: Run in headless mode
        verbose: Enable verbose output

    Returns:
        Exit code
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError:
        print("Error: playwright is not installed.", file=sys.stderr)
        print("Install with: pip install playwright", file=sys.stderr)
        print("Then run: playwright install", file=sys.stderr)
        return EXIT_BROWSER_ERROR

    # Determine browser
    browser_name = browser_name or os.environ.get('PLAYWRIGHT_BROWSER', 'chromium')
    if browser_name not in ('chromium', 'firefox', 'webkit'):
        print(f"Error: Invalid browser '{browser_name}'", file=sys.stderr)
        return EXIT_INVALID_ARGS

    timeout_ms = timeout * 1000 if timeout > 0 else 0

    if verbose:
        print(f"URL: {url}")
        print(f"Output: {output}")
        print(f"Format: {format_type}")
        print(f"Browser: {browser_name}")
        print(f"Headless: {headless}")

    try:
        with sync_playwright() as playwright:
            # Get browser launcher
            browser_type = getattr(playwright, browser_name)

            # Validate device if specified
            if device and device not in playwright.devices:
                print(f"Error: Unknown device '{device}'", file=sys.stderr)
                print("Use --list-devices to see available options", file=sys.stderr)
                return EXIT_INVALID_ARGS

            if verbose:
                print(f"Launching {browser_name}...")

            # Launch browser
            try:
                browser = browser_type.launch(headless=headless)
            except Exception as e:
                print(f"Error: Failed to launch browser: {e}", file=sys.stderr)
                return EXIT_BROWSER_ERROR

            # Prepare context options
            context_options: Dict[str, Any] = {}

            if device:
                # Use device emulation
                device_config = playwright.devices[device]
                context_options.update(device_config)
                if verbose:
                    print(f"Using device: {device}")
            elif viewport:
                # Use custom viewport
                context_options['viewport'] = {'width': viewport[0], 'height': viewport[1]}
                if verbose:
                    print(f"Viewport: {viewport[0]}x{viewport[1]}")

            # Create context and page
            context = browser.new_context(**context_options)
            page = context.new_page()

            if timeout_ms > 0:
                page.set_default_timeout(timeout_ms)

            # Navigate to URL
            if verbose:
                print(f"Navigating to {url}...")

            try:
                page.goto(url, wait_until=wait_until, timeout=timeout_ms if timeout_ms > 0 else None)
            except PlaywrightTimeout:
                print(f"Error: Navigation timeout after {timeout}s", file=sys.stderr)
                browser.close()
                return EXIT_TIMEOUT_ERROR
            except Exception as e:
                error_msg = str(e).lower()
                if 'timeout' in error_msg:
                    print(f"Error: Navigation timeout: {e}", file=sys.stderr)
                    browser.close()
                    return EXIT_TIMEOUT_ERROR
                print(f"Error: Navigation failed: {e}", file=sys.stderr)
                browser.close()
                return EXIT_NAVIGATION_ERROR

            # Wait for selector if specified
            if wait_for:
                if verbose:
                    print(f"Waiting for selector: {wait_for}")
                try:
                    page.wait_for_selector(wait_for, timeout=timeout_ms if timeout_ms > 0 else None)
                except PlaywrightTimeout:
                    print(f"Error: Timeout waiting for selector '{wait_for}'", file=sys.stderr)
                    browser.close()
                    return EXIT_TIMEOUT_ERROR
                except Exception as e:
                    print(f"Error: Failed waiting for selector: {e}", file=sys.stderr)
                    browser.close()
                    return EXIT_NAVIGATION_ERROR

            # Additional delay
            if delay > 0:
                if verbose:
                    print(f"Waiting {delay}s...")
                page.wait_for_timeout(int(delay * 1000))

            # Capture screenshot or PDF
            if verbose:
                print(f"Capturing {format_type}...")

            try:
                if format_type == 'pdf':
                    # PDF capture (Chromium only)
                    if browser_name != 'chromium':
                        print("Warning: PDF generation is only supported in Chromium", file=sys.stderr)

                    pdf_options = {
                        'path': output,
                        'format': pdf_format,
                        'landscape': pdf_landscape,
                        'print_background': pdf_print_background,
                    }
                    page.pdf(**pdf_options)
                else:
                    # Image screenshot
                    screenshot_options: Dict[str, Any] = {
                        'path': output,
                        'full_page': full_page,
                        'scale': scale,
                    }

                    if format_type == 'jpeg':
                        screenshot_options['type'] = 'jpeg'
                        screenshot_options['quality'] = quality if quality is not None else 80
                    else:
                        screenshot_options['type'] = 'png'
                        if omit_background:
                            screenshot_options['omit_background'] = True

                    page.screenshot(**screenshot_options)

            except PlaywrightTimeout:
                print(f"Error: Screenshot timeout after {timeout}s", file=sys.stderr)
                browser.close()
                return EXIT_TIMEOUT_ERROR
            except Exception as e:
                print(f"Error: Failed to capture screenshot: {e}", file=sys.stderr)
                browser.close()
                return EXIT_BROWSER_ERROR

            browser.close()

            if verbose:
                print(f"Saved to: {output}")

            return EXIT_SUCCESS

    except Exception as e:
        print(f"Error: Unexpected error: {e}", file=sys.stderr)
        return EXIT_BROWSER_ERROR


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point.

    Args:
        argv: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle --list-devices
    if args.list_devices:
        list_devices()
        return EXIT_SUCCESS  # list_devices exits, but for testing

    # Validate arguments
    error = validate_args(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return EXIT_INVALID_ARGS

    # Determine format and output
    format_type = infer_format(args.output, args.format)
    output = args.output or get_default_output(format_type)

    # Parse viewport if specified
    viewport = None
    if args.viewport:
        try:
            viewport = parse_viewport(args.viewport)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return EXIT_INVALID_ARGS

    # Determine headless mode
    headless = not args.no_headless

    # Capture screenshot
    return capture_screenshot(
        url=args.url,
        output=output,
        format_type=format_type,
        full_page=args.full_page,
        viewport=viewport,
        device=args.device,
        wait_for=args.wait_for,
        wait_until=args.wait_until,
        timeout=args.timeout,
        delay=args.delay,
        quality=args.quality,
        scale=args.scale,
        omit_background=args.omit_background,
        pdf_landscape=args.pdf_landscape,
        pdf_format=args.pdf_format,
        pdf_print_background=args.pdf_print_background,
        browser_name=args.browser,
        headless=headless,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    sys.exit(main())
