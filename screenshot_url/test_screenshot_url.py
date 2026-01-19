#!/usr/bin/env python3
"""
Comprehensive tests for screenshot_url CLI tool.

Uses mocking to test all functionality without requiring actual browser instances.
"""

import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from io import StringIO

# Import the module under test
import screenshot_url as su


class TestParseViewport:
    """Tests for viewport string parsing."""

    def test_valid_viewport_standard(self):
        """Test parsing standard viewport format."""
        assert su.parse_viewport('1920x1080') == (1920, 1080)

    def test_valid_viewport_lowercase(self):
        """Test parsing viewport with lowercase x."""
        assert su.parse_viewport('1280x720') == (1280, 720)

    def test_valid_viewport_uppercase(self):
        """Test parsing viewport with uppercase X."""
        assert su.parse_viewport('1440X900') == (1440, 900)

    def test_valid_viewport_small(self):
        """Test parsing small viewport."""
        assert su.parse_viewport('320x480') == (320, 480)

    def test_invalid_viewport_no_separator(self):
        """Test parsing viewport without separator raises error."""
        with pytest.raises(ValueError) as exc_info:
            su.parse_viewport('1920-1080')
        assert 'Invalid viewport format' in str(exc_info.value)

    def test_invalid_viewport_too_many_parts(self):
        """Test parsing viewport with too many parts raises error."""
        with pytest.raises(ValueError) as exc_info:
            su.parse_viewport('1920x1080x100')
        assert 'Invalid viewport format' in str(exc_info.value)

    def test_invalid_viewport_non_numeric(self):
        """Test parsing viewport with non-numeric values raises error."""
        with pytest.raises(ValueError) as exc_info:
            su.parse_viewport('widexhigh')
        assert 'Invalid viewport format' in str(exc_info.value)

    def test_invalid_viewport_zero_width(self):
        """Test parsing viewport with zero width raises error."""
        with pytest.raises(ValueError) as exc_info:
            su.parse_viewport('0x1080')
        assert 'Invalid viewport format' in str(exc_info.value)

    def test_invalid_viewport_negative(self):
        """Test parsing viewport with negative values raises error."""
        with pytest.raises(ValueError) as exc_info:
            su.parse_viewport('-100x200')
        assert 'Invalid viewport format' in str(exc_info.value)

    def test_invalid_viewport_empty(self):
        """Test parsing empty viewport raises error."""
        with pytest.raises(ValueError) as exc_info:
            su.parse_viewport('')
        assert 'Invalid viewport format' in str(exc_info.value)


class TestInferFormat:
    """Tests for format inference from file extension."""

    def test_explicit_format_png(self):
        """Test explicit PNG format takes precedence."""
        assert su.infer_format('file.jpg', 'png') == 'png'

    def test_explicit_format_jpeg(self):
        """Test explicit JPEG format takes precedence."""
        assert su.infer_format('file.png', 'jpeg') == 'jpeg'

    def test_explicit_format_pdf(self):
        """Test explicit PDF format takes precedence."""
        assert su.infer_format('file.png', 'pdf') == 'pdf'

    def test_infer_from_png_extension(self):
        """Test inference from .png extension."""
        assert su.infer_format('screenshot.png', None) == 'png'

    def test_infer_from_jpg_extension(self):
        """Test inference from .jpg extension."""
        assert su.infer_format('photo.jpg', None) == 'jpeg'

    def test_infer_from_jpeg_extension(self):
        """Test inference from .jpeg extension."""
        assert su.infer_format('image.jpeg', None) == 'jpeg'

    def test_infer_from_pdf_extension(self):
        """Test inference from .pdf extension."""
        assert su.infer_format('document.pdf', None) == 'pdf'

    def test_infer_from_uppercase_extension(self):
        """Test inference from uppercase extension."""
        assert su.infer_format('IMAGE.PNG', None) == 'png'

    def test_infer_default_no_extension(self):
        """Test default format when no extension."""
        assert su.infer_format('screenshot', None) == 'png'

    def test_infer_default_unknown_extension(self):
        """Test default format for unknown extension."""
        assert su.infer_format('file.bmp', None) == 'png'

    def test_infer_default_no_path(self):
        """Test default format when no path provided."""
        assert su.infer_format(None, None) == 'png'


class TestGetDefaultOutput:
    """Tests for default output filename generation."""

    def test_default_output_png(self):
        """Test default output for PNG format."""
        assert su.get_default_output('png') == 'screenshot.png'

    def test_default_output_jpeg(self):
        """Test default output for JPEG format."""
        assert su.get_default_output('jpeg') == 'screenshot.jpg'

    def test_default_output_pdf(self):
        """Test default output for PDF format."""
        assert su.get_default_output('pdf') == 'screenshot.pdf'

    def test_default_output_unknown(self):
        """Test default output for unknown format."""
        assert su.get_default_output('unknown') == 'screenshot.png'


class TestValidateArgs:
    """Tests for argument validation."""

    def test_valid_basic_args(self):
        """Test validation passes for basic valid args."""
        args = Mock()
        args.list_devices = False
        args.url = 'https://example.com'
        args.quality = None
        args.format = None
        args.omit_background = False
        args.viewport = None
        args.device = None
        args.pdf_landscape = False
        args.pdf_print_background = False
        assert su.validate_args(args) is None

    def test_missing_url(self):
        """Test validation fails when URL missing."""
        args = Mock()
        args.list_devices = False
        args.url = None
        error = su.validate_args(args)
        assert 'URL is required' in error

    def test_url_not_required_with_list_devices(self):
        """Test URL not required when listing devices."""
        args = Mock()
        args.list_devices = True
        args.url = None
        args.quality = None
        args.format = None
        args.omit_background = False
        args.viewport = None
        args.device = None
        args.pdf_landscape = False
        args.pdf_print_background = False
        assert su.validate_args(args) is None

    def test_quality_with_non_jpeg(self):
        """Test quality fails with non-JPEG format."""
        args = Mock()
        args.list_devices = False
        args.url = 'https://example.com'
        args.quality = 80
        args.format = 'png'
        args.omit_background = False
        args.viewport = None
        args.device = None
        args.pdf_landscape = False
        args.pdf_print_background = False
        error = su.validate_args(args)
        assert 'quality' in error.lower() and 'jpeg' in error.lower()

    def test_quality_out_of_range_high(self):
        """Test quality fails when over 100."""
        args = Mock()
        args.list_devices = False
        args.url = 'https://example.com'
        args.quality = 150
        args.format = 'jpeg'
        args.omit_background = False
        args.viewport = None
        args.device = None
        args.pdf_landscape = False
        args.pdf_print_background = False
        error = su.validate_args(args)
        assert 'between 0 and 100' in error

    def test_quality_out_of_range_negative(self):
        """Test quality fails when negative."""
        args = Mock()
        args.list_devices = False
        args.url = 'https://example.com'
        args.quality = -10
        args.format = 'jpeg'
        args.omit_background = False
        args.viewport = None
        args.device = None
        args.pdf_landscape = False
        args.pdf_print_background = False
        error = su.validate_args(args)
        assert 'between 0 and 100' in error

    def test_omit_background_with_jpeg(self):
        """Test omit_background fails with JPEG."""
        args = Mock()
        args.list_devices = False
        args.url = 'https://example.com'
        args.quality = None
        args.format = 'jpeg'
        args.omit_background = True
        args.viewport = None
        args.device = None
        args.pdf_landscape = False
        args.pdf_print_background = False
        error = su.validate_args(args)
        assert 'omit-background' in error.lower()

    def test_invalid_viewport_format(self):
        """Test validation catches invalid viewport format."""
        args = Mock()
        args.list_devices = False
        args.url = 'https://example.com'
        args.quality = None
        args.format = None
        args.omit_background = False
        args.viewport = 'invalid'
        args.device = None
        args.pdf_landscape = False
        args.pdf_print_background = False
        error = su.validate_args(args)
        assert 'viewport' in error.lower()

    def test_device_and_viewport_conflict(self):
        """Test device and viewport cannot be used together."""
        args = Mock()
        args.list_devices = False
        args.url = 'https://example.com'
        args.quality = None
        args.format = None
        args.omit_background = False
        args.viewport = '1920x1080'
        args.device = 'iPhone 13'
        args.pdf_landscape = False
        args.pdf_print_background = False
        error = su.validate_args(args)
        assert 'cannot be used together' in error


class TestCreateParser:
    """Tests for argument parser configuration."""

    def test_parser_creation(self):
        """Test parser is created successfully."""
        parser = su.create_parser()
        assert parser is not None
        assert parser.prog == 'screenshot_url'

    def test_parser_help_contains_examples(self):
        """Test parser help contains usage examples."""
        parser = su.create_parser()
        help_text = parser.format_help()
        assert 'example.com' in help_text
        assert 'full-page' in help_text
        assert 'device' in help_text

    def test_parser_basic_url(self):
        """Test parser handles basic URL."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com'])
        assert args.url == 'https://example.com'

    def test_parser_output_short(self):
        """Test parser handles -o option."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '-o', 'out.png'])
        assert args.output == 'out.png'

    def test_parser_output_long(self):
        """Test parser handles --output option."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '--output', 'out.png'])
        assert args.output == 'out.png'

    def test_parser_format(self):
        """Test parser handles --format option."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '--format', 'jpeg'])
        assert args.format == 'jpeg'

    def test_parser_quality(self):
        """Test parser handles --quality option."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '-q', '85'])
        assert args.quality == 85

    def test_parser_full_page(self):
        """Test parser handles --full-page flag."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '--full-page'])
        assert args.full_page is True

    def test_parser_viewport(self):
        """Test parser handles --viewport option."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '--viewport', '1920x1080'])
        assert args.viewport == '1920x1080'

    def test_parser_device(self):
        """Test parser handles --device option."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '--device', 'iPhone 14'])
        assert args.device == 'iPhone 14'

    def test_parser_wait_for(self):
        """Test parser handles --wait-for option."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '--wait-for', '.loaded'])
        assert args.wait_for == '.loaded'

    def test_parser_wait_until(self):
        """Test parser handles --wait-until option."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '--wait-until', 'networkidle'])
        assert args.wait_until == 'networkidle'

    def test_parser_timeout(self):
        """Test parser handles --timeout option."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '-t', '60'])
        assert args.timeout == 60

    def test_parser_delay(self):
        """Test parser handles --delay option."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '--delay', '2.5'])
        assert args.delay == 2.5

    def test_parser_pdf_options(self):
        """Test parser handles PDF options."""
        parser = su.create_parser()
        args = parser.parse_args([
            'https://example.com',
            '--format', 'pdf',
            '--pdf-landscape',
            '--pdf-format', 'A4',
            '--pdf-print-background'
        ])
        assert args.format == 'pdf'
        assert args.pdf_landscape is True
        assert args.pdf_format == 'A4'
        assert args.pdf_print_background is True

    def test_parser_browser(self):
        """Test parser handles --browser option."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '--browser', 'firefox'])
        assert args.browser == 'firefox'

    def test_parser_verbose(self):
        """Test parser handles --verbose flag."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '-v'])
        assert args.verbose is True

    def test_parser_list_devices(self):
        """Test parser handles --list-devices flag."""
        parser = su.create_parser()
        args = parser.parse_args(['--list-devices'])
        assert args.list_devices is True

    def test_parser_no_headless(self):
        """Test parser handles --no-headless flag."""
        parser = su.create_parser()
        args = parser.parse_args(['https://example.com', '--no-headless'])
        assert args.no_headless is True


def create_playwright_mock():
    """Create a standard playwright mock for testing."""
    mock_page = Mock()
    mock_context = Mock()
    mock_context.new_page.return_value = mock_page
    mock_browser = Mock()
    mock_browser.new_context.return_value = mock_context
    mock_browser_type = Mock()
    mock_browser_type.launch.return_value = mock_browser
    mock_playwright = Mock()
    mock_playwright.chromium = mock_browser_type
    mock_playwright.firefox = mock_browser_type
    mock_playwright.webkit = mock_browser_type
    mock_playwright.devices = {
        'iPhone 13': {'viewport': {'width': 390, 'height': 844}},
        'iPhone 14': {
            'viewport': {'width': 390, 'height': 844},
            'user_agent': 'Mozilla/5.0 iPhone',
            'device_scale_factor': 3,
            'is_mobile': True,
            'has_touch': True,
        },
        'iPad Pro': {'viewport': {'width': 1024, 'height': 1366}},
        'Pixel 5': {'viewport': {'width': 393, 'height': 851}},
    }
    return mock_playwright, mock_browser, mock_page, mock_browser_type


class TestCaptureScreenshotMocked:
    """Tests for capture_screenshot function with mocked Playwright."""

    @patch('playwright.sync_api.sync_playwright')
    def test_basic_screenshot(self, mock_sync_playwright):
        """Test basic screenshot capture."""
        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
        )

        assert result == su.EXIT_SUCCESS
        mock_page.goto.assert_called_once()
        mock_page.screenshot.assert_called_once()
        mock_browser.close.assert_called_once()

    @patch('playwright.sync_api.sync_playwright')
    def test_full_page_screenshot(self, mock_sync_playwright):
        """Test full page screenshot option."""
        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='full.png',
            format_type='png',
            full_page=True,
        )

        assert result == su.EXIT_SUCCESS
        screenshot_call = mock_page.screenshot.call_args
        assert screenshot_call[1]['full_page'] is True

    @patch('playwright.sync_api.sync_playwright')
    def test_jpeg_with_quality(self, mock_sync_playwright):
        """Test JPEG screenshot with quality setting."""
        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='photo.jpg',
            format_type='jpeg',
            quality=85,
        )

        assert result == su.EXIT_SUCCESS
        screenshot_call = mock_page.screenshot.call_args
        assert screenshot_call[1]['type'] == 'jpeg'
        assert screenshot_call[1]['quality'] == 85

    @patch('playwright.sync_api.sync_playwright')
    def test_pdf_capture(self, mock_sync_playwright):
        """Test PDF capture."""
        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='doc.pdf',
            format_type='pdf',
            pdf_landscape=True,
            pdf_format='A4',
            pdf_print_background=True,
        )

        assert result == su.EXIT_SUCCESS
        pdf_call = mock_page.pdf.call_args
        assert pdf_call[1]['path'] == 'doc.pdf'
        assert pdf_call[1]['landscape'] is True
        assert pdf_call[1]['format'] == 'A4'
        assert pdf_call[1]['print_background'] is True

    @patch('playwright.sync_api.sync_playwright')
    def test_custom_viewport(self, mock_sync_playwright):
        """Test custom viewport setting."""
        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='custom.png',
            format_type='png',
            viewport=(1920, 1080),
        )

        assert result == su.EXIT_SUCCESS
        context_call = mock_browser.new_context.call_args
        assert context_call[1]['viewport'] == {'width': 1920, 'height': 1080}

    @patch('playwright.sync_api.sync_playwright')
    def test_device_emulation(self, mock_sync_playwright):
        """Test mobile device emulation."""
        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='mobile.png',
            format_type='png',
            device='iPhone 14',
        )

        assert result == su.EXIT_SUCCESS
        context_call = mock_browser.new_context.call_args
        # Device config should be unpacked into context
        assert context_call[1]['viewport'] == {'width': 390, 'height': 844}

    @patch('playwright.sync_api.sync_playwright')
    def test_wait_for_selector(self, mock_sync_playwright):
        """Test wait for CSS selector."""
        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
            wait_for='.loaded',
        )

        assert result == su.EXIT_SUCCESS
        mock_page.wait_for_selector.assert_called_once()
        selector_call = mock_page.wait_for_selector.call_args
        assert selector_call[0][0] == '.loaded'

    @patch('playwright.sync_api.sync_playwright')
    def test_wait_until_networkidle(self, mock_sync_playwright):
        """Test wait until network idle."""
        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
            wait_until='networkidle',
        )

        assert result == su.EXIT_SUCCESS
        goto_call = mock_page.goto.call_args
        assert goto_call[1]['wait_until'] == 'networkidle'

    @patch('playwright.sync_api.sync_playwright')
    def test_delay_option(self, mock_sync_playwright):
        """Test additional delay after load."""
        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
            delay=2.5,
        )

        assert result == su.EXIT_SUCCESS
        mock_page.wait_for_timeout.assert_called_once_with(2500)

    @patch('playwright.sync_api.sync_playwright')
    def test_firefox_browser(self, mock_sync_playwright):
        """Test Firefox browser selection."""
        mock_playwright, mock_browser, mock_page, mock_browser_type = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
            browser_name='firefox',
        )

        assert result == su.EXIT_SUCCESS
        mock_browser_type.launch.assert_called_once()

    @patch('playwright.sync_api.sync_playwright')
    def test_webkit_browser(self, mock_sync_playwright):
        """Test WebKit browser selection."""
        mock_playwright, mock_browser, mock_page, mock_browser_type = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
            browser_name='webkit',
        )

        assert result == su.EXIT_SUCCESS
        mock_browser_type.launch.assert_called_once()

    @patch('playwright.sync_api.sync_playwright')
    def test_omit_background(self, mock_sync_playwright):
        """Test transparent background option."""
        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
            omit_background=True,
        )

        assert result == su.EXIT_SUCCESS
        screenshot_call = mock_page.screenshot.call_args
        assert screenshot_call[1]['omit_background'] is True

    def test_invalid_browser(self):
        """Test invalid browser name returns error."""
        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
            browser_name='invalid',
        )

        assert result == su.EXIT_INVALID_ARGS

    @patch('playwright.sync_api.sync_playwright')
    def test_unknown_device(self, mock_sync_playwright):
        """Test unknown device returns error."""
        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
            device='Unknown Device',
        )

        assert result == su.EXIT_INVALID_ARGS
        # Browser is not launched before device validation, so close is not called


class TestCaptureScreenshotErrors:
    """Tests for error handling in capture_screenshot."""

    @patch('playwright.sync_api.sync_playwright')
    def test_browser_launch_error(self, mock_sync_playwright):
        """Test browser launch failure."""
        mock_playwright, _, _, _ = create_playwright_mock()
        mock_playwright.chromium.launch.side_effect = Exception("Failed to launch")
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
        )

        assert result == su.EXIT_BROWSER_ERROR

    @patch('playwright.sync_api.sync_playwright')
    def test_navigation_timeout(self, mock_sync_playwright):
        """Test navigation timeout error."""
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_page.goto.side_effect = PlaywrightTimeout("Timeout")
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
        )

        assert result == su.EXIT_TIMEOUT_ERROR
        mock_browser.close.assert_called_once()

    @patch('playwright.sync_api.sync_playwright')
    def test_navigation_error(self, mock_sync_playwright):
        """Test navigation failure."""
        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_page.goto.side_effect = Exception("Navigation failed")
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
        )

        assert result == su.EXIT_NAVIGATION_ERROR
        mock_browser.close.assert_called_once()

    @patch('playwright.sync_api.sync_playwright')
    def test_wait_for_selector_timeout(self, mock_sync_playwright):
        """Test wait for selector timeout."""
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_page.wait_for_selector.side_effect = PlaywrightTimeout("Timeout")
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
            wait_for='.element',
        )

        assert result == su.EXIT_TIMEOUT_ERROR
        mock_browser.close.assert_called_once()

    @patch('playwright.sync_api.sync_playwright')
    def test_screenshot_error(self, mock_sync_playwright):
        """Test screenshot capture failure."""
        mock_playwright, mock_browser, mock_page, _ = create_playwright_mock()
        mock_page.screenshot.side_effect = Exception("Screenshot failed")
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
        )

        assert result == su.EXIT_BROWSER_ERROR
        mock_browser.close.assert_called_once()


class TestMain:
    """Tests for main entry point."""

    @patch('screenshot_url.capture_screenshot')
    def test_main_basic(self, mock_capture):
        """Test main with basic arguments."""
        mock_capture.return_value = su.EXIT_SUCCESS

        result = su.main(['https://example.com', '-o', 'test.png'])

        assert result == su.EXIT_SUCCESS
        mock_capture.assert_called_once()

    @patch('screenshot_url.capture_screenshot')
    def test_main_all_options(self, mock_capture):
        """Test main with all options."""
        mock_capture.return_value = su.EXIT_SUCCESS

        result = su.main([
            'https://example.com',
            '-o', 'output.png',
            '--format', 'png',
            '--full-page',
            '--viewport', '1920x1080',
            '--wait-for', '.ready',
            '--wait-until', 'networkidle',
            '-t', '60',
            '--delay', '2',
            '-v',
        ])

        assert result == su.EXIT_SUCCESS
        call_kwargs = mock_capture.call_args[1]
        assert call_kwargs['url'] == 'https://example.com'
        assert call_kwargs['output'] == 'output.png'
        assert call_kwargs['format_type'] == 'png'
        assert call_kwargs['full_page'] is True
        assert call_kwargs['viewport'] == (1920, 1080)
        assert call_kwargs['wait_for'] == '.ready'
        assert call_kwargs['wait_until'] == 'networkidle'
        assert call_kwargs['timeout'] == 60
        assert call_kwargs['delay'] == 2.0
        assert call_kwargs['verbose'] is True

    def test_main_no_url(self):
        """Test main with no URL returns error."""
        result = su.main([])
        assert result == su.EXIT_INVALID_ARGS

    def test_main_invalid_viewport(self):
        """Test main with invalid viewport returns error."""
        result = su.main(['https://example.com', '--viewport', 'invalid'])
        assert result == su.EXIT_INVALID_ARGS

    @patch('screenshot_url.list_devices')
    def test_main_list_devices(self, mock_list_devices):
        """Test main with --list-devices flag."""
        mock_list_devices.side_effect = SystemExit(0)

        with pytest.raises(SystemExit):
            su.main(['--list-devices'])

        mock_list_devices.assert_called_once()

    @patch('screenshot_url.capture_screenshot')
    def test_main_infers_jpeg_from_extension(self, mock_capture):
        """Test main infers JPEG format from file extension."""
        mock_capture.return_value = su.EXIT_SUCCESS

        result = su.main(['https://example.com', '-o', 'photo.jpg'])

        assert result == su.EXIT_SUCCESS
        call_kwargs = mock_capture.call_args[1]
        assert call_kwargs['format_type'] == 'jpeg'

    @patch('screenshot_url.capture_screenshot')
    def test_main_infers_pdf_from_extension(self, mock_capture):
        """Test main infers PDF format from file extension."""
        mock_capture.return_value = su.EXIT_SUCCESS

        result = su.main(['https://example.com', '-o', 'doc.pdf'])

        assert result == su.EXIT_SUCCESS
        call_kwargs = mock_capture.call_args[1]
        assert call_kwargs['format_type'] == 'pdf'

    @patch('screenshot_url.capture_screenshot')
    def test_main_default_output_filename(self, mock_capture):
        """Test main uses default output filename."""
        mock_capture.return_value = su.EXIT_SUCCESS

        result = su.main(['https://example.com'])

        assert result == su.EXIT_SUCCESS
        call_kwargs = mock_capture.call_args[1]
        assert call_kwargs['output'] == 'screenshot.png'


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    @patch('playwright.sync_api.sync_playwright')
    @patch.dict(os.environ, {'PLAYWRIGHT_BROWSER': 'firefox'})
    def test_browser_from_env(self, mock_sync_playwright):
        """Test browser selection from environment variable."""
        mock_playwright, mock_browser, mock_page, mock_browser_type = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
        )

        assert result == su.EXIT_SUCCESS
        # Firefox should be used from env var
        mock_browser_type.launch.assert_called_once()

    @patch('playwright.sync_api.sync_playwright')
    @patch.dict(os.environ, {'PLAYWRIGHT_BROWSER': 'webkit'})
    def test_browser_arg_overrides_env(self, mock_sync_playwright):
        """Test browser argument overrides environment variable."""
        mock_playwright, mock_browser, mock_page, mock_browser_type = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        result = su.capture_screenshot(
            url='https://example.com',
            output='test.png',
            format_type='png',
            browser_name='chromium',  # Explicit argument
        )

        assert result == su.EXIT_SUCCESS
        # Chromium should be used from argument, not webkit from env
        mock_browser_type.launch.assert_called_once()


class TestListDevices:
    """Tests for device listing functionality."""

    @patch('playwright.sync_api.sync_playwright')
    def test_list_devices_output(self, mock_sync_playwright):
        """Test list_devices outputs device names."""
        mock_playwright, _, _, _ = create_playwright_mock()
        mock_sync_playwright.return_value.__enter__ = Mock(return_value=mock_playwright)
        mock_sync_playwright.return_value.__exit__ = Mock(return_value=False)

        with pytest.raises(SystemExit) as exc_info:
            su.list_devices()

        assert exc_info.value.code == su.EXIT_SUCCESS

    def test_get_available_devices(self):
        """Test get_available_devices returns sorted list."""
        mock_playwright = Mock()
        mock_playwright.devices = {
            'iPhone 13': {},
            'iPad Pro': {},
            'Pixel 5': {},
        }

        devices = su.get_available_devices(mock_playwright)

        # Devices are sorted alphabetically
        assert 'iPad Pro' in devices
        assert 'iPhone 13' in devices
        assert 'Pixel 5' in devices
        assert len(devices) == 3
        # Check sorting
        assert devices == sorted(devices)


class TestExitCodes:
    """Tests to verify correct exit codes."""

    def test_exit_code_constants(self):
        """Test exit code constants are defined correctly."""
        assert su.EXIT_SUCCESS == 0
        assert su.EXIT_BROWSER_ERROR == 1
        assert su.EXIT_NAVIGATION_ERROR == 2
        assert su.EXIT_TIMEOUT_ERROR == 3
        assert su.EXIT_INVALID_ARGS == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
