#!/usr/bin/env python3
"""
Comprehensive tests for webhook_dispatch CLI tool.

This module tests all functionality of the webhook_dispatch tool including:
- Argument parsing
- Header and form field parsing
- JSON body loading
- Authentication handling
- Retry logic with exponential backoff
- Request dispatching
- Response formatting
- Exit codes
"""

import base64
import json
import logging
import os
import sys
import tempfile
import time
import unittest
from io import StringIO
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import the module under test
import webhook_dispatch as wd


class TestParseHeader(unittest.TestCase):
    """Tests for parse_header function."""

    def test_valid_header(self):
        """Test parsing a valid header string."""
        name, value = wd.parse_header("Content-Type: application/json")
        self.assertEqual(name, "Content-Type")
        self.assertEqual(value, "application/json")

    def test_header_with_multiple_colons(self):
        """Test parsing header with multiple colons in value."""
        name, value = wd.parse_header("Authorization: Bearer abc:def:ghi")
        self.assertEqual(name, "Authorization")
        self.assertEqual(value, "Bearer abc:def:ghi")

    def test_header_with_extra_spaces(self):
        """Test parsing header with extra whitespace."""
        name, value = wd.parse_header("  X-Custom  :   some value  ")
        self.assertEqual(name, "X-Custom")
        self.assertEqual(value, "some value")

    def test_invalid_header_no_colon(self):
        """Test that invalid header without colon raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            wd.parse_header("InvalidHeader")
        self.assertIn("Invalid header format", str(ctx.exception))

    def test_empty_value(self):
        """Test parsing header with empty value."""
        name, value = wd.parse_header("X-Empty:")
        self.assertEqual(name, "X-Empty")
        self.assertEqual(value, "")


class TestParseFormField(unittest.TestCase):
    """Tests for parse_form_field function."""

    def test_valid_form_field(self):
        """Test parsing a valid form field."""
        key, value = wd.parse_form_field("username=john")
        self.assertEqual(key, "username")
        self.assertEqual(value, "john")

    def test_form_field_with_equals_in_value(self):
        """Test parsing form field with equals sign in value."""
        key, value = wd.parse_form_field("equation=x=y+z")
        self.assertEqual(key, "equation")
        self.assertEqual(value, "x=y+z")

    def test_form_field_with_spaces(self):
        """Test parsing form field with spaces in key."""
        key, value = wd.parse_form_field("  key  =value")
        self.assertEqual(key, "key")
        self.assertEqual(value, "value")

    def test_invalid_form_field_no_equals(self):
        """Test that invalid form field without equals raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            wd.parse_form_field("invalidfield")
        self.assertIn("Invalid form field format", str(ctx.exception))

    def test_empty_value(self):
        """Test parsing form field with empty value."""
        key, value = wd.parse_form_field("key=")
        self.assertEqual(key, "key")
        self.assertEqual(value, "")


class TestLoadJsonBody(unittest.TestCase):
    """Tests for load_json_body function."""

    def test_valid_json_string(self):
        """Test loading valid JSON string."""
        result = wd.load_json_body('{"key": "value", "number": 42}')
        self.assertEqual(result, {"key": "value", "number": 42})

    def test_json_array(self):
        """Test loading JSON array."""
        result = wd.load_json_body('[1, 2, 3]')
        self.assertEqual(result, [1, 2, 3])

    def test_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            wd.load_json_body("{invalid json}")
        self.assertIn("Invalid JSON", str(ctx.exception))

    def test_json_from_file(self):
        """Test loading JSON from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"from": "file"}, f)
            f.flush()
            try:
                result = wd.load_json_body(f"@{f.name}")
                self.assertEqual(result, {"from": "file"})
            finally:
                os.unlink(f.name)

    def test_json_file_not_found(self):
        """Test that missing file raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            wd.load_json_body("@/nonexistent/file.json")
        self.assertIn("JSON file not found", str(ctx.exception))

    def test_json_file_invalid_content(self):
        """Test that file with invalid JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            f.flush()
            try:
                with self.assertRaises(ValueError) as ctx:
                    wd.load_json_body(f"@{f.name}")
                self.assertIn("Invalid JSON", str(ctx.exception))
            finally:
                os.unlink(f.name)


class TestBuildAuthHeader(unittest.TestCase):
    """Tests for build_auth_header function."""

    def test_bearer_token(self):
        """Test Bearer token authentication."""
        result = wd.build_auth_header(auth_bearer="mytoken123")
        self.assertEqual(result, "Bearer mytoken123")

    def test_basic_auth(self):
        """Test Basic authentication."""
        result = wd.build_auth_header(auth_basic="user:password")
        expected = "Basic " + base64.b64encode(b"user:password").decode()
        self.assertEqual(result, expected)

    def test_basic_auth_with_colon_in_password(self):
        """Test Basic auth with colon in password."""
        result = wd.build_auth_header(auth_basic="user:pass:word")
        expected = "Basic " + base64.b64encode(b"user:pass:word").decode()
        self.assertEqual(result, expected)

    def test_invalid_basic_auth(self):
        """Test that Basic auth without colon raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            wd.build_auth_header(auth_basic="invalidformat")
        self.assertIn("user:password", str(ctx.exception))

    def test_env_token_fallback(self):
        """Test environment token fallback."""
        result = wd.build_auth_header(env_token="envtoken")
        self.assertEqual(result, "Bearer envtoken")

    def test_bearer_takes_priority(self):
        """Test that explicit Bearer takes priority over env token."""
        result = wd.build_auth_header(auth_bearer="explicit", env_token="fallback")
        self.assertEqual(result, "Bearer explicit")

    def test_no_auth(self):
        """Test no authentication returns None."""
        result = wd.build_auth_header()
        self.assertIsNone(result)


class TestCalculateBackoffDelay(unittest.TestCase):
    """Tests for calculate_backoff_delay function."""

    def test_first_attempt(self):
        """Test delay for first retry attempt."""
        delay = wd.calculate_backoff_delay(0, base_delay=1.0)
        self.assertEqual(delay, 1.0)

    def test_exponential_growth(self):
        """Test exponential growth of delay."""
        delays = [wd.calculate_backoff_delay(i, base_delay=1.0) for i in range(4)]
        self.assertEqual(delays, [1.0, 2.0, 4.0, 8.0])

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        delay = wd.calculate_backoff_delay(10, base_delay=1.0, max_delay=60.0)
        self.assertEqual(delay, 60.0)

    def test_custom_base_delay(self):
        """Test custom base delay."""
        delay = wd.calculate_backoff_delay(0, base_delay=2.0)
        self.assertEqual(delay, 2.0)


class TestSendRequest(unittest.TestCase):
    """Tests for send_request function."""

    @patch("webhook_dispatch.requests.request")
    def test_get_request(self, mock_request):
        """Test sending GET request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        response = wd.send_request("GET", "https://example.com/api")

        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args[1]
        self.assertEqual(call_kwargs["method"], "GET")
        self.assertEqual(call_kwargs["url"], "https://example.com/api")
        self.assertEqual(response, mock_response)

    @patch("webhook_dispatch.requests.request")
    def test_post_with_json(self, mock_request):
        """Test sending POST request with JSON body."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_request.return_value = mock_response

        json_body = {"key": "value"}
        response = wd.send_request("POST", "https://example.com/api", json_body=json_body)

        call_kwargs = mock_request.call_args[1]
        self.assertEqual(call_kwargs["method"], "POST")
        self.assertEqual(call_kwargs["json"], json_body)

    @patch("webhook_dispatch.requests.request")
    def test_post_with_form_data(self, mock_request):
        """Test sending POST request with form data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        form_data = {"field1": "value1", "field2": "value2"}
        response = wd.send_request("POST", "https://example.com/api", form_data=form_data)

        call_kwargs = mock_request.call_args[1]
        self.assertEqual(call_kwargs["data"], form_data)

    @patch("webhook_dispatch.requests.request")
    def test_request_with_headers(self, mock_request):
        """Test sending request with custom headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        headers = {"Authorization": "Bearer token", "X-Custom": "value"}
        response = wd.send_request("GET", "https://example.com/api", headers=headers)

        call_kwargs = mock_request.call_args[1]
        self.assertEqual(call_kwargs["headers"], headers)

    @patch("webhook_dispatch.requests.request")
    def test_request_with_timeout(self, mock_request):
        """Test sending request with custom timeout."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        response = wd.send_request("GET", "https://example.com/api", timeout=60)

        call_kwargs = mock_request.call_args[1]
        self.assertEqual(call_kwargs["timeout"], 60)

    @patch("webhook_dispatch.requests.request")
    def test_request_with_raw_body(self, mock_request):
        """Test sending request with raw body."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        raw_body = "plain text body"
        response = wd.send_request("POST", "https://example.com/api", raw_body=raw_body)

        call_kwargs = mock_request.call_args[1]
        self.assertEqual(call_kwargs["data"], raw_body)


class TestDispatchWebhook(unittest.TestCase):
    """Tests for dispatch_webhook function."""

    @patch("webhook_dispatch.send_request")
    def test_successful_request(self, mock_send):
        """Test successful request returns exit code 0."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_send.return_value = mock_response

        response, exit_code = wd.dispatch_webhook("GET", "https://example.com")

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        self.assertEqual(response, mock_response)

    @patch("webhook_dispatch.send_request")
    def test_client_error(self, mock_send):
        """Test client error returns exit code 1."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_send.return_value = mock_response

        response, exit_code = wd.dispatch_webhook("GET", "https://example.com")

        self.assertEqual(exit_code, wd.EXIT_CLIENT_ERROR)

    @patch("webhook_dispatch.send_request")
    def test_server_error(self, mock_send):
        """Test server error returns exit code 2."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_send.return_value = mock_response

        response, exit_code = wd.dispatch_webhook("GET", "https://example.com")

        self.assertEqual(exit_code, wd.EXIT_SERVER_ERROR)

    @patch("webhook_dispatch.send_request")
    def test_connection_error(self, mock_send):
        """Test connection error returns exit code 3."""
        from requests.exceptions import ConnectionError
        mock_send.side_effect = ConnectionError("Connection refused")

        response, exit_code = wd.dispatch_webhook("GET", "https://example.com")

        self.assertEqual(exit_code, wd.EXIT_CONNECTION_ERROR)
        self.assertIsNone(response)

    @patch("webhook_dispatch.send_request")
    def test_timeout_error(self, mock_send):
        """Test timeout error returns exit code 3."""
        from requests.exceptions import Timeout
        mock_send.side_effect = Timeout("Request timed out")

        response, exit_code = wd.dispatch_webhook("GET", "https://example.com")

        self.assertEqual(exit_code, wd.EXIT_CONNECTION_ERROR)

    @patch("webhook_dispatch.time.sleep")
    @patch("webhook_dispatch.send_request")
    def test_retry_on_server_error(self, mock_send, mock_sleep):
        """Test retry on server error."""
        mock_error_response = Mock()
        mock_error_response.status_code = 503

        mock_success_response = Mock()
        mock_success_response.status_code = 200

        mock_send.side_effect = [mock_error_response, mock_success_response]

        response, exit_code = wd.dispatch_webhook(
            "GET", "https://example.com", retry_count=1
        )

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        self.assertEqual(mock_send.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("webhook_dispatch.time.sleep")
    @patch("webhook_dispatch.send_request")
    def test_retry_on_connection_error(self, mock_send, mock_sleep):
        """Test retry on connection error."""
        from requests.exceptions import ConnectionError

        mock_response = Mock()
        mock_response.status_code = 200

        mock_send.side_effect = [ConnectionError(), mock_response]

        response, exit_code = wd.dispatch_webhook(
            "GET", "https://example.com", retry_count=1
        )

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        self.assertEqual(mock_send.call_count, 2)

    @patch("webhook_dispatch.time.sleep")
    @patch("webhook_dispatch.send_request")
    def test_no_retry_on_client_error(self, mock_send, mock_sleep):
        """Test no retry on client error (4xx)."""
        mock_response = Mock()
        mock_response.status_code = 400

        mock_send.return_value = mock_response

        response, exit_code = wd.dispatch_webhook(
            "GET", "https://example.com", retry_count=3
        )

        self.assertEqual(exit_code, wd.EXIT_CLIENT_ERROR)
        self.assertEqual(mock_send.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("webhook_dispatch.time.sleep")
    @patch("webhook_dispatch.send_request")
    def test_all_retries_exhausted(self, mock_send, mock_sleep):
        """Test behavior when all retries are exhausted."""
        mock_response = Mock()
        mock_response.status_code = 500

        mock_send.return_value = mock_response

        response, exit_code = wd.dispatch_webhook(
            "GET", "https://example.com", retry_count=2
        )

        self.assertEqual(exit_code, wd.EXIT_SERVER_ERROR)
        self.assertEqual(mock_send.call_count, 3)  # Initial + 2 retries


class TestFormatResponseOutput(unittest.TestCase):
    """Tests for format_response_output function."""

    def test_none_response(self):
        """Test formatting None response."""
        output = wd.format_response_output(None)
        self.assertEqual(output, "")

    def test_simple_response(self):
        """Test formatting simple response."""
        mock_response = Mock()
        mock_response.text = "Hello, World!"
        mock_response.headers = {}

        output = wd.format_response_output(mock_response)
        self.assertEqual(output, "Hello, World!")

    def test_json_auto_format(self):
        """Test auto-formatting JSON response."""
        mock_response = Mock()
        mock_response.text = '{"key":"value"}'
        mock_response.headers = {"Content-Type": "application/json"}

        output = wd.format_response_output(mock_response, output_format="auto")
        self.assertIn('"key"', output)
        self.assertIn('"value"', output)

    def test_json_forced_format(self):
        """Test forcing JSON formatting."""
        mock_response = Mock()
        mock_response.text = '{"key":"value"}'
        mock_response.headers = {"Content-Type": "text/plain"}

        output = wd.format_response_output(mock_response, output_format="json")
        # Should be pretty-printed
        self.assertIn("\n", output)

    def test_raw_format(self):
        """Test raw output format."""
        mock_response = Mock()
        mock_response.text = '{"key":"value"}'
        mock_response.headers = {"Content-Type": "application/json"}

        output = wd.format_response_output(mock_response, output_format="raw")
        # Should not be pretty-printed
        self.assertEqual(output, '{"key":"value"}')

    def test_show_headers(self):
        """Test showing response headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.reason = "OK"
        mock_response.text = "body"
        mock_response.headers = {"Content-Type": "text/plain", "X-Custom": "value"}

        output = wd.format_response_output(mock_response, show_headers=True)
        self.assertIn("HTTP 200 OK", output)
        self.assertIn("Content-Type: text/plain", output)
        self.assertIn("X-Custom: value", output)
        self.assertIn("body", output)


class TestArgumentParser(unittest.TestCase):
    """Tests for argument parser."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = wd.create_argument_parser()

    def test_required_args(self):
        """Test parsing required arguments."""
        args = self.parser.parse_args(["GET", "https://example.com"])
        self.assertEqual(args.method, "GET")
        self.assertEqual(args.url, "https://example.com")

    def test_all_methods(self):
        """Test all supported HTTP methods."""
        for method in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
            args = self.parser.parse_args([method, "https://example.com"])
            self.assertEqual(args.method, method)

    def test_invalid_method(self):
        """Test invalid HTTP method."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["INVALID", "https://example.com"])

    def test_json_body(self):
        """Test JSON body argument."""
        args = self.parser.parse_args([
            "POST", "https://example.com",
            "--json", '{"key": "value"}'
        ])
        self.assertEqual(args.json_body, '{"key": "value"}')

    def test_json_short_flag(self):
        """Test JSON body with short flag."""
        args = self.parser.parse_args([
            "POST", "https://example.com",
            "-j", '{"key": "value"}'
        ])
        self.assertEqual(args.json_body, '{"key": "value"}')

    def test_form_fields(self):
        """Test form field arguments."""
        args = self.parser.parse_args([
            "POST", "https://example.com",
            "--form", "key1=value1",
            "--form", "key2=value2"
        ])
        self.assertEqual(args.form_fields, ["key1=value1", "key2=value2"])

    def test_form_short_flag(self):
        """Test form field with short flag."""
        args = self.parser.parse_args([
            "POST", "https://example.com",
            "-f", "key=value"
        ])
        self.assertEqual(args.form_fields, ["key=value"])

    def test_raw_body(self):
        """Test raw body argument."""
        args = self.parser.parse_args([
            "POST", "https://example.com",
            "--data", "raw content"
        ])
        self.assertEqual(args.raw_body, "raw content")

    def test_headers(self):
        """Test header arguments."""
        args = self.parser.parse_args([
            "GET", "https://example.com",
            "-H", "Authorization: Bearer token",
            "-H", "X-Custom: value"
        ])
        self.assertEqual(len(args.headers), 2)

    def test_auth_bearer(self):
        """Test Bearer auth argument."""
        args = self.parser.parse_args([
            "GET", "https://example.com",
            "--auth-bearer", "mytoken"
        ])
        self.assertEqual(args.auth_bearer, "mytoken")

    def test_auth_basic(self):
        """Test Basic auth argument."""
        args = self.parser.parse_args([
            "GET", "https://example.com",
            "--auth-basic", "user:pass"
        ])
        self.assertEqual(args.auth_basic, "user:pass")

    def test_retry(self):
        """Test retry argument."""
        args = self.parser.parse_args([
            "GET", "https://example.com",
            "--retry", "3"
        ])
        self.assertEqual(args.retry, 3)

    def test_timeout(self):
        """Test timeout argument."""
        args = self.parser.parse_args([
            "GET", "https://example.com",
            "--timeout", "60"
        ])
        self.assertEqual(args.timeout, 60)

    def test_verbose(self):
        """Test verbose flag."""
        args = self.parser.parse_args([
            "GET", "https://example.com",
            "--verbose"
        ])
        self.assertTrue(args.verbose)

    def test_debug(self):
        """Test debug flag."""
        args = self.parser.parse_args([
            "GET", "https://example.com",
            "--debug"
        ])
        self.assertTrue(args.debug)

    def test_quiet(self):
        """Test quiet flag."""
        args = self.parser.parse_args([
            "GET", "https://example.com",
            "--quiet"
        ])
        self.assertTrue(args.quiet)

    def test_show_headers(self):
        """Test show-headers flag."""
        args = self.parser.parse_args([
            "GET", "https://example.com",
            "--show-headers"
        ])
        self.assertTrue(args.show_headers)

    def test_mutually_exclusive_body(self):
        """Test that body options are mutually exclusive."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                "POST", "https://example.com",
                "--json", '{}',
                "--form", "key=value"
            ])

    def test_mutually_exclusive_auth(self):
        """Test that auth options are mutually exclusive."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                "GET", "https://example.com",
                "--auth-bearer", "token",
                "--auth-basic", "user:pass"
            ])


class TestMain(unittest.TestCase):
    """Tests for main function."""

    @patch("webhook_dispatch.dispatch_webhook")
    def test_successful_get(self, mock_dispatch):
        """Test successful GET request through main."""
        mock_response = Mock()
        mock_response.text = '{"status": "ok"}'
        mock_response.headers = {"Content-Type": "application/json"}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        exit_code = wd.main(["GET", "https://example.com", "--quiet"])

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        mock_dispatch.assert_called_once()

    @patch("webhook_dispatch.dispatch_webhook")
    def test_post_with_json(self, mock_dispatch):
        """Test POST with JSON through main."""
        mock_response = Mock()
        mock_response.text = "{}"
        mock_response.headers = {}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        exit_code = wd.main([
            "POST", "https://example.com",
            "--json", '{"key": "value"}',
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        call_kwargs = mock_dispatch.call_args[1]
        self.assertEqual(call_kwargs["json_body"], {"key": "value"})

    @patch("webhook_dispatch.dispatch_webhook")
    def test_post_with_form(self, mock_dispatch):
        """Test POST with form data through main."""
        mock_response = Mock()
        mock_response.text = "{}"
        mock_response.headers = {}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        exit_code = wd.main([
            "POST", "https://example.com",
            "--form", "key1=value1",
            "--form", "key2=value2",
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        call_kwargs = mock_dispatch.call_args[1]
        self.assertEqual(call_kwargs["form_data"], {"key1": "value1", "key2": "value2"})

    @patch("webhook_dispatch.dispatch_webhook")
    def test_request_with_headers(self, mock_dispatch):
        """Test request with headers through main."""
        mock_response = Mock()
        mock_response.text = "{}"
        mock_response.headers = {}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        exit_code = wd.main([
            "GET", "https://example.com",
            "-H", "X-Custom: value",
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        call_kwargs = mock_dispatch.call_args[1]
        self.assertEqual(call_kwargs["headers"]["X-Custom"], "value")

    @patch("webhook_dispatch.dispatch_webhook")
    def test_bearer_auth(self, mock_dispatch):
        """Test Bearer auth through main."""
        mock_response = Mock()
        mock_response.text = "{}"
        mock_response.headers = {}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        exit_code = wd.main([
            "GET", "https://example.com",
            "--auth-bearer", "mytoken",
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        call_kwargs = mock_dispatch.call_args[1]
        self.assertEqual(call_kwargs["headers"]["Authorization"], "Bearer mytoken")

    @patch("webhook_dispatch.dispatch_webhook")
    def test_basic_auth(self, mock_dispatch):
        """Test Basic auth through main."""
        mock_response = Mock()
        mock_response.text = "{}"
        mock_response.headers = {}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        exit_code = wd.main([
            "GET", "https://example.com",
            "--auth-basic", "user:pass",
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        call_kwargs = mock_dispatch.call_args[1]
        expected = "Basic " + base64.b64encode(b"user:pass").decode()
        self.assertEqual(call_kwargs["headers"]["Authorization"], expected)

    @patch.dict(os.environ, {"WEBHOOK_AUTH_TOKEN": "envtoken"})
    @patch("webhook_dispatch.dispatch_webhook")
    def test_env_token(self, mock_dispatch):
        """Test environment token fallback."""
        mock_response = Mock()
        mock_response.text = "{}"
        mock_response.headers = {}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        exit_code = wd.main([
            "GET", "https://example.com",
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        call_kwargs = mock_dispatch.call_args[1]
        self.assertEqual(call_kwargs["headers"]["Authorization"], "Bearer envtoken")

    @patch.dict(os.environ, {"WEBHOOK_AUTH_TOKEN": "envtoken"})
    @patch("webhook_dispatch.dispatch_webhook")
    def test_explicit_auth_overrides_env(self, mock_dispatch):
        """Test explicit auth overrides environment token."""
        mock_response = Mock()
        mock_response.text = "{}"
        mock_response.headers = {}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        exit_code = wd.main([
            "GET", "https://example.com",
            "--auth-bearer", "explicit",
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        call_kwargs = mock_dispatch.call_args[1]
        self.assertEqual(call_kwargs["headers"]["Authorization"], "Bearer explicit")

    def test_invalid_json(self):
        """Test invalid JSON returns exit code 4."""
        exit_code = wd.main([
            "POST", "https://example.com",
            "--json", "{invalid}",
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_INVALID_ARGS)

    def test_invalid_header(self):
        """Test invalid header returns exit code 4."""
        exit_code = wd.main([
            "GET", "https://example.com",
            "-H", "InvalidHeader",
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_INVALID_ARGS)

    def test_invalid_form_field(self):
        """Test invalid form field returns exit code 4."""
        exit_code = wd.main([
            "POST", "https://example.com",
            "--form", "invalidfield",
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_INVALID_ARGS)

    @patch("webhook_dispatch.dispatch_webhook")
    def test_retry_options(self, mock_dispatch):
        """Test retry options are passed through."""
        mock_response = Mock()
        mock_response.text = "{}"
        mock_response.headers = {}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        exit_code = wd.main([
            "GET", "https://example.com",
            "--retry", "3",
            "--retry-delay", "2.0",
            "--retry-max-delay", "30.0",
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        call_kwargs = mock_dispatch.call_args[1]
        self.assertEqual(call_kwargs["retry_count"], 3)
        self.assertEqual(call_kwargs["retry_base_delay"], 2.0)
        self.assertEqual(call_kwargs["retry_max_delay"], 30.0)

    @patch("webhook_dispatch.dispatch_webhook")
    def test_timeout_option(self, mock_dispatch):
        """Test timeout option is passed through."""
        mock_response = Mock()
        mock_response.text = "{}"
        mock_response.headers = {}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        exit_code = wd.main([
            "GET", "https://example.com",
            "--timeout", "60",
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        call_kwargs = mock_dispatch.call_args[1]
        self.assertEqual(call_kwargs["timeout"], 60)


class TestSetupLogging(unittest.TestCase):
    """Tests for setup_logging function."""

    def test_default_level(self):
        """Test default logging level is WARNING."""
        logger = wd.setup_logging(verbose=False, debug=False)
        self.assertEqual(logger.level, logging.WARNING)

    def test_verbose_level(self):
        """Test verbose logging level is INFO."""
        logger = wd.setup_logging(verbose=True, debug=False)
        self.assertEqual(logger.level, logging.INFO)

    def test_debug_level(self):
        """Test debug logging level is DEBUG."""
        logger = wd.setup_logging(verbose=False, debug=True)
        self.assertEqual(logger.level, logging.DEBUG)

    def test_debug_overrides_verbose(self):
        """Test debug overrides verbose."""
        logger = wd.setup_logging(verbose=True, debug=True)
        self.assertEqual(logger.level, logging.DEBUG)


class TestExitCodes(unittest.TestCase):
    """Tests for exit code constants."""

    def test_exit_code_values(self):
        """Test exit code constant values."""
        self.assertEqual(wd.EXIT_SUCCESS, 0)
        self.assertEqual(wd.EXIT_CLIENT_ERROR, 1)
        self.assertEqual(wd.EXIT_SERVER_ERROR, 2)
        self.assertEqual(wd.EXIT_CONNECTION_ERROR, 3)
        self.assertEqual(wd.EXIT_INVALID_ARGS, 4)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and special scenarios."""

    @patch("webhook_dispatch.dispatch_webhook")
    def test_empty_response_body(self, mock_dispatch):
        """Test handling empty response body."""
        mock_response = Mock()
        mock_response.text = ""
        mock_response.headers = {}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        exit_code = wd.main(["DELETE", "https://example.com"])
        self.assertEqual(exit_code, wd.EXIT_SUCCESS)

    @patch("webhook_dispatch.dispatch_webhook")
    def test_url_with_query_params(self, mock_dispatch):
        """Test URL with query parameters."""
        mock_response = Mock()
        mock_response.text = "{}"
        mock_response.headers = {}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        url = "https://example.com/api?param1=value1&param2=value2"
        exit_code = wd.main(["GET", url, "--quiet"])

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        call_args = mock_dispatch.call_args
        self.assertEqual(call_args[1]["url"], url)

    @patch("webhook_dispatch.dispatch_webhook")
    def test_special_characters_in_json(self, mock_dispatch):
        """Test JSON with special characters."""
        mock_response = Mock()
        mock_response.text = "{}"
        mock_response.headers = {}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        json_body = '{"message": "Hello, \\"World\\"!", "emoji": "test"}'
        exit_code = wd.main([
            "POST", "https://example.com",
            "--json", json_body,
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)

    @patch("webhook_dispatch.dispatch_webhook")
    def test_authorization_header_not_duplicated(self, mock_dispatch):
        """Test that explicit Authorization header is not overwritten."""
        mock_response = Mock()
        mock_response.text = "{}"
        mock_response.headers = {}
        mock_dispatch.return_value = (mock_response, wd.EXIT_SUCCESS)

        exit_code = wd.main([
            "GET", "https://example.com",
            "-H", "Authorization: Custom auth",
            "--auth-bearer", "token",
            "--quiet"
        ])

        self.assertEqual(exit_code, wd.EXIT_SUCCESS)
        call_kwargs = mock_dispatch.call_args[1]
        # The explicit header should be preserved
        self.assertEqual(call_kwargs["headers"]["Authorization"], "Custom auth")


if __name__ == "__main__":
    unittest.main()
