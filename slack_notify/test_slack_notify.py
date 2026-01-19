#!/usr/bin/env python3
"""
Comprehensive tests for slack_notify CLI tool.

This test suite covers:
- Argument parsing and validation
- Payload building
- API and webhook communication (mocked)
- Error handling
- Exit codes

Run with: pytest test_slack_notify.py -v
"""

import json
import os
import sys
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock, mock_open
import urllib.error

# Import the module under test
import slack_notify


class TestArgumentParser(unittest.TestCase):
    """Tests for command line argument parsing."""

    def test_basic_channel_message(self):
        """Test basic channel message arguments."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-c", "#general", "-m", "Hello"])
        self.assertEqual(args.channel, "#general")
        self.assertEqual(args.message, "Hello")
        self.assertFalse(args.webhook)

    def test_user_message(self):
        """Test direct message to user."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-u", "U12345", "-m", "Hello"])
        self.assertEqual(args.user, "U12345")
        self.assertEqual(args.message, "Hello")

    def test_webhook_mode(self):
        """Test webhook mode flag."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["--webhook", "-m", "Alert"])
        self.assertTrue(args.webhook)
        self.assertEqual(args.message, "Alert")

    def test_thread_options(self):
        """Test threading options."""
        parser = slack_notify.create_parser()
        args = parser.parse_args([
            "-c", "#general",
            "-m", "Reply",
            "--thread-ts", "1234.5678",
            "--reply-broadcast"
        ])
        self.assertEqual(args.thread_ts, "1234.5678")
        self.assertTrue(args.reply_broadcast)

    def test_customization_options(self):
        """Test username and icon customization."""
        parser = slack_notify.create_parser()
        args = parser.parse_args([
            "-c", "#general",
            "-m", "Hello",
            "--username", "TestBot",
            "--icon-emoji", ":robot_face:"
        ])
        self.assertEqual(args.username, "TestBot")
        self.assertEqual(args.icon_emoji, ":robot_face:")

    def test_blocks_argument(self):
        """Test blocks JSON argument."""
        parser = slack_notify.create_parser()
        blocks_json = '[{"type":"section","text":{"type":"mrkdwn","text":"test"}}]'
        args = parser.parse_args(["-c", "#general", "--blocks", blocks_json])
        self.assertEqual(args.blocks, blocks_json)

    def test_verbose_quiet_flags(self):
        """Test verbose and quiet output flags."""
        parser = slack_notify.create_parser()

        args_verbose = parser.parse_args(["-c", "#general", "-m", "test", "-v"])
        self.assertTrue(args_verbose.verbose)

        args_quiet = parser.parse_args(["-c", "#general", "-m", "test", "-q"])
        self.assertTrue(args_quiet.quiet)

    def test_dry_run_flag(self):
        """Test dry run flag."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-c", "#general", "-m", "test", "--dry-run"])
        self.assertTrue(args.dry_run)

    def test_unfurl_options(self):
        """Test link and media unfurling options."""
        parser = slack_notify.create_parser()

        args_unfurl = parser.parse_args([
            "-c", "#general", "-m", "test",
            "--unfurl-links", "--unfurl-media"
        ])
        self.assertTrue(args_unfurl.unfurl_links)
        self.assertTrue(args_unfurl.unfurl_media)

        args_no_unfurl = parser.parse_args([
            "-c", "#general", "-m", "test",
            "--no-unfurl-links", "--no-unfurl-media"
        ])
        self.assertTrue(args_no_unfurl.no_unfurl_links)
        self.assertTrue(args_no_unfurl.no_unfurl_media)


class TestArgumentValidation(unittest.TestCase):
    """Tests for argument validation."""

    def test_missing_message_content(self):
        """Test validation fails without message content."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-c", "#general"])

        with self.assertRaises(SystemExit) as cm:
            slack_notify.validate_args(args)
        self.assertEqual(cm.exception.code, slack_notify.EXIT_INVALID_ARGS)

    def test_missing_target_in_api_mode(self):
        """Test validation fails without channel/user in API mode."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-m", "Hello"])

        with self.assertRaises(SystemExit) as cm:
            slack_notify.validate_args(args)
        self.assertEqual(cm.exception.code, slack_notify.EXIT_INVALID_ARGS)

    def test_webhook_mode_no_target_required(self):
        """Test webhook mode doesn't require channel/user."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["--webhook", "-m", "Hello"])

        # Should not raise
        slack_notify.validate_args(args)

    def test_reply_broadcast_requires_thread_ts(self):
        """Test reply_broadcast requires thread_ts."""
        parser = slack_notify.create_parser()
        args = parser.parse_args([
            "-c", "#general",
            "-m", "test",
            "--reply-broadcast"
        ])

        with self.assertRaises(SystemExit) as cm:
            slack_notify.validate_args(args)
        self.assertEqual(cm.exception.code, slack_notify.EXIT_INVALID_ARGS)

    def test_verbose_quiet_mutually_exclusive(self):
        """Test verbose and quiet are mutually exclusive."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-c", "#general", "-m", "test", "-v", "-q"])

        with self.assertRaises(SystemExit) as cm:
            slack_notify.validate_args(args)
        self.assertEqual(cm.exception.code, slack_notify.EXIT_INVALID_ARGS)


class TestPayloadBuilding(unittest.TestCase):
    """Tests for payload construction."""

    def test_basic_payload(self):
        """Test basic payload construction."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-c", "#general", "-m", "Hello"])

        payload = slack_notify.build_payload(args)

        self.assertEqual(payload["channel"], "general")  # # stripped
        self.assertEqual(payload["text"], "Hello")

    def test_user_target_payload(self):
        """Test payload with user target."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-u", "@john", "-m", "Hello"])

        payload = slack_notify.build_payload(args)

        self.assertEqual(payload["channel"], "john")  # @ stripped
        self.assertEqual(payload["text"], "Hello")

    def test_webhook_payload_no_channel(self):
        """Test webhook payload doesn't include channel."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["--webhook", "-m", "Hello"])

        payload = slack_notify.build_payload(args)

        self.assertNotIn("channel", payload)
        self.assertEqual(payload["text"], "Hello")

    def test_thread_payload(self):
        """Test payload with threading."""
        parser = slack_notify.create_parser()
        args = parser.parse_args([
            "-c", "#general",
            "-m", "Reply",
            "--thread-ts", "1234.5678",
            "--reply-broadcast"
        ])

        payload = slack_notify.build_payload(args)

        self.assertEqual(payload["thread_ts"], "1234.5678")
        self.assertTrue(payload["reply_broadcast"])

    def test_customization_payload(self):
        """Test payload with customization."""
        parser = slack_notify.create_parser()
        args = parser.parse_args([
            "-c", "#general",
            "-m", "Hello",
            "--username", "TestBot",
            "--icon-emoji", ":robot:",
            "--icon-url", "https://example.com/icon.png"
        ])

        payload = slack_notify.build_payload(args)

        self.assertEqual(payload["username"], "TestBot")
        self.assertEqual(payload["icon_emoji"], ":robot:")
        self.assertEqual(payload["icon_url"], "https://example.com/icon.png")

    def test_blocks_payload(self):
        """Test payload with blocks JSON."""
        parser = slack_notify.create_parser()
        blocks_json = '[{"type":"section","text":{"type":"mrkdwn","text":"test"}}]'
        args = parser.parse_args(["-c", "#general", "--blocks", blocks_json])

        payload = slack_notify.build_payload(args)

        self.assertEqual(payload["blocks"], json.loads(blocks_json))

    def test_unfurl_options_payload(self):
        """Test payload with unfurl options."""
        parser = slack_notify.create_parser()

        args_disable = parser.parse_args([
            "-c", "#general", "-m", "test",
            "--no-unfurl-links", "--no-unfurl-media"
        ])
        payload = slack_notify.build_payload(args_disable)
        self.assertFalse(payload["unfurl_links"])
        self.assertFalse(payload["unfurl_media"])

        args_enable = parser.parse_args([
            "-c", "#general", "-m", "test",
            "--unfurl-links", "--unfurl-media"
        ])
        payload = slack_notify.build_payload(args_enable)
        self.assertTrue(payload["unfurl_links"])
        self.assertTrue(payload["unfurl_media"])

    def test_blocks_from_file(self):
        """Test loading blocks from file."""
        parser = slack_notify.create_parser()
        args = parser.parse_args([
            "-c", "#general",
            "--blocks-file", "/path/to/blocks.json"
        ])

        blocks_content = '[{"type":"section"}]'
        with patch("builtins.open", mock_open(read_data=blocks_content)):
            payload = slack_notify.build_payload(args)

        self.assertEqual(payload["blocks"], json.loads(blocks_content))

    def test_blocks_file_not_found(self):
        """Test error when blocks file not found."""
        parser = slack_notify.create_parser()
        args = parser.parse_args([
            "-c", "#general",
            "--blocks-file", "/nonexistent/blocks.json"
        ])

        with self.assertRaises(SystemExit) as cm:
            slack_notify.build_payload(args)
        self.assertEqual(cm.exception.code, slack_notify.EXIT_INVALID_ARGS)

    def test_invalid_blocks_json(self):
        """Test error with invalid blocks JSON."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-c", "#general", "--blocks", "invalid json"])

        with self.assertRaises(SystemExit) as cm:
            slack_notify.build_payload(args)
        self.assertEqual(cm.exception.code, slack_notify.EXIT_INVALID_ARGS)


class TestCredentials(unittest.TestCase):
    """Tests for credential handling."""

    def test_credentials_from_env(self):
        """Test getting credentials from environment."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-c", "#general", "-m", "test"])

        with patch.dict(os.environ, {
            "SLACK_BOT_TOKEN": "xoxb-test-token",
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"
        }):
            token, webhook = slack_notify.get_credentials(args)

        self.assertEqual(token, "xoxb-test-token")
        self.assertEqual(webhook, "https://hooks.slack.com/test")

    def test_credentials_from_args_override_env(self):
        """Test command line args override environment."""
        parser = slack_notify.create_parser()
        args = parser.parse_args([
            "-c", "#general", "-m", "test",
            "--token", "xoxb-arg-token",
            "--webhook-url", "https://hooks.slack.com/arg"
        ])

        with patch.dict(os.environ, {
            "SLACK_BOT_TOKEN": "xoxb-env-token",
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/env"
        }):
            token, webhook = slack_notify.get_credentials(args)

        self.assertEqual(token, "xoxb-arg-token")
        self.assertEqual(webhook, "https://hooks.slack.com/arg")


class TestAPIMessages(unittest.TestCase):
    """Tests for Slack Web API message sending."""

    @patch('urllib.request.urlopen')
    def test_successful_api_message(self, mock_urlopen):
        """Test successful API message sending."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "ok": True,
            "channel": "C12345",
            "ts": "1234567890.123456"
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        payload = {"channel": "general", "text": "Hello"}
        response = slack_notify.send_api_message("xoxb-test", payload)

        self.assertTrue(response["ok"])
        self.assertEqual(response["channel"], "C12345")
        self.assertEqual(response["ts"], "1234567890.123456")

    @patch('urllib.request.urlopen')
    def test_api_error_response(self, mock_urlopen):
        """Test handling API error response."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "ok": False,
            "error": "channel_not_found"
        }).encode('utf-8')
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        payload = {"channel": "nonexistent", "text": "Hello"}
        response = slack_notify.send_api_message("xoxb-test", payload)

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"], "channel_not_found")

    @patch('urllib.request.urlopen')
    def test_api_network_error(self, mock_urlopen):
        """Test handling network errors."""
        mock_urlopen.side_effect = urllib.error.URLError("Network unreachable")

        payload = {"channel": "general", "text": "Hello"}

        with self.assertRaises(SystemExit) as cm:
            slack_notify.send_api_message("xoxb-test", payload)
        self.assertEqual(cm.exception.code, slack_notify.EXIT_API_ERROR)

    @patch('urllib.request.urlopen')
    def test_api_http_error(self, mock_urlopen):
        """Test handling HTTP errors."""
        error = urllib.error.HTTPError(
            url="https://slack.com/api/chat.postMessage",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=StringIO('{"ok":false,"error":"internal_error"}')
        )
        error.read = lambda: b'{"ok":false,"error":"internal_error"}'
        mock_urlopen.side_effect = error

        payload = {"channel": "general", "text": "Hello"}
        response = slack_notify.send_api_message("xoxb-test", payload)

        self.assertFalse(response["ok"])


class TestWebhookMessages(unittest.TestCase):
    """Tests for webhook message sending."""

    @patch('urllib.request.urlopen')
    def test_successful_webhook_message(self, mock_urlopen):
        """Test successful webhook message sending."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'ok'
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        payload = {"text": "Hello"}
        response = slack_notify.send_webhook_message(
            "https://hooks.slack.com/test",
            payload
        )

        self.assertTrue(response["ok"])

    @patch('urllib.request.urlopen')
    def test_webhook_error_response(self, mock_urlopen):
        """Test handling webhook error response."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'invalid_payload'
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        payload = {"text": "Hello"}
        response = slack_notify.send_webhook_message(
            "https://hooks.slack.com/test",
            payload
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"], "invalid_payload")

    @patch('urllib.request.urlopen')
    def test_webhook_http_error(self, mock_urlopen):
        """Test handling webhook HTTP errors."""
        error = urllib.error.HTTPError(
            url="https://hooks.slack.com/test",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None
        )
        mock_urlopen.side_effect = error

        payload = {"text": "Hello"}

        with self.assertRaises(SystemExit) as cm:
            slack_notify.send_webhook_message(
                "https://hooks.slack.com/test",
                payload
            )
        self.assertEqual(cm.exception.code, slack_notify.EXIT_API_ERROR)


class TestResponseFormatting(unittest.TestCase):
    """Tests for response formatting."""

    def test_success_format(self):
        """Test formatting successful response."""
        response = {
            "ok": True,
            "channel": "C12345",
            "ts": "1234567890.123456"
        }
        formatted = slack_notify.format_response(response)
        self.assertIn("Message sent successfully", formatted)
        self.assertIn("1234567890.123456", formatted)

    def test_success_format_simple(self):
        """Test formatting simple success response."""
        response = {"ok": True}
        formatted = slack_notify.format_response(response)
        self.assertEqual(formatted, "Message sent successfully")

    def test_error_format(self):
        """Test formatting error response."""
        response = {"ok": False, "error": "channel_not_found"}
        formatted = slack_notify.format_response(response)
        self.assertIn("Failed to send message", formatted)
        self.assertIn("channel_not_found", formatted)

    def test_verbose_format(self):
        """Test verbose formatting."""
        response = {"ok": True, "channel": "C12345", "ts": "1234567890.123456"}
        formatted = slack_notify.format_response(response, verbose=True)
        # Should be full JSON
        self.assertIn('"ok": true', formatted)


class TestMainFunction(unittest.TestCase):
    """Integration tests for the main function."""

    @patch('slack_notify.send_api_message')
    @patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"})
    def test_main_success(self, mock_send):
        """Test successful main execution."""
        mock_send.return_value = {"ok": True, "channel": "C12345", "ts": "123.456"}

        with patch('sys.argv', ['slack_notify', '-c', '#general', '-m', 'Hello']):
            exit_code = slack_notify.main()

        self.assertEqual(exit_code, slack_notify.EXIT_SUCCESS)
        mock_send.assert_called_once()

    @patch('slack_notify.send_webhook_message')
    @patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"})
    def test_main_webhook_success(self, mock_send):
        """Test successful webhook execution."""
        mock_send.return_value = {"ok": True}

        with patch('sys.argv', ['slack_notify', '--webhook', '-m', 'Alert']):
            exit_code = slack_notify.main()

        self.assertEqual(exit_code, slack_notify.EXIT_SUCCESS)
        mock_send.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    def test_main_missing_token(self):
        """Test missing bot token error."""
        # Clear environment
        for key in ["SLACK_BOT_TOKEN", "SLACK_WEBHOOK_URL"]:
            os.environ.pop(key, None)

        with patch('sys.argv', ['slack_notify', '-c', '#general', '-m', 'Hello']):
            exit_code = slack_notify.main()

        self.assertEqual(exit_code, slack_notify.EXIT_MISSING_CREDENTIALS)

    @patch.dict(os.environ, {}, clear=True)
    def test_main_missing_webhook(self):
        """Test missing webhook URL error."""
        for key in ["SLACK_BOT_TOKEN", "SLACK_WEBHOOK_URL"]:
            os.environ.pop(key, None)

        with patch('sys.argv', ['slack_notify', '--webhook', '-m', 'Hello']):
            exit_code = slack_notify.main()

        self.assertEqual(exit_code, slack_notify.EXIT_MISSING_CREDENTIALS)

    @patch('slack_notify.send_api_message')
    @patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"})
    def test_main_api_error(self, mock_send):
        """Test API error handling in main."""
        mock_send.return_value = {"ok": False, "error": "channel_not_found"}

        with patch('sys.argv', ['slack_notify', '-c', '#nonexistent', '-m', 'Hello']):
            exit_code = slack_notify.main()

        self.assertEqual(exit_code, slack_notify.EXIT_API_ERROR)

    @patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"})
    def test_main_dry_run(self):
        """Test dry run mode."""
        with patch('sys.argv', ['slack_notify', '-c', '#general', '-m', 'Hello', '--dry-run']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                exit_code = slack_notify.main()

        self.assertEqual(exit_code, slack_notify.EXIT_SUCCESS)
        output = mock_stdout.getvalue()
        self.assertIn("Dry run", output)
        self.assertIn("general", output)

    @patch('slack_notify.send_api_message')
    @patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"})
    def test_main_quiet_mode(self, mock_send):
        """Test quiet mode suppresses output."""
        mock_send.return_value = {"ok": True}

        with patch('sys.argv', ['slack_notify', '-c', '#general', '-m', 'Hello', '-q']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                exit_code = slack_notify.main()

        self.assertEqual(exit_code, slack_notify.EXIT_SUCCESS)
        self.assertEqual(mock_stdout.getvalue(), "")


class TestJSONHelpers(unittest.TestCase):
    """Tests for JSON helper functions."""

    def test_parse_json_string_valid(self):
        """Test parsing valid JSON string."""
        result = slack_notify.parse_json_string('{"key": "value"}', "test")
        self.assertEqual(result, {"key": "value"})

    def test_parse_json_string_invalid(self):
        """Test parsing invalid JSON string raises error."""
        with self.assertRaises(SystemExit) as cm:
            slack_notify.parse_json_string('invalid', "test")
        self.assertEqual(cm.exception.code, slack_notify.EXIT_INVALID_ARGS)

    def test_load_json_file_valid(self):
        """Test loading valid JSON file."""
        json_content = '{"blocks": []}'
        with patch("builtins.open", mock_open(read_data=json_content)):
            result = slack_notify.load_json_file("/path/to/file.json")
        self.assertEqual(result, {"blocks": []})

    def test_load_json_file_not_found(self):
        """Test loading nonexistent file raises error."""
        with self.assertRaises(FileNotFoundError):
            slack_notify.load_json_file("/nonexistent/file.json")


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and special scenarios."""

    def test_channel_with_hash_prefix(self):
        """Test channel name with # prefix is handled correctly."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-c", "#my-channel", "-m", "test"])
        payload = slack_notify.build_payload(args)
        self.assertEqual(payload["channel"], "my-channel")

    def test_channel_without_hash_prefix(self):
        """Test channel name without # prefix."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-c", "my-channel", "-m", "test"])
        payload = slack_notify.build_payload(args)
        self.assertEqual(payload["channel"], "my-channel")

    def test_user_with_at_prefix(self):
        """Test user ID with @ prefix is handled correctly."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-u", "@john.doe", "-m", "test"])
        payload = slack_notify.build_payload(args)
        self.assertEqual(payload["channel"], "john.doe")

    def test_channel_id_format(self):
        """Test channel ID format (C1234567890)."""
        parser = slack_notify.create_parser()
        args = parser.parse_args(["-c", "C1234567890", "-m", "test"])
        payload = slack_notify.build_payload(args)
        self.assertEqual(payload["channel"], "C1234567890")

    def test_attachments_from_file(self):
        """Test loading attachments from file."""
        parser = slack_notify.create_parser()
        args = parser.parse_args([
            "-c", "#general",
            "-m", "test",
            "--attachments-file", "/path/to/attachments.json"
        ])

        attachments_content = '[{"color":"#36a64f","text":"Attachment text"}]'
        with patch("builtins.open", mock_open(read_data=attachments_content)):
            payload = slack_notify.build_payload(args)

        self.assertEqual(payload["attachments"], json.loads(attachments_content))

    def test_attachments_inline_json(self):
        """Test inline attachments JSON."""
        parser = slack_notify.create_parser()
        attachments_json = '[{"color":"#ff0000","text":"Alert"}]'
        args = parser.parse_args([
            "-c", "#general",
            "-m", "test",
            "--attachments", attachments_json
        ])

        payload = slack_notify.build_payload(args)
        self.assertEqual(payload["attachments"], json.loads(attachments_json))

    def test_message_with_blocks_as_fallback(self):
        """Test message serves as fallback when blocks present."""
        parser = slack_notify.create_parser()
        blocks_json = '[{"type":"section","text":{"type":"mrkdwn","text":"Rich text"}}]'
        args = parser.parse_args([
            "-c", "#general",
            "-m", "Fallback text",
            "--blocks", blocks_json
        ])

        payload = slack_notify.build_payload(args)
        self.assertEqual(payload["text"], "Fallback text")
        self.assertEqual(payload["blocks"], json.loads(blocks_json))


if __name__ == "__main__":
    unittest.main()
