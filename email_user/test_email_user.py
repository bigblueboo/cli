#!/usr/bin/env python3
"""Tests for email_user CLI tool."""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
import email_user


class TestExitCodes:
    """Tests for exit code enum."""

    def test_exit_code_values(self):
        """Should have correct exit code values."""
        assert email_user.ExitCode.SUCCESS == 0
        assert email_user.ExitCode.CONFIG_ERROR == 1
        assert email_user.ExitCode.API_ERROR == 2
        assert email_user.ExitCode.INVALID_INPUT == 3
        assert email_user.ExitCode.NETWORK_ERROR == 4


class TestExceptions:
    """Tests for custom exception classes."""

    def test_email_error_base(self):
        """Should create base exception with exit code."""
        error = email_user.EmailError("test error", email_user.ExitCode.API_ERROR)
        assert str(error) == "test error"
        assert error.exit_code == email_user.ExitCode.API_ERROR

    def test_config_error(self):
        """Should create config error with correct exit code."""
        error = email_user.ConfigError("missing key")
        assert str(error) == "missing key"
        assert error.exit_code == email_user.ExitCode.CONFIG_ERROR

    def test_api_error(self):
        """Should create API error with correct exit code."""
        error = email_user.APIError("rate limit")
        assert str(error) == "rate limit"
        assert error.exit_code == email_user.ExitCode.API_ERROR

    def test_input_error(self):
        """Should create input error with correct exit code."""
        error = email_user.InputError("invalid subject")
        assert str(error) == "invalid subject"
        assert error.exit_code == email_user.ExitCode.INVALID_INPUT

    def test_network_error(self):
        """Should create network error with correct exit code."""
        error = email_user.NetworkError("connection failed")
        assert str(error) == "connection failed"
        assert error.exit_code == email_user.ExitCode.NETWORK_ERROR


class TestConstants:
    """Tests for module constants."""

    def test_sender_email(self):
        """Should have correct sender email."""
        assert email_user.SENDER_EMAIL == "charlie@charliedeck.com"

    def test_recipient_email(self):
        """Should have correct recipient email."""
        assert email_user.RECIPIENT_EMAIL == "charliedeck@gmail.com"


class TestGetApiKey:
    """Tests for get_api_key function."""

    def test_valid_api_key(self, monkeypatch):
        """Should return API key when properly set."""
        monkeypatch.setenv("RESEND_API_KEY", "re_test_api_key_123")

        api_key = email_user.get_api_key()

        assert api_key == "re_test_api_key_123"

    def test_missing_api_key(self, monkeypatch):
        """Should raise ConfigError when API key is missing."""
        monkeypatch.delenv("RESEND_API_KEY", raising=False)

        with pytest.raises(email_user.ConfigError) as exc_info:
            email_user.get_api_key()

        assert "RESEND_API_KEY" in str(exc_info.value)
        assert exc_info.value.exit_code == email_user.ExitCode.CONFIG_ERROR

    def test_invalid_api_key_format(self, monkeypatch):
        """Should raise ConfigError when API key has invalid format."""
        monkeypatch.setenv("RESEND_API_KEY", "invalid_key_without_prefix")

        with pytest.raises(email_user.ConfigError) as exc_info:
            email_user.get_api_key()

        assert "re_" in str(exc_info.value)
        assert exc_info.value.exit_code == email_user.ExitCode.CONFIG_ERROR


class TestValidateInput:
    """Tests for validate_input function."""

    def test_valid_input(self):
        """Should not raise for valid input."""
        email_user.validate_input("Test Subject", "Test body")

    def test_empty_subject(self):
        """Should raise InputError for empty subject."""
        with pytest.raises(email_user.InputError) as exc_info:
            email_user.validate_input("", "Test body")
        assert "subject" in str(exc_info.value).lower()

    def test_whitespace_subject(self):
        """Should raise InputError for whitespace-only subject."""
        with pytest.raises(email_user.InputError) as exc_info:
            email_user.validate_input("   ", "Test body")
        assert "subject" in str(exc_info.value).lower()

    def test_empty_body(self):
        """Should raise InputError for empty body."""
        with pytest.raises(email_user.InputError) as exc_info:
            email_user.validate_input("Subject", "")
        assert "body" in str(exc_info.value).lower()

    def test_whitespace_body(self):
        """Should raise InputError for whitespace-only body."""
        with pytest.raises(email_user.InputError) as exc_info:
            email_user.validate_input("Subject", "   \n\t  ")
        assert "body" in str(exc_info.value).lower()

    def test_subject_too_long(self):
        """Should raise InputError when subject exceeds RFC limit."""
        long_subject = "x" * 999
        with pytest.raises(email_user.InputError) as exc_info:
            email_user.validate_input(long_subject, "Body")
        assert "length" in str(exc_info.value).lower()

    def test_body_too_large(self):
        """Should raise InputError when body exceeds size limit."""
        large_body = "x" * (10_000_001)
        with pytest.raises(email_user.InputError) as exc_info:
            email_user.validate_input("Subject", large_body)
        assert "size" in str(exc_info.value).lower()


class TestEmailResult:
    """Tests for EmailResult dataclass."""

    def test_successful_result(self):
        """Should create successful result with message ID."""
        result = email_user.EmailResult(success=True, message_id="msg-123")
        assert result.success is True
        assert result.message_id == "msg-123"
        assert result.error is None

    def test_failed_result(self):
        """Should create failed result with error."""
        result = email_user.EmailResult(success=False, error="API error")
        assert result.success is False
        assert result.message_id is None
        assert result.error == "API error"


class TestSendEmail:
    """Tests for send_email function."""

    def test_send_plain_text_email(self):
        """Should send plain text email successfully."""
        mock_response = Mock()
        mock_response.id = "msg-123"

        mock_resend = Mock()
        mock_resend.Emails.send.return_value = mock_response
        mock_resend.exceptions.ResendError = Exception

        with patch.object(email_user, "resend", mock_resend):
            result = email_user.send_email(
                api_key="re_test_key",
                subject="Test Subject",
                body="Test body content"
            )

        assert result.success is True
        assert result.message_id == "msg-123"
        mock_resend.Emails.send.assert_called_once()

        # Verify params
        call_args = mock_resend.Emails.send.call_args[0][0]
        assert call_args["subject"] == "Test Subject"
        assert call_args["text"] == "Test body content"
        assert "html" not in call_args

    def test_send_html_email(self):
        """Should send HTML email when html=True."""
        mock_response = Mock()
        mock_response.id = "msg-456"

        mock_resend = Mock()
        mock_resend.Emails.send.return_value = mock_response
        mock_resend.exceptions.ResendError = Exception

        with patch.object(email_user, "resend", mock_resend):
            result = email_user.send_email(
                api_key="re_test_key",
                subject="Test Subject",
                body="<h1>HTML Content</h1>",
                html=True
            )

        assert result.success is True
        call_args = mock_resend.Emails.send.call_args[0][0]
        assert call_args["html"] == "<h1>HTML Content</h1>"
        assert "text" not in call_args

    def test_send_high_priority_email(self):
        """Should set priority headers for high priority emails."""
        mock_response = Mock()
        mock_response.id = "msg-789"

        mock_resend = Mock()
        mock_resend.Emails.send.return_value = mock_response
        mock_resend.exceptions.ResendError = Exception

        with patch.object(email_user, "resend", mock_resend):
            result = email_user.send_email(
                api_key="re_test_key",
                subject="Urgent",
                body="Important message",
                priority="high"
            )

        call_args = mock_resend.Emails.send.call_args[0][0]
        assert call_args["headers"]["X-Priority"] == "1"
        assert call_args["headers"]["Importance"] == "high"

    def test_send_low_priority_email(self):
        """Should set priority headers for low priority emails."""
        mock_response = Mock()
        mock_response.id = "msg-low"

        mock_resend = Mock()
        mock_resend.Emails.send.return_value = mock_response
        mock_resend.exceptions.ResendError = Exception

        with patch.object(email_user, "resend", mock_resend):
            result = email_user.send_email(
                api_key="re_test_key",
                subject="FYI",
                body="Low priority message",
                priority="low"
            )

        call_args = mock_resend.Emails.send.call_args[0][0]
        assert call_args["headers"]["X-Priority"] == "5"
        assert call_args["headers"]["Importance"] == "low"

    def test_send_with_tags(self):
        """Should include tags in email params."""
        mock_response = Mock()
        mock_response.id = "msg-tags"

        mock_resend = Mock()
        mock_resend.Emails.send.return_value = mock_response
        mock_resend.exceptions.ResendError = Exception

        with patch.object(email_user, "resend", mock_resend):
            result = email_user.send_email(
                api_key="re_test_key",
                subject="Tagged",
                body="Message with tags",
                tags=["deploy", "production"]
            )

        call_args = mock_resend.Emails.send.call_args[0][0]
        assert len(call_args["tags"]) == 2
        assert call_args["tags"][0]["value"] == "deploy"
        assert call_args["tags"][1]["value"] == "production"

    def test_send_api_error_auth(self):
        """Should raise APIError on authentication failure."""
        mock_resend = Mock()
        mock_resend.exceptions.ResendError = type("ResendError", (Exception,), {})
        mock_resend.Emails.send.side_effect = mock_resend.exceptions.ResendError("401 unauthorized")

        with patch.object(email_user, "resend", mock_resend):
            with pytest.raises(email_user.APIError) as exc_info:
                email_user.send_email(
                    api_key="re_invalid_key",
                    subject="Test",
                    body="Test"
                )
            assert "Authentication" in str(exc_info.value)

    def test_send_api_error_rate_limit(self):
        """Should raise APIError on rate limit."""
        mock_resend = Mock()
        mock_resend.exceptions.ResendError = type("ResendError", (Exception,), {})
        mock_resend.Emails.send.side_effect = mock_resend.exceptions.ResendError("429 rate limit exceeded")

        with patch.object(email_user, "resend", mock_resend):
            with pytest.raises(email_user.APIError) as exc_info:
                email_user.send_email(
                    api_key="re_test_key",
                    subject="Test",
                    body="Test"
                )
            assert "Rate limit" in str(exc_info.value)

    def test_send_network_error(self):
        """Should raise NetworkError on connection failure."""
        mock_resend = Mock()
        mock_resend.exceptions.ResendError = type("ResendError", (Exception,), {})
        mock_resend.Emails.send.side_effect = ConnectionError("Connection refused")

        with patch.object(email_user, "resend", mock_resend):
            with pytest.raises(email_user.NetworkError) as exc_info:
                email_user.send_email(
                    api_key="re_test_key",
                    subject="Test",
                    body="Test"
                )
            assert "Network error" in str(exc_info.value)

    def test_send_dict_response(self):
        """Should handle dict response format."""
        mock_response = {"id": "msg-dict-123"}

        mock_resend = Mock()
        mock_resend.Emails.send.return_value = mock_response
        mock_resend.exceptions.ResendError = Exception

        with patch.object(email_user, "resend", mock_resend):
            result = email_user.send_email(
                api_key="re_test_key",
                subject="Test",
                body="Test"
            )

        assert result.success is True
        assert result.message_id == "msg-dict-123"


class TestCreateParser:
    """Tests for argument parser."""

    def test_parser_requires_subject(self):
        """Should require --subject argument."""
        parser = email_user.create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_accepts_short_flags(self):
        """Should accept -s and -b short flags."""
        parser = email_user.create_parser()
        args = parser.parse_args(["-s", "Subject", "-b", "Body"])

        assert args.subject == "Subject"
        assert args.body == "Body"

    def test_parser_accepts_long_flags(self):
        """Should accept --subject and --body long flags."""
        parser = email_user.create_parser()
        args = parser.parse_args(["--subject", "Subject", "--body", "Body"])

        assert args.subject == "Subject"
        assert args.body == "Body"

    def test_parser_html_flag(self):
        """Should parse --html flag."""
        parser = email_user.create_parser()
        args = parser.parse_args(["-s", "Sub", "-b", "Body", "--html"])

        assert args.html is True

    def test_parser_priority_choices(self):
        """Should accept valid priority values."""
        parser = email_user.create_parser()

        for priority in ["low", "normal", "high"]:
            args = parser.parse_args(["-s", "Sub", "-b", "Body", "--priority", priority])
            assert args.priority == priority

    def test_parser_tag_accumulation(self):
        """Should accumulate multiple --tag flags."""
        parser = email_user.create_parser()
        args = parser.parse_args(["-s", "Sub", "-b", "Body", "--tag", "a", "--tag", "b"])

        assert args.tags == ["a", "b"]

    def test_parser_dry_run_flag(self):
        """Should parse --dry-run flag."""
        parser = email_user.create_parser()
        args = parser.parse_args(["-s", "Sub", "-b", "Body", "--dry-run"])

        assert args.dry_run is True

    def test_parser_quiet_flag(self):
        """Should parse --quiet flag."""
        parser = email_user.create_parser()
        args = parser.parse_args(["-s", "Sub", "-b", "Body", "-q"])

        assert args.quiet is True

    def test_parser_verbose_flag(self):
        """Should parse --verbose flag."""
        parser = email_user.create_parser()
        args = parser.parse_args(["-s", "Sub", "-b", "Body", "-v"])

        assert args.verbose is True

    def test_parser_default_priority(self):
        """Should default priority to normal."""
        parser = email_user.create_parser()
        args = parser.parse_args(["-s", "Sub", "-b", "Body"])

        assert args.priority == "normal"


class TestMain:
    """Tests for main function."""

    def test_main_missing_resend_package(self, monkeypatch, capsys):
        """Should return config error when resend is not available."""
        monkeypatch.setattr(email_user, "RESEND_AVAILABLE", False)

        result = email_user.main(["-s", "Subject", "-b", "Body"])

        assert result == email_user.ExitCode.CONFIG_ERROR
        captured = capsys.readouterr()
        assert "resend package not installed" in captured.err

    def test_main_dry_run(self, monkeypatch, capsys):
        """Should print email details in dry run mode."""
        monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
        monkeypatch.setattr(email_user, "RESEND_AVAILABLE", True)

        result = email_user.main(["-s", "Test Subject", "-b", "Test body", "--dry-run"])

        assert result == email_user.ExitCode.SUCCESS
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "Test Subject" in captured.out
        assert "Test body" in captured.out
        assert email_user.RECIPIENT_EMAIL in captured.out
        assert email_user.SENDER_EMAIL in captured.out

    def test_main_dry_run_with_tags(self, monkeypatch, capsys):
        """Should show tags in dry run mode."""
        monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
        monkeypatch.setattr(email_user, "RESEND_AVAILABLE", True)

        result = email_user.main(["-s", "Sub", "-b", "Body", "--dry-run", "--tag", "deploy"])

        assert result == email_user.ExitCode.SUCCESS
        captured = capsys.readouterr()
        assert "deploy" in captured.out

    def test_main_empty_body_error(self, monkeypatch, capsys):
        """Should return error code 3 for empty body."""
        monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
        monkeypatch.setattr(email_user, "RESEND_AVAILABLE", True)

        result = email_user.main(["-s", "Subject", "-b", "   "])

        assert result == email_user.ExitCode.INVALID_INPUT
        captured = capsys.readouterr()
        assert "body" in captured.err.lower()

    def test_main_missing_api_key(self, monkeypatch, capsys):
        """Should return config error when API key is missing."""
        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        monkeypatch.setattr(email_user, "RESEND_AVAILABLE", True)

        result = email_user.main(["-s", "Subject", "-b", "Body"])

        assert result == email_user.ExitCode.CONFIG_ERROR
        captured = capsys.readouterr()
        assert "RESEND_API_KEY" in captured.err

    def test_main_successful_send(self, monkeypatch, capsys):
        """Should return 0 on successful send."""
        monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
        monkeypatch.setattr(email_user, "RESEND_AVAILABLE", True)

        mock_response = Mock()
        mock_response.id = "msg-123"

        mock_resend = Mock()
        mock_resend.Emails.send.return_value = mock_response
        mock_resend.exceptions.ResendError = Exception

        with patch.object(email_user, "resend", mock_resend):
            result = email_user.main(["-s", "Subject", "-b", "Body"])

        assert result == email_user.ExitCode.SUCCESS
        captured = capsys.readouterr()
        assert "Email sent successfully" in captured.out

    def test_main_quiet_mode(self, monkeypatch, capsys):
        """Should suppress output in quiet mode."""
        monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
        monkeypatch.setattr(email_user, "RESEND_AVAILABLE", True)

        mock_response = Mock()
        mock_response.id = "msg-quiet"

        mock_resend = Mock()
        mock_resend.Emails.send.return_value = mock_response
        mock_resend.exceptions.ResendError = Exception

        with patch.object(email_user, "resend", mock_resend):
            result = email_user.main(["-s", "Subject", "-b", "Body", "-q"])

        assert result == email_user.ExitCode.SUCCESS
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_main_verbose_mode(self, monkeypatch, capsys):
        """Should show message ID in verbose mode."""
        monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
        monkeypatch.setattr(email_user, "RESEND_AVAILABLE", True)

        mock_response = Mock()
        mock_response.id = "msg-verbose-123"

        mock_resend = Mock()
        mock_resend.Emails.send.return_value = mock_response
        mock_resend.exceptions.ResendError = Exception

        with patch.object(email_user, "resend", mock_resend):
            result = email_user.main(["-s", "Subject", "-b", "Body", "-v"])

        assert result == email_user.ExitCode.SUCCESS
        captured = capsys.readouterr()
        assert "msg-verbose-123" in captured.out

    def test_main_api_error(self, monkeypatch, capsys):
        """Should return API error code on API failure."""
        monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
        monkeypatch.setattr(email_user, "RESEND_AVAILABLE", True)

        mock_resend = Mock()
        mock_resend.exceptions.ResendError = type("ResendError", (Exception,), {})
        mock_resend.Emails.send.side_effect = mock_resend.exceptions.ResendError("Server error")

        with patch.object(email_user, "resend", mock_resend):
            result = email_user.main(["-s", "Subject", "-b", "Body"])

        assert result == email_user.ExitCode.API_ERROR
        captured = capsys.readouterr()
        assert "API error" in captured.err

    def test_main_network_error(self, monkeypatch, capsys):
        """Should return network error code on connection failure."""
        monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
        monkeypatch.setattr(email_user, "RESEND_AVAILABLE", True)

        mock_resend = Mock()
        mock_resend.exceptions.ResendError = type("ResendError", (Exception,), {})
        mock_resend.Emails.send.side_effect = ConnectionError("No network")

        with patch.object(email_user, "resend", mock_resend):
            result = email_user.main(["-s", "Subject", "-b", "Body"])

        assert result == email_user.ExitCode.NETWORK_ERROR
        captured = capsys.readouterr()
        assert "Network error" in captured.err

    def test_main_no_body_no_stdin(self, monkeypatch, capsys):
        """Should return error when no body and stdin is tty."""
        monkeypatch.setattr(email_user, "RESEND_AVAILABLE", True)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        result = email_user.main(["-s", "Subject"])

        assert result == email_user.ExitCode.INVALID_INPUT
        captured = capsys.readouterr()
        assert "body" in captured.err.lower() or "stdin" in captured.err.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
