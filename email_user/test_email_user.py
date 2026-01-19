#!/usr/bin/env python3
"""Tests for email_user CLI tool."""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Import the module under test
import email_user


class TestGetConfig:
    """Tests for get_config function."""

    def test_valid_config(self, monkeypatch):
        """Should return EmailConfig when all env vars are set."""
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        monkeypatch.setenv("EMAIL_RECIPIENT", "user@example.com")
        monkeypatch.setenv("EMAIL_SENDER", "sender@example.com")

        config = email_user.get_config()

        assert config.api_key == "test-api-key"
        assert config.recipient == "user@example.com"
        assert config.sender == "sender@example.com"

    def test_missing_api_key(self, monkeypatch):
        """Should exit with code 1 when API key is missing."""
        monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
        monkeypatch.setenv("EMAIL_RECIPIENT", "user@example.com")
        monkeypatch.setenv("EMAIL_SENDER", "sender@example.com")

        with pytest.raises(SystemExit) as exc_info:
            email_user.get_config()
        assert exc_info.value.code == 1

    def test_missing_recipient(self, monkeypatch):
        """Should exit with code 1 when recipient is missing."""
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        monkeypatch.delenv("EMAIL_RECIPIENT", raising=False)
        monkeypatch.setenv("EMAIL_SENDER", "sender@example.com")

        with pytest.raises(SystemExit) as exc_info:
            email_user.get_config()
        assert exc_info.value.code == 1

    def test_missing_all_vars(self, monkeypatch):
        """Should exit with code 1 when all vars are missing."""
        monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
        monkeypatch.delenv("EMAIL_RECIPIENT", raising=False)
        monkeypatch.delenv("EMAIL_SENDER", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            email_user.get_config()
        assert exc_info.value.code == 1


class TestSendEmail:
    """Tests for send_email function."""

    def test_send_plain_text_email(self, monkeypatch):
        """Should send plain text email successfully."""
        config = email_user.EmailConfig(
            api_key="test-key",
            sender="sender@example.com",
            recipient="user@example.com"
        )

        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.headers = {"X-Message-Id": "msg-123"}

        mock_sg = Mock()
        mock_sg.send.return_value = mock_response

        with patch("email_user.SendGridAPIClient", return_value=mock_sg):
            result = email_user.send_email(
                config=config,
                subject="Test Subject",
                body="Test body content"
            )

        assert result["status_code"] == 202
        assert result["message_id"] == "msg-123"
        assert result["recipient"] == "user@example.com"
        mock_sg.send.assert_called_once()

    def test_send_html_email(self, monkeypatch):
        """Should send HTML email when html=True."""
        config = email_user.EmailConfig(
            api_key="test-key",
            sender="sender@example.com",
            recipient="user@example.com"
        )

        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.headers = {}

        mock_sg = Mock()
        mock_sg.send.return_value = mock_response

        with patch("email_user.SendGridAPIClient", return_value=mock_sg):
            result = email_user.send_email(
                config=config,
                subject="Test Subject",
                body="<h1>HTML Content</h1>",
                html=True
            )

        assert result["status_code"] == 202
        mock_sg.send.assert_called_once()

    def test_send_high_priority_email(self, monkeypatch):
        """Should set priority headers for high priority emails."""
        config = email_user.EmailConfig(
            api_key="test-key",
            sender="sender@example.com",
            recipient="user@example.com"
        )

        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.headers = {}

        mock_sg = Mock()
        mock_sg.send.return_value = mock_response

        with patch("email_user.SendGridAPIClient", return_value=mock_sg):
            result = email_user.send_email(
                config=config,
                subject="Urgent",
                body="Important message",
                priority="high"
            )

        assert result["status_code"] == 202


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


class TestMain:
    """Tests for main function."""

    def test_main_dry_run(self, monkeypatch, capsys):
        """Should print email details in dry run mode."""
        monkeypatch.setenv("SENDGRID_API_KEY", "test-key")
        monkeypatch.setenv("EMAIL_RECIPIENT", "user@example.com")
        monkeypatch.setenv("EMAIL_SENDER", "sender@example.com")
        monkeypatch.setattr(sys, "argv", [
            "email_user", "-s", "Test Subject", "-b", "Test body", "--dry-run"
        ])

        result = email_user.main()

        assert result == 0
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "Test Subject" in captured.out
        assert "Test body" in captured.out

    def test_main_empty_body_error(self, monkeypatch, capsys):
        """Should return error code 3 for empty body."""
        monkeypatch.setenv("SENDGRID_API_KEY", "test-key")
        monkeypatch.setenv("EMAIL_RECIPIENT", "user@example.com")
        monkeypatch.setenv("EMAIL_SENDER", "sender@example.com")
        monkeypatch.setattr(sys, "argv", [
            "email_user", "-s", "Subject", "-b", "   "
        ])

        result = email_user.main()

        assert result == 3

    def test_main_successful_send(self, monkeypatch, capsys):
        """Should return 0 on successful send."""
        monkeypatch.setenv("SENDGRID_API_KEY", "test-key")
        monkeypatch.setenv("EMAIL_RECIPIENT", "user@example.com")
        monkeypatch.setenv("EMAIL_SENDER", "sender@example.com")
        monkeypatch.setattr(sys, "argv", [
            "email_user", "-s", "Subject", "-b", "Body"
        ])

        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.headers = {"X-Message-Id": "msg-123"}

        mock_sg = Mock()
        mock_sg.send.return_value = mock_response

        with patch("email_user.SendGridAPIClient", return_value=mock_sg):
            result = email_user.main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Email sent successfully" in captured.out

    def test_main_quiet_mode(self, monkeypatch, capsys):
        """Should suppress output in quiet mode."""
        monkeypatch.setenv("SENDGRID_API_KEY", "test-key")
        monkeypatch.setenv("EMAIL_RECIPIENT", "user@example.com")
        monkeypatch.setenv("EMAIL_SENDER", "sender@example.com")
        monkeypatch.setattr(sys, "argv", [
            "email_user", "-s", "Subject", "-b", "Body", "-q"
        ])

        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.headers = {}

        mock_sg = Mock()
        mock_sg.send.return_value = mock_response

        with patch("email_user.SendGridAPIClient", return_value=mock_sg):
            result = email_user.main()

        assert result == 0
        captured = capsys.readouterr()
        assert captured.out == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
