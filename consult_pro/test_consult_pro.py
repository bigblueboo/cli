#!/usr/bin/env python3
"""Tests for consult_pro CLI tool."""

import json
import os
import sys
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Import the module under test
import consult_pro


class TestGetApiKey:
    """Tests for get_api_key function."""

    def test_returns_api_key_when_set(self, monkeypatch):
        """Should return API key from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")

        result = consult_pro.get_api_key()

        assert result == "sk-test-key-123"

    def test_exits_when_not_set(self, monkeypatch):
        """Should exit with code 1 when API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            consult_pro.get_api_key()

        assert exc_info.value.code == 1


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_seconds(self):
        """Should format seconds."""
        assert consult_pro.format_duration(30) == "30.0s"
        assert consult_pro.format_duration(5.5) == "5.5s"

    def test_minutes(self):
        """Should format minutes."""
        assert consult_pro.format_duration(120) == "2.0m"
        assert consult_pro.format_duration(90) == "1.5m"

    def test_hours(self):
        """Should format hours."""
        assert consult_pro.format_duration(3600) == "1.0h"
        assert consult_pro.format_duration(5400) == "1.5h"


class TestReasoningEffort:
    """Tests for ReasoningEffort enum."""

    def test_values(self):
        """Should have correct values."""
        assert consult_pro.ReasoningEffort.MEDIUM.value == "medium"
        assert consult_pro.ReasoningEffort.HIGH.value == "high"
        assert consult_pro.ReasoningEffort.XHIGH.value == "xhigh"


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_values(self):
        """Should have correct status values."""
        assert consult_pro.TaskStatus.QUEUED.value == "queued"
        assert consult_pro.TaskStatus.IN_PROGRESS.value == "in_progress"
        assert consult_pro.TaskStatus.COMPLETED.value == "completed"
        assert consult_pro.TaskStatus.FAILED.value == "failed"
        assert consult_pro.TaskStatus.CANCELLED.value == "cancelled"


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_creation(self):
        """Should create TaskResult with all fields."""
        now = datetime.now()
        result = consult_pro.TaskResult(
            id="resp_123",
            status="completed",
            model="gpt-5.2-pro",
            output_text="Response text",
            usage={"total_tokens": 100},
            created_at=now,
            completed_at=now,
            duration_seconds=30.5,
            error=None,
        )

        assert result.id == "resp_123"
        assert result.status == "completed"
        assert result.model == "gpt-5.2-pro"
        assert result.output_text == "Response text"
        assert result.duration_seconds == 30.5


class TestCreateBackgroundTask:
    """Tests for create_background_task function."""

    def test_creates_task_with_basic_params(self):
        """Should create background task with basic parameters."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.id = "resp_abc123"
        mock_response.status = "queued"
        mock_client.responses.create.return_value = mock_response

        result = consult_pro.create_background_task(
            client=mock_client,
            prompt="Test prompt",
            model="gpt-5.2-pro",
        )

        assert result.id == "resp_abc123"
        mock_client.responses.create.assert_called_once()

        # Verify background=True was passed
        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["background"] is True
        assert call_kwargs["store"] is True
        assert call_kwargs["model"] == "gpt-5.2-pro"

    def test_creates_task_with_reasoning_effort(self):
        """Should include reasoning effort for pro models."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.id = "resp_123"
        mock_response.status = "queued"
        mock_client.responses.create.return_value = mock_response

        consult_pro.create_background_task(
            client=mock_client,
            prompt="Test",
            model="gpt-5.2-pro",
            reasoning_effort="xhigh",
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["reasoning"] == {"effort": "xhigh"}

    def test_creates_task_with_system_prompt(self):
        """Should include system prompt in messages."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.id = "resp_123"
        mock_response.status = "queued"
        mock_client.responses.create.return_value = mock_response

        consult_pro.create_background_task(
            client=mock_client,
            prompt="User prompt",
            model="gpt-5.2-pro",
            system_prompt="You are helpful",
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        messages = call_kwargs["input"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"
        assert messages[1]["role"] == "user"

    def test_creates_task_with_optional_params(self):
        """Should include temperature and max_tokens when provided."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.id = "resp_123"
        mock_response.status = "queued"
        mock_client.responses.create.return_value = mock_response

        consult_pro.create_background_task(
            client=mock_client,
            prompt="Test",
            model="gpt-5.2",
            temperature=0.7,
            max_tokens=1000,
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_output_tokens"] == 1000

    def test_skips_reasoning_for_non_pro_models(self):
        """Should not include reasoning effort for non-pro models."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.id = "resp_123"
        mock_response.status = "queued"
        mock_client.responses.create.return_value = mock_response

        consult_pro.create_background_task(
            client=mock_client,
            prompt="Test",
            model="gpt-5.2",  # Not a pro model
            reasoning_effort="high",
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "reasoning" not in call_kwargs


class TestPollTask:
    """Tests for poll_task function."""

    def test_returns_immediately_on_completed(self):
        """Should return immediately if task is already completed."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status = "completed"
        mock_response.output_text = "Done"
        mock_client.responses.retrieve.return_value = mock_response

        results = list(consult_pro.poll_task(mock_client, "resp_123", quiet=True))

        assert len(results) == 1
        assert results[0].status == "completed"
        mock_client.responses.retrieve.assert_called_once_with("resp_123")

    def test_polls_until_completion(self):
        """Should poll until task completes."""
        mock_client = Mock()

        # First call: queued, second call: in_progress, third call: completed
        responses = [
            Mock(status="queued"),
            Mock(status="in_progress"),
            Mock(status="completed", output_text="Done"),
        ]
        mock_client.responses.retrieve.side_effect = responses

        with patch("consult_pro.time.sleep"):  # Skip actual sleeping
            results = list(consult_pro.poll_task(mock_client, "resp_123", quiet=True))

        assert len(results) == 3
        assert results[-1].status == "completed"
        assert mock_client.responses.retrieve.call_count == 3

    def test_returns_on_failed_status(self):
        """Should return when task fails."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status = "failed"
        mock_response.error = "Something went wrong"
        mock_client.responses.retrieve.return_value = mock_response

        results = list(consult_pro.poll_task(mock_client, "resp_123", quiet=True))

        assert results[-1].status == "failed"


class TestRunTaskAndWait:
    """Tests for run_task_and_wait function."""

    def test_successful_completion(self):
        """Should return TaskResult on successful completion."""
        mock_client = Mock()

        # Mock create response
        create_response = Mock()
        create_response.id = "resp_123"
        create_response.status = "queued"
        mock_client.responses.create.return_value = create_response

        # Mock retrieve response (completed immediately)
        retrieve_response = Mock()
        retrieve_response.status = "completed"
        retrieve_response.output_text = "Final output"
        retrieve_response.usage = Mock()
        retrieve_response.usage.input_tokens = 10
        retrieve_response.usage.output_tokens = 50
        retrieve_response.usage.total_tokens = 60
        mock_client.responses.retrieve.return_value = retrieve_response

        with patch("consult_pro.time.sleep"):
            result = consult_pro.run_task_and_wait(
                client=mock_client,
                prompt="Test prompt",
                quiet=True,
            )

        assert result.id == "resp_123"
        assert result.status == "completed"
        assert result.output_text == "Final output"

    def test_handles_failed_task(self):
        """Should return error info on failed task."""
        mock_client = Mock()

        create_response = Mock()
        create_response.id = "resp_123"
        create_response.status = "queued"
        mock_client.responses.create.return_value = create_response

        retrieve_response = Mock()
        retrieve_response.status = "failed"
        retrieve_response.output_text = None
        retrieve_response.usage = None
        retrieve_response.error = "API error occurred"
        mock_client.responses.retrieve.return_value = retrieve_response

        with patch("consult_pro.time.sleep"):
            result = consult_pro.run_task_and_wait(
                client=mock_client,
                prompt="Test",
                quiet=True,
            )

        assert result.status == "failed"
        assert result.error == "API error occurred"


class TestStreamBackgroundTask:
    """Tests for stream_background_task function."""

    def test_streams_chunks(self):
        """Should yield text chunks from stream."""
        mock_client = Mock()

        # Mock streaming response
        chunk1 = Mock(delta="Hello ")
        chunk2 = Mock(delta="world!")
        mock_client.responses.create.return_value = [chunk1, chunk2]

        chunks = list(consult_pro.stream_background_task(
            client=mock_client,
            prompt="Test",
        ))

        assert chunks == ["Hello ", "world!"]

        # Verify stream=True was passed
        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["background"] is True


class TestCreateParser:
    """Tests for argument parser."""

    def test_prompt_argument(self):
        """Should accept prompt as positional argument."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["Test prompt"])

        assert args.prompt == "Test prompt"

    def test_file_argument(self):
        """Should accept -f/--file argument."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["-f", "prompt.txt"])

        assert args.file == Path("prompt.txt")

    def test_stdin_flag(self):
        """Should accept --stdin flag."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["--stdin"])

        assert args.stdin is True

    def test_model_argument(self):
        """Should accept -m/--model argument."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["prompt", "-m", "gpt-5.2"])

        assert args.model == "gpt-5.2"

    def test_effort_argument(self):
        """Should accept -e/--effort argument."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["prompt", "-e", "xhigh"])

        assert args.effort == "xhigh"

    def test_effort_choices(self):
        """Should only accept valid effort choices."""
        parser = consult_pro.create_parser()

        for effort in ["medium", "high", "xhigh"]:
            args = parser.parse_args(["prompt", "--effort", effort])
            assert args.effort == effort

    def test_system_argument(self):
        """Should accept -s/--system argument."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["prompt", "-s", "You are helpful"])

        assert args.system == "You are helpful"

    def test_temperature_argument(self):
        """Should accept -t/--temperature argument."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["prompt", "-t", "0.7"])

        assert args.temperature == 0.7

    def test_max_tokens_argument(self):
        """Should accept --max-tokens argument."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["prompt", "--max-tokens", "1000"])

        assert args.max_tokens == 1000

    def test_no_wait_flag(self):
        """Should accept --no-wait flag."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["prompt", "--no-wait"])

        assert args.no_wait is True

    def test_stream_flag(self):
        """Should accept --stream flag."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["prompt", "--stream"])

        assert args.stream is True

    def test_poll_id_argument(self):
        """Should accept --poll-id argument."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["--poll-id", "resp_123"])

        assert args.poll_id == "resp_123"

    def test_output_argument(self):
        """Should accept -o/--output argument."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["prompt", "-o", "output.txt"])

        assert args.output == Path("output.txt")

    def test_json_flag(self):
        """Should accept --json flag."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["prompt", "--json"])

        assert args.json is True

    def test_verbose_flag(self):
        """Should accept -v/--verbose flag."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["prompt", "-v"])

        assert args.verbose is True

    def test_quiet_flag(self):
        """Should accept -q/--quiet flag."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["prompt", "-q"])

        assert args.quiet is True

    def test_default_model(self):
        """Should use default model."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["prompt"])

        assert args.model == consult_pro.DEFAULT_MODEL

    def test_default_effort(self):
        """Should use default effort."""
        parser = consult_pro.create_parser()
        args = parser.parse_args(["prompt"])

        assert args.effort == "high"


class TestMain:
    """Tests for main function."""

    def test_missing_prompt_error(self, monkeypatch, capsys):
        """Should return error when no prompt provided."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(sys, "argv", ["consult_pro"])
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        with patch.object(consult_pro, "OPENAI_AVAILABLE", True), \
             patch.object(consult_pro, "OpenAI", Mock()):
            result = consult_pro.main()

        assert result == 4
        captured = capsys.readouterr()
        assert "No prompt provided" in captured.err

    def test_file_not_found_error(self, monkeypatch, capsys):
        """Should return error when file not found."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(sys, "argv", ["consult_pro", "-f", "/nonexistent/file.txt"])

        with patch.object(consult_pro, "OPENAI_AVAILABLE", True), \
             patch.object(consult_pro, "OpenAI", Mock()):
            result = consult_pro.main()

        assert result == 4
        captured = capsys.readouterr()
        assert "File not found" in captured.err

    def test_successful_no_wait_mode(self, monkeypatch, capsys):
        """Should return task ID in no-wait mode."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(sys, "argv", ["consult_pro", "Test prompt", "--no-wait"])

        mock_client = Mock()
        mock_response = Mock()
        mock_response.id = "resp_test_123"
        mock_response.status = "queued"
        mock_client.responses.create.return_value = mock_response

        with patch.object(consult_pro, "OPENAI_AVAILABLE", True), \
             patch("consult_pro.OpenAI", return_value=mock_client):
            result = consult_pro.main()

        assert result == 0
        captured = capsys.readouterr()
        assert "resp_test_123" in captured.out

    def test_successful_completion(self, monkeypatch, capsys):
        """Should print output on successful completion."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(sys, "argv", ["consult_pro", "Test prompt", "-q"])

        mock_client = Mock()

        # Create response
        create_resp = Mock()
        create_resp.id = "resp_123"
        create_resp.status = "completed"
        create_resp.output_text = "This is the response"
        create_resp.usage = Mock()
        create_resp.usage.input_tokens = 10
        create_resp.usage.output_tokens = 20
        create_resp.usage.total_tokens = 30
        mock_client.responses.create.return_value = create_resp

        with patch.object(consult_pro, "OPENAI_AVAILABLE", True), \
             patch("consult_pro.OpenAI", return_value=mock_client):
            result = consult_pro.main()

        assert result == 0
        captured = capsys.readouterr()
        assert "This is the response" in captured.out

    def test_json_output_mode(self, monkeypatch, capsys):
        """Should output JSON when --json flag is used."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(sys, "argv", ["consult_pro", "Test", "--no-wait", "--json"])

        mock_client = Mock()
        mock_response = Mock()
        mock_response.id = "resp_json_test"
        mock_response.status = "queued"
        mock_client.responses.create.return_value = mock_response

        with patch.object(consult_pro, "OPENAI_AVAILABLE", True), \
             patch("consult_pro.OpenAI", return_value=mock_client):
            result = consult_pro.main()

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["id"] == "resp_json_test"
        assert output["status"] == "queued"

    def test_poll_existing_task(self, monkeypatch, capsys):
        """Should poll existing task by ID."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(sys, "argv", ["consult_pro", "--poll-id", "resp_existing", "-q"])

        mock_client = Mock()
        mock_response = Mock()
        mock_response.status = "completed"
        mock_response.output_text = "Polled result"
        mock_client.responses.retrieve.return_value = mock_response

        with patch.object(consult_pro, "OPENAI_AVAILABLE", True), \
             patch("consult_pro.OpenAI", return_value=mock_client):
            result = consult_pro.main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Polled result" in captured.out

    def test_failed_task_returns_error_code(self, monkeypatch, capsys):
        """Should return error code 3 when task fails."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(sys, "argv", ["consult_pro", "Test", "-q"])

        mock_client = Mock()

        create_resp = Mock()
        create_resp.id = "resp_123"
        create_resp.status = "failed"
        create_resp.output_text = None
        create_resp.usage = None
        create_resp.error = "Task failed"
        mock_client.responses.create.return_value = create_resp

        with patch.object(consult_pro, "OPENAI_AVAILABLE", True), \
             patch("consult_pro.OpenAI", return_value=mock_client):
            result = consult_pro.main()

        assert result == 3


class TestConstants:
    """Tests for module constants."""

    def test_default_model(self):
        """Should have correct default model."""
        assert consult_pro.DEFAULT_MODEL == "gpt-5.2-pro"

    def test_reasoning_models(self):
        """Should include correct reasoning models."""
        assert "gpt-5.2-pro" in consult_pro.REASONING_MODELS
        assert "o3-pro" in consult_pro.REASONING_MODELS
        assert "o3" in consult_pro.REASONING_MODELS

    def test_poll_intervals(self):
        """Should have reasonable poll intervals."""
        assert consult_pro.INITIAL_POLL_INTERVAL > 0
        assert consult_pro.MAX_POLL_INTERVAL > consult_pro.INITIAL_POLL_INTERVAL
        assert consult_pro.POLL_BACKOFF_FACTOR > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
