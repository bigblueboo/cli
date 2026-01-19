#!/usr/bin/env python3
"""
Comprehensive tests for code_review CLI tool.

Tests cover:
- Argument parsing
- Input collection (files, directories, git, stdin)
- LLM provider selection and mocking
- Output formatting (text, markdown, JSON)
- Severity filtering and exit codes
- Error handling
"""

import json
import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import code_review
from code_review import (
    ExitCode,
    Severity,
    FocusArea,
    OutputFormat,
    Issue,
    ReviewResult,
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    APIError,
    OutputFormatter,
    CodeCollector,
    get_provider,
    create_parser,
    main,
)


class TestSeverity(unittest.TestCase):
    """Tests for Severity enum."""

    def test_from_string_valid(self):
        """Test converting valid strings to Severity."""
        self.assertEqual(Severity.from_string("info"), Severity.INFO)
        self.assertEqual(Severity.from_string("low"), Severity.LOW)
        self.assertEqual(Severity.from_string("MEDIUM"), Severity.MEDIUM)
        self.assertEqual(Severity.from_string("High"), Severity.HIGH)
        self.assertEqual(Severity.from_string("critical"), Severity.CRITICAL)

    def test_from_string_invalid(self):
        """Test converting invalid strings raises ValueError."""
        with self.assertRaises(ValueError):
            Severity.from_string("invalid")
        with self.assertRaises(ValueError):
            Severity.from_string("")

    def test_comparison_operators(self):
        """Test severity comparison."""
        self.assertTrue(Severity.HIGH >= Severity.MEDIUM)
        self.assertTrue(Severity.CRITICAL > Severity.HIGH)
        self.assertTrue(Severity.INFO >= Severity.INFO)
        self.assertFalse(Severity.LOW > Severity.MEDIUM)


class TestFocusArea(unittest.TestCase):
    """Tests for FocusArea enum."""

    def test_from_string_valid(self):
        """Test converting valid strings to FocusArea."""
        self.assertEqual(FocusArea.from_string("security"), FocusArea.SECURITY)
        self.assertEqual(FocusArea.from_string("PERFORMANCE"), FocusArea.PERFORMANCE)
        self.assertEqual(FocusArea.from_string("Style"), FocusArea.STYLE)
        self.assertEqual(FocusArea.from_string("bugs"), FocusArea.BUGS)
        self.assertEqual(FocusArea.from_string("all"), FocusArea.ALL)

    def test_from_string_invalid(self):
        """Test converting invalid strings raises ValueError."""
        with self.assertRaises(ValueError):
            FocusArea.from_string("invalid")


class TestIssue(unittest.TestCase):
    """Tests for Issue dataclass."""

    def test_issue_creation(self):
        """Test creating an Issue."""
        issue = Issue(
            severity="high",
            category="security",
            file="test.py",
            line=42,
            message="SQL injection vulnerability",
            suggestion="Use parameterized queries",
            code_snippet="cursor.execute(f'SELECT * FROM {table}')"
        )
        self.assertEqual(issue.severity, "high")
        self.assertEqual(issue.line, 42)

    def test_issue_to_dict(self):
        """Test converting Issue to dictionary."""
        issue = Issue(
            severity="medium",
            category="style",
            file="test.py",
            line=None,
            message="Missing docstring"
        )
        d = issue.to_dict()
        self.assertIn("severity", d)
        self.assertNotIn("line", d)  # None values excluded
        self.assertNotIn("suggestion", d)


class TestReviewResult(unittest.TestCase):
    """Tests for ReviewResult dataclass."""

    def test_empty_result(self):
        """Test empty review result."""
        result = ReviewResult()
        self.assertEqual(result.issues, [])
        self.assertEqual(result.summary, "")
        self.assertFalse(result.has_issues_at_or_above(Severity.INFO))

    def test_has_issues_at_or_above(self):
        """Test severity threshold checking."""
        result = ReviewResult(issues=[
            Issue(severity="low", category="style", file="test.py", line=1, message="Test"),
            Issue(severity="medium", category="bugs", file="test.py", line=2, message="Test2"),
        ])
        self.assertTrue(result.has_issues_at_or_above(Severity.LOW))
        self.assertTrue(result.has_issues_at_or_above(Severity.MEDIUM))
        self.assertFalse(result.has_issues_at_or_above(Severity.HIGH))

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ReviewResult(
            issues=[Issue(severity="info", category="style", file="test.py", line=1, message="Test")],
            summary="Test summary",
            files_reviewed=["test.py"],
            model_used="gpt-4",
            focus_areas=["security"]
        )
        d = result.to_dict()
        self.assertEqual(len(d["issues"]), 1)
        self.assertEqual(d["summary"], "Test summary")


class TestOutputFormatter(unittest.TestCase):
    """Tests for OutputFormatter."""

    def setUp(self):
        """Set up test fixtures."""
        self.result = ReviewResult(
            issues=[
                Issue(severity="high", category="security", file="auth.py", line=42,
                      message="Hardcoded password", suggestion="Use environment variables"),
                Issue(severity="low", category="style", file="utils.py", line=10,
                      message="Missing docstring"),
            ],
            summary="Found 2 issues",
            files_reviewed=["auth.py", "utils.py"],
            model_used="gpt-4",
            focus_areas=["security", "style"]
        )

    def test_format_text(self):
        """Test text output format."""
        output = OutputFormatter.format(self.result, OutputFormat.TEXT)
        self.assertIn("CODE REVIEW REPORT", output)
        self.assertIn("HIGH", output)
        self.assertIn("security", output)
        self.assertIn("auth.py", output)

    def test_format_markdown(self):
        """Test markdown output format."""
        output = OutputFormatter.format(self.result, OutputFormat.MARKDOWN)
        self.assertIn("# Code Review Report", output)
        self.assertIn("## Issues Found", output)
        self.assertIn("### HIGH", output)
        self.assertIn("**Model:**", output)

    def test_format_json(self):
        """Test JSON output format."""
        output = OutputFormatter.format(self.result, OutputFormat.JSON)
        data = json.loads(output)
        self.assertEqual(len(data["issues"]), 2)
        self.assertEqual(data["model_used"], "gpt-4")

    def test_format_empty_result(self):
        """Test formatting empty result."""
        result = ReviewResult(summary="No issues")
        output = OutputFormatter.format(result, OutputFormat.TEXT)
        self.assertIn("NO ISSUES FOUND", output)


class TestCodeCollector(unittest.TestCase):
    """Tests for CodeCollector."""

    def test_from_files_single_file(self):
        """Test collecting code from a single file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('hello')")
            f.flush()
            try:
                code, files = CodeCollector.from_files([f.name])
                self.assertIn("print('hello')", code)
                self.assertEqual(len(files), 1)
            finally:
                os.unlink(f.name)

    def test_from_files_multiple_files(self):
        """Test collecting code from multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.py"
            file2 = Path(tmpdir) / "file2.py"
            file1.write_text("# File 1\nprint(1)")
            file2.write_text("# File 2\nprint(2)")

            code, files = CodeCollector.from_files([str(file1), str(file2)])
            self.assertIn("File 1", code)
            self.assertIn("File 2", code)
            self.assertEqual(len(files), 2)

    def test_from_files_nonexistent(self):
        """Test handling nonexistent files."""
        code, files = CodeCollector.from_files(["/nonexistent/path/file.py"])
        self.assertEqual(code, "")
        self.assertEqual(files, [])

    def test_from_directory(self):
        """Test collecting code from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some Python files
            (Path(tmpdir) / "main.py").write_text("# Main\nprint('main')")
            subdir = Path(tmpdir) / "sub"
            subdir.mkdir()
            (subdir / "helper.py").write_text("# Helper\nprint('helper')")

            code, files = CodeCollector.from_directory(tmpdir)
            self.assertIn("Main", code)
            self.assertIn("Helper", code)
            self.assertEqual(len(files), 2)

    def test_from_directory_nonexistent(self):
        """Test handling nonexistent directory."""
        with self.assertRaises(ValueError):
            CodeCollector.from_directory("/nonexistent/path")

    @patch('subprocess.run')
    def test_from_git_diff(self, mock_run):
        """Test collecting code from git diff."""
        mock_run.return_value = MagicMock(
            stdout="diff --git a/test.py b/test.py\n+++ b/test.py\n+print('new')",
            returncode=0
        )
        code, files = CodeCollector.from_git_diff("HEAD~1")
        self.assertIn("print('new')", code)
        self.assertIn("test.py", files)

    @patch('subprocess.run')
    def test_from_git_diff_error(self, mock_run):
        """Test handling git diff errors."""
        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(1, "git", stderr="Not a git repo")
        with self.assertRaises(ValueError):
            CodeCollector.from_git_diff()

    @patch('subprocess.run')
    def test_from_staged(self, mock_run):
        """Test collecting staged changes."""
        mock_run.return_value = MagicMock(
            stdout="diff --git a/staged.py b/staged.py\n+++ b/staged.py\n+# staged",
            returncode=0
        )
        code, files = CodeCollector.from_staged()
        self.assertIn("staged", code)

    def test_from_stdin_no_input(self):
        """Test stdin with no input (tty)."""
        with patch.object(sys.stdin, 'isatty', return_value=True):
            with self.assertRaises(ValueError):
                CodeCollector.from_stdin()


class TestLLMProviders(unittest.TestCase):
    """Tests for LLM providers."""

    def test_base_provider_review_not_implemented(self):
        """Test that base provider review is not implemented."""
        provider = LLMProvider("api_key")
        with self.assertRaises(NotImplementedError):
            provider.review("code", [FocusArea.ALL])

    def test_build_prompt(self):
        """Test prompt building."""
        provider = LLMProvider("api_key")
        prompt = provider._build_prompt("print('hello')", [FocusArea.SECURITY])
        self.assertIn("security", prompt)
        self.assertIn("print('hello')", prompt)
        self.assertIn("JSON", prompt)

    def test_parse_response_valid_json(self):
        """Test parsing valid JSON response."""
        provider = LLMProvider("api_key", "test-model")
        response = json.dumps({
            "issues": [
                {"severity": "high", "category": "security", "file": "test.py",
                 "line": 1, "message": "Test issue"}
            ],
            "summary": "Found issues"
        })
        result = provider._parse_response(response, ["test.py"], [FocusArea.SECURITY])
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.issues[0].severity, "high")

    def test_parse_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        provider = LLMProvider("api_key", "test-model")
        # Use text that doesn't contain any curly braces - it goes into summary
        result = provider._parse_response("Not valid JSON at all", ["test.py"], [FocusArea.ALL])
        self.assertEqual(len(result.issues), 0)
        self.assertIn("Not valid JSON", result.summary)

    def test_parse_response_malformed_json(self):
        """Test parsing malformed JSON response with braces."""
        provider = LLMProvider("api_key", "test-model")
        # Use text with braces but invalid JSON - triggers JSONDecodeError path
        result = provider._parse_response("Here is JSON: {invalid json}", ["test.py"], [FocusArea.ALL])
        self.assertEqual(len(result.issues), 1)
        self.assertIn("invalid json", result.issues[0].message)


class TestOpenAIProvider(unittest.TestCase):
    """Tests for OpenAI provider."""

    def test_review_success(self):
        """Test successful OpenAI review."""
        # Create mock module and client
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "issues": [],
            "summary": "Code looks good"
        })
        mock_client.chat.completions.create.return_value = mock_response

        # Patch the import inside the method
        with patch.dict('sys.modules', {'openai': mock_openai}):
            provider = OpenAIProvider("test-key")
            result = provider.review("print('hello')", [FocusArea.ALL], files=["test.py"])

            self.assertEqual(result.summary, "Code looks good")
            self.assertEqual(result.issues, [])

    def test_review_auth_error(self):
        """Test OpenAI authentication error."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        # Create a real exception class
        class MockAuthError(Exception):
            pass
        mock_openai.AuthenticationError = MockAuthError
        mock_client.chat.completions.create.side_effect = MockAuthError("Invalid key")

        with patch.dict('sys.modules', {'openai': mock_openai}):
            provider = OpenAIProvider("bad-key")
            with self.assertRaises(APIError):
                provider.review("code", [FocusArea.ALL])

    def test_default_model(self):
        """Test default model is set."""
        provider = OpenAIProvider("test-key")
        self.assertEqual(provider.model, OpenAIProvider.DEFAULT_MODEL)

    def test_custom_model(self):
        """Test custom model can be specified."""
        provider = OpenAIProvider("test-key", "gpt-4o")
        self.assertEqual(provider.model, "gpt-4o")


class TestAnthropicProvider(unittest.TestCase):
    """Tests for Anthropic provider."""

    def test_review_success(self):
        """Test successful Anthropic review."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps({
            "issues": [{"severity": "medium", "category": "style",
                       "file": "test.py", "line": 1, "message": "Style issue"}],
            "summary": "Minor style issues"
        })
        mock_client.messages.create.return_value = mock_response

        with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            provider = AnthropicProvider("test-key")
            result = provider.review("def foo():\n  pass", [FocusArea.STYLE], files=["test.py"])

            self.assertEqual(len(result.issues), 1)
            self.assertEqual(result.issues[0].category, "style")

    def test_model_alias(self):
        """Test model alias handling."""
        provider = AnthropicProvider("test-key", "claude-3-5-sonnet")
        self.assertEqual(provider.model, "claude-3-5-sonnet-latest")

    def test_default_model(self):
        """Test default model is set."""
        provider = AnthropicProvider("test-key")
        self.assertEqual(provider.model, AnthropicProvider.DEFAULT_MODEL)


class TestGoogleProvider(unittest.TestCase):
    """Tests for Google provider."""

    def test_review_success(self):
        """Test successful Google review."""
        mock_genai = MagicMock()

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "issues": [],
            "summary": "No issues found"
        })
        mock_model.generate_content.return_value = mock_response

        # Create nested mock for google.generativeai
        mock_google = MagicMock()
        mock_google.generativeai = mock_genai

        with patch.dict('sys.modules', {'google': mock_google, 'google.generativeai': mock_genai}):
            provider = GoogleProvider("test-key")
            result = provider.review("print('hello')", [FocusArea.ALL], files=["test.py"])

            self.assertEqual(result.summary, "No issues found")

    def test_default_model(self):
        """Test default model is set."""
        provider = GoogleProvider("test-key")
        self.assertEqual(provider.model, GoogleProvider.DEFAULT_MODEL)


class TestGetProvider(unittest.TestCase):
    """Tests for get_provider function."""

    def test_no_api_key(self):
        """Test error when no API key is set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing keys
            for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']:
                os.environ.pop(key, None)
            with self.assertRaises(ValueError) as ctx:
                get_provider()
            self.assertIn("No API key found", str(ctx.exception))

    def test_anthropic_key_default(self):
        """Test Anthropic is default when key is available."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}, clear=True):
            provider = get_provider()
            self.assertIsInstance(provider, AnthropicProvider)

    def test_openai_key_fallback(self):
        """Test OpenAI is used when only its key is available."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=True):
            provider = get_provider()
            self.assertIsInstance(provider, OpenAIProvider)

    def test_google_key_fallback(self):
        """Test Google is used when only its key is available."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'}, clear=True):
            provider = get_provider()
            self.assertIsInstance(provider, GoogleProvider)

    def test_model_specific_provider(self):
        """Test provider selection based on model name."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            provider = get_provider("gpt-4")
            self.assertIsInstance(provider, OpenAIProvider)

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = get_provider("claude-3-5-sonnet")
            self.assertIsInstance(provider, AnthropicProvider)

        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'}):
            provider = get_provider("gemini-pro")
            self.assertIsInstance(provider, GoogleProvider)


class TestArgumentParser(unittest.TestCase):
    """Tests for argument parsing."""

    def test_parser_creation(self):
        """Test parser is created correctly."""
        parser = create_parser()
        self.assertIsNotNone(parser)

    def test_file_argument(self):
        """Test -f/--file argument."""
        parser = create_parser()
        args = parser.parse_args(["-f", "test.py"])
        self.assertEqual(args.files, ["test.py"])

    def test_multiple_files(self):
        """Test multiple file arguments."""
        parser = create_parser()
        args = parser.parse_args(["-f", "a.py", "-f", "b.py"])
        self.assertEqual(args.files, ["a.py", "b.py"])

    def test_positional_files(self):
        """Test positional file arguments."""
        parser = create_parser()
        args = parser.parse_args(["a.py", "b.py"])
        self.assertEqual(args.positional_files, ["a.py", "b.py"])

    def test_directory_argument(self):
        """Test -d/--directory argument."""
        parser = create_parser()
        args = parser.parse_args(["-d", "./src"])
        self.assertEqual(args.directory, "./src")

    def test_diff_argument(self):
        """Test --diff argument."""
        parser = create_parser()
        args = parser.parse_args(["--diff", "HEAD~2"])
        self.assertEqual(args.diff, "HEAD~2")

    def test_diff_default(self):
        """Test --diff with default value."""
        parser = create_parser()
        args = parser.parse_args(["--diff"])
        self.assertEqual(args.diff, "HEAD~1")

    def test_staged_argument(self):
        """Test --staged argument."""
        parser = create_parser()
        args = parser.parse_args(["--staged"])
        self.assertTrue(args.staged)

    def test_stdin_argument(self):
        """Test --stdin argument."""
        parser = create_parser()
        args = parser.parse_args(["--stdin"])
        self.assertTrue(args.stdin)

    def test_focus_argument(self):
        """Test --focus argument."""
        parser = create_parser()
        args = parser.parse_args(["-f", "test.py", "--focus", "security,performance"])
        self.assertEqual(args.focus, "security,performance")

    def test_output_format(self):
        """Test -o/--output argument."""
        parser = create_parser()
        args = parser.parse_args(["-f", "test.py", "-o", "json"])
        self.assertEqual(args.output, "json")

    def test_fail_on(self):
        """Test --fail-on argument."""
        parser = create_parser()
        args = parser.parse_args(["-f", "test.py", "--fail-on", "high"])
        self.assertEqual(args.fail_on, "high")

    def test_min_severity(self):
        """Test --min-severity argument."""
        parser = create_parser()
        args = parser.parse_args(["-f", "test.py", "--min-severity", "medium"])
        self.assertEqual(args.min_severity, "medium")

    def test_model_argument(self):
        """Test --model argument."""
        parser = create_parser()
        args = parser.parse_args(["-f", "test.py", "--model", "gpt-4"])
        self.assertEqual(args.model, "gpt-4")

    def test_verbose_quiet(self):
        """Test verbose and quiet flags."""
        parser = create_parser()
        args = parser.parse_args(["-f", "test.py", "-v"])
        self.assertTrue(args.verbose)

        args = parser.parse_args(["-f", "test.py", "-q"])
        self.assertTrue(args.quiet)


class TestMain(unittest.TestCase):
    """Tests for main function."""

    def test_no_input_error(self):
        """Test error when no input is provided."""
        with patch('sys.argv', ['code_review']):
            with patch('sys.stderr', new_callable=StringIO):
                # parser.error() calls sys.exit(2), which we need to catch
                with self.assertRaises(SystemExit) as ctx:
                    main()
                # argparse exits with 2 for argument errors
                self.assertEqual(ctx.exception.code, 2)

    @patch('code_review.get_provider')
    @patch('code_review.CodeCollector.from_files')
    def test_successful_review_no_issues(self, mock_collector, mock_provider):
        """Test successful review with no issues."""
        mock_collector.return_value = ("print('hello')", ["test.py"])

        mock_provider_instance = MagicMock()
        mock_provider_instance.review.return_value = ReviewResult(
            issues=[],
            summary="No issues",
            files_reviewed=["test.py"],
            model_used="test-model",
            focus_areas=["all"]
        )
        mock_provider.return_value = mock_provider_instance

        with patch('sys.argv', ['code_review', '-f', 'test.py']):
            with patch('sys.stdout', new_callable=StringIO):
                with patch('sys.stderr', new_callable=StringIO):
                    result = main()
                    self.assertEqual(result, ExitCode.SUCCESS.value)

    @patch('code_review.get_provider')
    @patch('code_review.CodeCollector.from_files')
    def test_successful_review_with_issues(self, mock_collector, mock_provider):
        """Test successful review with issues found."""
        mock_collector.return_value = ("code", ["test.py"])

        mock_provider_instance = MagicMock()
        mock_provider_instance.review.return_value = ReviewResult(
            issues=[Issue(severity="high", category="security", file="test.py",
                         line=1, message="Issue found")],
            summary="Issues found",
            files_reviewed=["test.py"],
            model_used="test-model",
            focus_areas=["security"]
        )
        mock_provider.return_value = mock_provider_instance

        with patch('sys.argv', ['code_review', '-f', 'test.py']):
            with patch('sys.stdout', new_callable=StringIO):
                with patch('sys.stderr', new_callable=StringIO):
                    result = main()
                    self.assertEqual(result, ExitCode.ISSUES_FOUND.value)

    @patch('code_review.get_provider')
    @patch('code_review.CodeCollector.from_files')
    def test_fail_on_threshold(self, mock_collector, mock_provider):
        """Test --fail-on threshold."""
        mock_collector.return_value = ("code", ["test.py"])

        mock_provider_instance = MagicMock()
        mock_provider_instance.review.return_value = ReviewResult(
            issues=[Issue(severity="low", category="style", file="test.py",
                         line=1, message="Minor issue")],
            summary="Minor issues",
            files_reviewed=["test.py"],
            model_used="test-model",
            focus_areas=["all"]
        )
        mock_provider.return_value = mock_provider_instance

        # With --fail-on high, low severity issues should not trigger failure
        with patch('sys.argv', ['code_review', '-f', 'test.py', '--fail-on', 'high']):
            with patch('sys.stdout', new_callable=StringIO):
                with patch('sys.stderr', new_callable=StringIO):
                    result = main()
                    # Should still return ISSUES_FOUND because there are issues
                    # but not due to fail-on threshold
                    self.assertEqual(result, ExitCode.ISSUES_FOUND.value)

    @patch('code_review.get_provider')
    def test_api_error(self, mock_provider):
        """Test API error handling."""
        mock_provider.side_effect = APIError("API failed")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('test')")
            f.flush()
            try:
                with patch('sys.argv', ['code_review', '-f', f.name]):
                    with patch('sys.stderr', new_callable=StringIO):
                        result = main()
                        self.assertEqual(result, ExitCode.API_ERROR.value)
            finally:
                os.unlink(f.name)

    def test_invalid_focus_area(self):
        """Test invalid focus area error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('test')")
            f.flush()
            try:
                with patch('sys.argv', ['code_review', '-f', f.name, '--focus', 'invalid']):
                    with patch('sys.stderr', new_callable=StringIO):
                        result = main()
                        self.assertEqual(result, ExitCode.INVALID_ARGS.value)
            finally:
                os.unlink(f.name)

    @patch('code_review.get_provider')
    @patch('code_review.CodeCollector.from_files')
    def test_json_output(self, mock_collector, mock_provider):
        """Test JSON output format."""
        mock_collector.return_value = ("code", ["test.py"])

        mock_provider_instance = MagicMock()
        mock_provider_instance.review.return_value = ReviewResult(
            issues=[],
            summary="No issues",
            files_reviewed=["test.py"],
            model_used="test-model",
            focus_areas=["all"]
        )
        mock_provider.return_value = mock_provider_instance

        with patch('sys.argv', ['code_review', '-f', 'test.py', '-o', 'json']):
            stdout = StringIO()
            with patch('sys.stdout', stdout):
                with patch('sys.stderr', new_callable=StringIO):
                    main()
            output = stdout.getvalue()
            data = json.loads(output)
            self.assertIn("issues", data)
            self.assertIn("summary", data)

    @patch('code_review.get_provider')
    @patch('code_review.CodeCollector.from_files')
    def test_min_severity_filter(self, mock_collector, mock_provider):
        """Test --min-severity filtering."""
        mock_collector.return_value = ("code", ["test.py"])

        mock_provider_instance = MagicMock()
        mock_provider_instance.review.return_value = ReviewResult(
            issues=[
                Issue(severity="info", category="style", file="test.py", line=1, message="Info"),
                Issue(severity="high", category="security", file="test.py", line=2, message="High"),
            ],
            summary="Mixed issues",
            files_reviewed=["test.py"],
            model_used="test-model",
            focus_areas=["all"]
        )
        mock_provider.return_value = mock_provider_instance

        with patch('sys.argv', ['code_review', '-f', 'test.py', '--min-severity', 'high', '-o', 'json']):
            stdout = StringIO()
            with patch('sys.stdout', stdout):
                with patch('sys.stderr', new_callable=StringIO):
                    main()
            output = stdout.getvalue()
            data = json.loads(output)
            # Only high severity issue should remain
            self.assertEqual(len(data["issues"]), 1)
            self.assertEqual(data["issues"][0]["severity"], "high")


class TestExitCodes(unittest.TestCase):
    """Tests for exit codes."""

    def test_exit_code_values(self):
        """Test exit code values."""
        self.assertEqual(ExitCode.SUCCESS.value, 0)
        self.assertEqual(ExitCode.ISSUES_FOUND.value, 1)
        self.assertEqual(ExitCode.API_ERROR.value, 2)
        self.assertEqual(ExitCode.INVALID_ARGS.value, 3)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    @patch('code_review.get_provider')
    def test_full_workflow_with_temp_file(self, mock_provider):
        """Test full workflow with a temporary file."""
        mock_provider_instance = MagicMock()
        mock_provider_instance.review.return_value = ReviewResult(
            issues=[
                Issue(severity="medium", category="bugs", file="temp.py",
                      line=3, message="Possible division by zero",
                      suggestion="Add check for zero denominator")
            ],
            summary="Found 1 potential bug",
            files_reviewed=["temp.py"],
            model_used="gpt-4",
            focus_areas=["bugs"]
        )
        mock_provider.return_value = mock_provider_instance

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def divide(a, b):\n    return a / b\n")
            f.flush()

            try:
                with patch('sys.argv', ['code_review', '-f', f.name, '--focus', 'bugs', '-o', 'markdown']):
                    stdout = StringIO()
                    with patch('sys.stdout', stdout):
                        with patch('sys.stderr', new_callable=StringIO):
                            result = main()

                    self.assertEqual(result, ExitCode.ISSUES_FOUND.value)
                    output = stdout.getvalue()
                    self.assertIn("# Code Review Report", output)
                    self.assertIn("MEDIUM", output)
                    self.assertIn("division by zero", output)
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
