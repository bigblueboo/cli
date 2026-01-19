#!/usr/bin/env python3
"""
Comprehensive tests for test_generator.py

Tests cover:
- Argument parsing
- Language and framework detection
- LLM configuration
- API calls (mocked)
- Pattern detection
- Coverage analysis
- Error handling and exit codes
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Import the module under test
import test_generator as tg


class TestLanguageDetection:
    """Tests for language detection functionality."""

    @pytest.mark.parametrize("filename,expected", [
        ("test.py", "python"),
        ("main.js", "javascript"),
        ("app.ts", "typescript"),
        ("handler.go", "go"),
        ("lib.rs", "rust"),
        ("Service.java", "java"),
        ("model.rb", "ruby"),
        ("test.jsx", "javascript"),
        ("component.tsx", "typescript"),
        ("main.cpp", "cpp"),
        ("utils.c", "c"),
        ("app.swift", "swift"),
        ("main.kt", "kotlin"),
    ])
    def test_detect_language_various_extensions(self, filename, expected):
        """Test language detection for various file extensions."""
        path = Path(filename)
        assert tg.detect_language(path) == expected

    def test_detect_language_unknown_extension(self):
        """Test that unknown extensions return None."""
        path = Path("file.unknown")
        assert tg.detect_language(path) is None

    def test_detect_language_case_insensitive(self):
        """Test that extension detection is case-insensitive."""
        assert tg.detect_language(Path("test.PY")) == "python"
        assert tg.detect_language(Path("test.Js")) == "javascript"


class TestFrameworkDetection:
    """Tests for test framework detection."""

    def test_default_framework_python(self):
        """Test default framework for Python is pytest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.touch()
            framework = tg.detect_framework(test_file, "python")
            assert framework == tg.TestFramework.PYTEST

    def test_default_framework_javascript(self):
        """Test default framework for JavaScript is Jest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.js"
            test_file.touch()
            framework = tg.detect_framework(test_file, "javascript")
            assert framework == tg.TestFramework.JEST

    def test_default_framework_go(self):
        """Test default framework for Go is gotest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.go"
            test_file.touch()
            framework = tg.detect_framework(test_file, "go")
            assert framework == tg.TestFramework.GOTEST

    def test_detect_pytest_from_conftest(self):
        """Test pytest detection when conftest.py exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.touch()
            conftest = Path(tmpdir) / "conftest.py"
            conftest.touch()
            framework = tg.detect_framework(test_file, "python")
            assert framework == tg.TestFramework.PYTEST

    def test_detect_pytest_from_pyproject(self):
        """Test pytest detection from pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.touch()
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text("[tool.pytest]\ntestpaths = ['tests']")
            framework = tg.detect_framework(test_file, "python")
            assert framework == tg.TestFramework.PYTEST

    def test_detect_jest_from_package_json(self):
        """Test Jest detection from package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.js"
            test_file.touch()
            package_json = Path(tmpdir) / "package.json"
            package_json.write_text(json.dumps({
                "devDependencies": {"jest": "^29.0.0"}
            }))
            framework = tg.detect_framework(test_file, "javascript")
            assert framework == tg.TestFramework.JEST

    def test_detect_mocha_from_package_json(self):
        """Test Mocha detection from package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.js"
            test_file.touch()
            package_json = Path(tmpdir) / "package.json"
            package_json.write_text(json.dumps({
                "devDependencies": {"mocha": "^10.0.0"}
            }))
            framework = tg.detect_framework(test_file, "javascript")
            assert framework == tg.TestFramework.MOCHA

    def test_detect_vitest_from_package_json(self):
        """Test Vitest detection from package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.ts"
            test_file.touch()
            package_json = Path(tmpdir) / "package.json"
            package_json.write_text(json.dumps({
                "devDependencies": {"vitest": "^1.0.0"}
            }))
            framework = tg.detect_framework(test_file, "typescript")
            assert framework == tg.TestFramework.VITEST


class TestPatternDetection:
    """Tests for existing test pattern detection."""

    def test_detect_pytest_fixtures(self):
        """Test detection of pytest fixtures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            test_file = test_dir / "test_example.py"
            test_file.write_text("""
import pytest

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_something(sample_data):
    assert sample_data["key"] == "value"
""")
            patterns = tg.detect_existing_patterns(test_dir)
            assert "pytest fixtures" in patterns.lower()

    def test_detect_parametrized_tests(self):
        """Test detection of parametrized tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            test_file = test_dir / "test_example.py"
            test_file.write_text("""
import pytest

@pytest.mark.parametrize("input,expected", [(1, 2), (2, 4)])
def test_double(input, expected):
    assert input * 2 == expected
""")
            patterns = tg.detect_existing_patterns(test_dir)
            assert "parametrized" in patterns.lower()

    def test_detect_unittest_mock(self):
        """Test detection of unittest.mock usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            test_file = test_dir / "test_example.py"
            test_file.write_text("""
from unittest.mock import Mock, patch

def test_with_mock():
    mock_obj = Mock()
    mock_obj.method.return_value = 42
    assert mock_obj.method() == 42
""")
            patterns = tg.detect_existing_patterns(test_dir)
            assert "unittest.mock" in patterns.lower()

    def test_detect_async_tests(self):
        """Test detection of async tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            test_file = test_dir / "test_example.py"
            test_file.write_text("""
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result is not None
""")
            patterns = tg.detect_existing_patterns(test_dir)
            assert "async" in patterns.lower()

    def test_empty_directory(self):
        """Test pattern detection on empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            patterns = tg.detect_existing_patterns(test_dir)
            assert patterns == ""

    def test_nonexistent_directory(self):
        """Test pattern detection on non-existent directory."""
        patterns = tg.detect_existing_patterns(Path("/nonexistent/path"))
        assert patterns == ""


class TestLLMConfiguration:
    """Tests for LLM configuration and API key handling."""

    def test_anthropic_key_priority(self):
        """Test that Anthropic key takes priority."""
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "anthropic-key",
            "OPENAI_API_KEY": "openai-key",
            "GOOGLE_API_KEY": "google-key"
        }):
            config = tg.get_llm_config()
            assert config.provider == tg.LLMProvider.ANTHROPIC
            assert config.api_key == "anthropic-key"

    def test_openai_key_fallback(self):
        """Test fallback to OpenAI when Anthropic not available."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "openai-key",
            "GOOGLE_API_KEY": "google-key"
        }, clear=True):
            # Clear ANTHROPIC_API_KEY if it exists
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = tg.get_llm_config()
            assert config.provider == tg.LLMProvider.OPENAI
            assert config.api_key == "openai-key"

    def test_google_key_fallback(self):
        """Test fallback to Google when others not available."""
        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "google-key"
        }, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            config = tg.get_llm_config()
            assert config.provider == tg.LLMProvider.GOOGLE
            assert config.api_key == "google-key"

    def test_no_api_key_exits(self):
        """Test that missing API keys cause exit."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            with pytest.raises(SystemExit) as exc_info:
                tg.get_llm_config()
            assert exc_info.value.code == tg.EXIT_INVALID_ARGS


class TestAnthropicAPI:
    """Tests for Anthropic API integration."""

    def test_successful_call(self):
        """Test successful Anthropic API call."""
        import anthropic

        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text="Generated test code")]
        mock_client.messages.create.return_value = mock_message

        with patch.object(anthropic, "Anthropic", return_value=mock_client):
            config = tg.LLMConfig(
                provider=tg.LLMProvider.ANTHROPIC,
                api_key="test-key",
                model="claude-3-sonnet"
            )
            result = tg.call_anthropic(config, "test prompt")
            assert result == "Generated test code"

    def test_connection_error(self):
        """Test handling of API connection error."""
        import anthropic

        mock_client = Mock()
        mock_request = Mock()
        mock_client.messages.create.side_effect = anthropic.APIConnectionError(
            message="Connection failed",
            request=mock_request
        )

        with patch.object(anthropic, "Anthropic", return_value=mock_client):
            config = tg.LLMConfig(
                provider=tg.LLMProvider.ANTHROPIC,
                api_key="test-key",
                model="claude-3-sonnet"
            )
            with pytest.raises(SystemExit) as exc_info:
                tg.call_anthropic(config, "test prompt")
            assert exc_info.value.code == tg.EXIT_API_ERROR

    def test_rate_limit_error(self):
        """Test handling of rate limit error."""
        import anthropic

        mock_client = Mock()
        # RateLimitError requires a response parameter
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_client.messages.create.side_effect = anthropic.RateLimitError(
            message="Rate limited",
            response=mock_response,
            body=None
        )

        with patch.object(anthropic, "Anthropic", return_value=mock_client):
            config = tg.LLMConfig(
                provider=tg.LLMProvider.ANTHROPIC,
                api_key="test-key",
                model="claude-3-sonnet"
            )
            with pytest.raises(SystemExit) as exc_info:
                tg.call_anthropic(config, "test prompt")
            assert exc_info.value.code == tg.EXIT_API_ERROR


class TestOpenAIAPI:
    """Tests for OpenAI API integration."""

    def test_successful_call(self):
        """Test successful OpenAI API call."""
        import openai

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated test code"))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(openai, "OpenAI", return_value=mock_client):
            config = tg.LLMConfig(
                provider=tg.LLMProvider.OPENAI,
                api_key="test-key",
                model="gpt-4"
            )
            result = tg.call_openai(config, "test prompt")
            assert result == "Generated test code"

    def test_connection_error(self):
        """Test handling of OpenAI connection error."""
        import openai

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            message="Connection failed",
            request=Mock()
        )

        with patch.object(openai, "OpenAI", return_value=mock_client):
            config = tg.LLMConfig(
                provider=tg.LLMProvider.OPENAI,
                api_key="test-key",
                model="gpt-4"
            )
            with pytest.raises(SystemExit) as exc_info:
                tg.call_openai(config, "test prompt")
            assert exc_info.value.code == tg.EXIT_API_ERROR


class TestGoogleAPI:
    """Tests for Google Gemini API integration."""

    @patch("test_generator.genai", create=True)
    def test_successful_call(self, mock_genai):
        """Test successful Google API call."""
        # We need to patch at import time, so we'll mock the module
        with patch.dict(sys.modules, {"google.generativeai": mock_genai}):
            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            mock_response = Mock(text="Generated test code")
            mock_model.generate_content.return_value = mock_response

            config = tg.LLMConfig(
                provider=tg.LLMProvider.GOOGLE,
                api_key="test-key",
                model="gemini-pro"
            )
            result = tg.call_google(config, "test prompt")
            assert result == "Generated test code"


class TestPromptBuilding:
    """Tests for prompt construction."""

    def test_basic_prompt_structure(self):
        """Test that basic prompt contains required elements."""
        prompt = tg.build_prompt(
            source_code="def add(a, b): return a + b",
            language="python",
            framework=tg.TestFramework.PYTEST,
            test_type=tg.TestType.UNIT,
            include_edge_cases=False,
            existing_patterns=""
        )
        assert "python" in prompt.lower()
        assert "pytest" in prompt.lower()
        assert "def add(a, b): return a + b" in prompt

    def test_edge_cases_included(self):
        """Test that edge case instructions are added when requested."""
        prompt = tg.build_prompt(
            source_code="def process(data): pass",
            language="python",
            framework=tg.TestFramework.PYTEST,
            test_type=tg.TestType.UNIT,
            include_edge_cases=True,
            existing_patterns=""
        )
        assert "edge case" in prompt.lower()
        assert "empty" in prompt.lower()
        assert "boundary" in prompt.lower()

    def test_integration_test_instructions(self):
        """Test that integration test instructions are added."""
        prompt = tg.build_prompt(
            source_code="def api_call(): pass",
            language="python",
            framework=tg.TestFramework.PYTEST,
            test_type=tg.TestType.INTEGRATION,
            include_edge_cases=False,
            existing_patterns=""
        )
        assert "integration" in prompt.lower()
        assert "mock" in prompt.lower()

    def test_existing_patterns_included(self):
        """Test that existing patterns are included in prompt."""
        prompt = tg.build_prompt(
            source_code="def test_func(): pass",
            language="python",
            framework=tg.TestFramework.PYTEST,
            test_type=tg.TestType.UNIT,
            include_edge_cases=False,
            existing_patterns="Uses pytest fixtures; Uses parametrized tests"
        )
        assert "pytest fixtures" in prompt.lower()
        assert "parametrized" in prompt.lower()


class TestTestGeneration:
    """Tests for the main test generation function."""

    @patch("test_generator.call_anthropic")
    def test_generate_tests_success(self, mock_call):
        """Test successful test generation."""
        mock_call.return_value = "```python\ndef test_example(): pass\n```"

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def example(): return 42")
            f.flush()

            config = tg.TestGenerationConfig(
                source_file=Path(f.name),
                framework=tg.TestFramework.PYTEST,
                test_type=tg.TestType.UNIT,
                include_edge_cases=False,
                analyze_dir=None,
                suggest_missing=False,
                verbose=False,
                llm_config=tg.LLMConfig(
                    provider=tg.LLMProvider.ANTHROPIC,
                    api_key="test-key",
                    model="claude-3"
                ),
                existing_patterns=""
            )

            result = tg.generate_tests(config)
            assert "def test_example(): pass" in result
            os.unlink(f.name)

    def test_file_not_found(self):
        """Test that missing files cause exit with code 1."""
        config = tg.TestGenerationConfig(
            source_file=Path("/nonexistent/file.py"),
            framework=tg.TestFramework.PYTEST,
            test_type=tg.TestType.UNIT,
            include_edge_cases=False,
            analyze_dir=None,
            suggest_missing=False,
            verbose=False,
            llm_config=tg.LLMConfig(
                provider=tg.LLMProvider.ANTHROPIC,
                api_key="test-key",
                model="claude-3"
            ),
            existing_patterns=""
        )

        with pytest.raises(SystemExit) as exc_info:
            tg.generate_tests(config)
        assert exc_info.value.code == tg.EXIT_FILE_NOT_FOUND

    def test_unknown_file_type(self):
        """Test that unknown file types cause exit with code 3."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"some content")
            f.flush()

            config = tg.TestGenerationConfig(
                source_file=Path(f.name),
                framework=None,
                test_type=tg.TestType.UNIT,
                include_edge_cases=False,
                analyze_dir=None,
                suggest_missing=False,
                verbose=False,
                llm_config=tg.LLMConfig(
                    provider=tg.LLMProvider.ANTHROPIC,
                    api_key="test-key",
                    model="claude-3"
                ),
                existing_patterns=""
            )

            with pytest.raises(SystemExit) as exc_info:
                tg.generate_tests(config)
            assert exc_info.value.code == tg.EXIT_INVALID_ARGS
            os.unlink(f.name)


class TestCoverageAnalysis:
    """Tests for test coverage analysis functionality."""

    def test_analyze_empty_directory(self):
        """Test analysis of directory with no test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analysis = tg.analyze_test_coverage(Path(tmpdir))
            assert analysis["test_files"] == []
            assert len(analysis["tested_modules"]) == 0

    def test_analyze_with_test_files(self):
        """Test analysis finds test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            test_file = test_dir / "test_example.py"
            test_file.write_text("""
from mymodule import myfunction

def test_myfunction():
    assert myfunction() == 42
""")
            analysis = tg.analyze_test_coverage(test_dir)
            assert len(analysis["test_files"]) == 1
            assert "mymodule" in analysis["tested_modules"]

    def test_analyze_finds_coverage_gaps(self):
        """Test that coverage gaps are identified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # The analyze function looks for source files in parent directory
            # and tests in the analyzed directory
            tests_dir = Path(tmpdir) / "spec"  # Use "spec" to avoid "test" in path
            tests_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()

            # Create source file in src/ without corresponding tests
            # Name must not contain "test" as that's filtered out
            (src_dir / "mymodule.py").write_text("def my_function(): pass")

            # Create a test file that tests something else
            (tests_dir / "test_other.py").write_text("""
from other_module import something
def test_something(): pass
""")
            analysis = tg.analyze_test_coverage(tests_dir)
            # Should suggest testing mymodule (found in parent_dir/src/)
            # The function uses **/*.py which finds files in subdirectories
            assert any("mymodule" in gap for gap in analysis["coverage_gaps"])

    def test_analyze_nonexistent_directory(self):
        """Test that analyzing non-existent directory exits with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            tg.analyze_test_coverage(Path("/nonexistent/test/dir"))
        assert exc_info.value.code == tg.EXIT_FILE_NOT_FOUND


class TestArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = tg.create_parser()
        assert parser is not None
        assert parser.prog == "test_generator"

    def test_source_file_argument(self):
        """Test source file argument parsing."""
        parser = tg.create_parser()
        args = parser.parse_args(["test.py"])
        assert args.source_file == Path("test.py")

    def test_framework_argument(self):
        """Test framework argument parsing."""
        parser = tg.create_parser()
        args = parser.parse_args(["test.py", "--framework", "pytest"])
        assert args.framework == "pytest"

    def test_type_argument(self):
        """Test test type argument parsing."""
        parser = tg.create_parser()
        args = parser.parse_args(["test.py", "--type", "integration"])
        assert args.type == "integration"

    def test_edge_cases_flag(self):
        """Test edge cases flag parsing."""
        parser = tg.create_parser()
        args = parser.parse_args(["test.py", "--edge-cases"])
        assert args.edge_cases is True

    def test_analyze_argument(self):
        """Test analyze argument parsing."""
        parser = tg.create_parser()
        args = parser.parse_args(["--analyze", "tests/"])
        assert args.analyze == Path("tests/")

    def test_suggest_missing_flag(self):
        """Test suggest missing flag parsing."""
        parser = tg.create_parser()
        args = parser.parse_args(["--analyze", "tests/", "--suggest-missing"])
        assert args.suggest_missing is True

    def test_verbose_flag(self):
        """Test verbose flag parsing."""
        parser = tg.create_parser()
        args = parser.parse_args(["test.py", "--verbose"])
        assert args.verbose is True

    def test_provider_argument(self):
        """Test provider argument parsing."""
        parser = tg.create_parser()
        args = parser.parse_args(["test.py", "--provider", "openai"])
        assert args.provider == "openai"

    def test_model_argument(self):
        """Test model argument parsing."""
        parser = tg.create_parser()
        args = parser.parse_args(["test.py", "--model", "gpt-4-turbo"])
        assert args.model == "gpt-4-turbo"

    def test_short_flags(self):
        """Test short flag versions."""
        parser = tg.create_parser()
        args = parser.parse_args(["test.py", "-f", "jest", "-t", "unit", "-e", "-v"])
        assert args.framework == "jest"
        assert args.type == "unit"
        assert args.edge_cases is True
        assert args.verbose is True


class TestMainFunction:
    """Tests for the main entry point."""

    @patch("test_generator.get_llm_config")
    @patch("test_generator.generate_tests")
    def test_main_success(self, mock_generate, mock_config):
        """Test successful main execution."""
        mock_config.return_value = tg.LLMConfig(
            provider=tg.LLMProvider.ANTHROPIC,
            api_key="test-key",
            model="claude-3"
        )
        mock_generate.return_value = "def test_example(): pass"

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def example(): pass")
            f.flush()

            with patch.object(sys, "argv", ["test_generator", f.name]):
                result = tg.main()
                assert result == tg.EXIT_SUCCESS
                os.unlink(f.name)

    def test_main_no_args_shows_help(self):
        """Test that running without args shows help."""
        with patch.object(sys, "argv", ["test_generator"]):
            with patch.object(sys, "stderr", new_callable=StringIO):
                result = tg.main()
                assert result == tg.EXIT_INVALID_ARGS

    @patch("test_generator.analyze_test_coverage")
    def test_main_analyze_mode(self, mock_analyze):
        """Test main function in analyze mode."""
        mock_analyze.return_value = {
            "test_files": ["test_a.py"],
            "tested_modules": ["module_a"],
            "suggestions": [],
            "coverage_gaps": []
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(sys, "argv", ["test_generator", "--analyze", tmpdir]):
                result = tg.main()
                assert result == tg.EXIT_SUCCESS


class TestOutputFormatting:
    """Tests for output formatting functions."""

    def test_format_analysis_output(self):
        """Test analysis output formatting."""
        analysis = {
            "test_files": ["tests/test_a.py", "tests/test_b.py"],
            "tested_modules": ["module_a", "module_b"],
            "suggestions": ["Consider adding tests for: src/untested.py"],
            "coverage_gaps": ["src/untested.py"]
        }
        output = tg.format_analysis_output(analysis)

        assert "TEST COVERAGE ANALYSIS" in output
        assert "2 test file(s)" in output
        assert "SUGGESTED MISSING TESTS" in output
        assert "src/untested.py" in output

    def test_format_analysis_no_gaps(self):
        """Test output when no coverage gaps."""
        analysis = {
            "test_files": ["tests/test_a.py"],
            "tested_modules": ["module_a"],
            "suggestions": [],
            "coverage_gaps": []
        }
        output = tg.format_analysis_output(analysis)

        assert "No obvious coverage gaps found" in output


class TestCodeBlockExtraction:
    """Tests for extracting code from markdown code blocks."""

    @patch("test_generator.call_anthropic")
    def test_extracts_code_from_markdown(self, mock_call):
        """Test that code is extracted from markdown blocks."""
        mock_call.return_value = """
Here is the test code:

```python
import pytest

def test_example():
    assert True
```

This tests the example function.
"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def example(): pass")
            f.flush()

            config = tg.TestGenerationConfig(
                source_file=Path(f.name),
                framework=tg.TestFramework.PYTEST,
                test_type=tg.TestType.UNIT,
                include_edge_cases=False,
                analyze_dir=None,
                suggest_missing=False,
                verbose=False,
                llm_config=tg.LLMConfig(
                    provider=tg.LLMProvider.ANTHROPIC,
                    api_key="test-key",
                    model="claude-3"
                ),
                existing_patterns=""
            )

            result = tg.generate_tests(config)
            assert "import pytest" in result
            assert "def test_example():" in result
            assert "Here is the test code" not in result
            os.unlink(f.name)


class TestEnums:
    """Tests for enum definitions."""

    def test_test_framework_values(self):
        """Test TestFramework enum has expected values."""
        assert tg.TestFramework.PYTEST.value == "pytest"
        assert tg.TestFramework.UNITTEST.value == "unittest"
        assert tg.TestFramework.JEST.value == "jest"
        assert tg.TestFramework.GOTEST.value == "gotest"

    def test_test_type_values(self):
        """Test TestType enum has expected values."""
        assert tg.TestType.UNIT.value == "unit"
        assert tg.TestType.INTEGRATION.value == "integration"
        assert tg.TestType.EDGE_CASES.value == "edge_cases"

    def test_llm_provider_values(self):
        """Test LLMProvider enum has expected values."""
        assert tg.LLMProvider.ANTHROPIC.value == "anthropic"
        assert tg.LLMProvider.OPENAI.value == "openai"
        assert tg.LLMProvider.GOOGLE.value == "google"


class TestExitCodes:
    """Tests verifying correct exit codes."""

    def test_exit_code_constants(self):
        """Test exit code constant values."""
        assert tg.EXIT_SUCCESS == 0
        assert tg.EXIT_FILE_NOT_FOUND == 1
        assert tg.EXIT_API_ERROR == 2
        assert tg.EXIT_INVALID_ARGS == 3


class TestVerboseMode:
    """Tests for verbose output mode."""

    @patch("test_generator.call_anthropic")
    def test_verbose_prints_info(self, mock_call):
        """Test that verbose mode prints additional info."""
        mock_call.return_value = "def test_example(): pass"

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def example(): pass")
            f.flush()

            config = tg.TestGenerationConfig(
                source_file=Path(f.name),
                framework=tg.TestFramework.PYTEST,
                test_type=tg.TestType.UNIT,
                include_edge_cases=False,
                analyze_dir=None,
                suggest_missing=False,
                verbose=True,
                llm_config=tg.LLMConfig(
                    provider=tg.LLMProvider.ANTHROPIC,
                    api_key="test-key",
                    model="claude-3"
                ),
                existing_patterns=""
            )

            stderr_capture = StringIO()
            with patch.object(sys, "stderr", stderr_capture):
                tg.generate_tests(config)

            stderr_output = stderr_capture.getvalue()
            assert "Detected language: python" in stderr_output
            assert "Using framework: pytest" in stderr_output
            os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
