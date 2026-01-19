#!/usr/bin/env python3
"""
Comprehensive tests for doc_generator CLI tool.

Tests cover:
- CLI argument parsing
- File collection
- LLM provider selection
- Documentation generation with mocked API calls
- Output formatting
- Error handling and exit codes
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Import the module under test
from doc_generator import (
    ExitCode,
    OutputFormat,
    DocType,
    SourceFile,
    DocumentationRequest,
    DocumentationResult,
    DocumentationGenerator,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    get_provider,
    collect_source_files,
    format_output,
    create_parser,
    main,
    LANGUAGE_EXTENSIONS,
)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def temp_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample Python file
        py_file = Path(tmpdir) / "sample.py"
        py_file.write_text("""
def hello(name: str) -> str:
    '''Say hello to someone.'''
    return f"Hello, {name}!"

class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
""")

        # Create sample JavaScript file
        js_file = Path(tmpdir) / "utils.js"
        js_file.write_text("""
function formatDate(date) {
    return date.toISOString();
}

export { formatDate };
""")

        # Create sample TypeScript file
        ts_file = Path(tmpdir) / "types.ts"
        ts_file.write_text("""
interface User {
    id: number;
    name: string;
}

function getUser(id: number): User {
    return { id, name: 'Test' };
}
""")

        # Create sample Go file
        go_file = Path(tmpdir) / "main.go"
        go_file.write_text("""
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
""")

        # Create sample Rust file
        rs_file = Path(tmpdir) / "lib.rs"
        rs_file.write_text("""
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
""")

        # Create a subdirectory with more files
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("x = 1")

        yield tmpdir


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return "# Sample Documentation\n\nThis is generated documentation."


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response."""
    return "# API Reference\n\nDetailed API documentation."


# =============================================================================
# Test: SourceFile
# =============================================================================
class TestSourceFile:
    """Tests for SourceFile class."""

    def test_from_path_python(self, temp_dir):
        """Test loading a Python file."""
        path = Path(temp_dir) / "sample.py"
        source = SourceFile.from_path(path)

        assert source.path == path
        assert source.language == "python"
        assert "def hello" in source.content

    def test_from_path_javascript(self, temp_dir):
        """Test loading a JavaScript file."""
        path = Path(temp_dir) / "utils.js"
        source = SourceFile.from_path(path)

        assert source.language == "javascript"
        assert "formatDate" in source.content

    def test_from_path_typescript(self, temp_dir):
        """Test loading a TypeScript file."""
        path = Path(temp_dir) / "types.ts"
        source = SourceFile.from_path(path)

        assert source.language == "typescript"
        assert "interface User" in source.content

    def test_from_path_go(self, temp_dir):
        """Test loading a Go file."""
        path = Path(temp_dir) / "main.go"
        source = SourceFile.from_path(path)

        assert source.language == "go"
        assert "package main" in source.content

    def test_from_path_rust(self, temp_dir):
        """Test loading a Rust file."""
        path = Path(temp_dir) / "lib.rs"
        source = SourceFile.from_path(path)

        assert source.language == "rust"
        assert "pub fn add" in source.content

    def test_from_path_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            SourceFile.from_path(Path("/nonexistent/file.py"))

    def test_from_path_unsupported_extension(self, temp_dir):
        """Test ValueError for unsupported file type."""
        path = Path(temp_dir) / "readme.txt"
        path.write_text("Some text")

        with pytest.raises(ValueError, match="Unsupported file type"):
            SourceFile.from_path(path)


# =============================================================================
# Test: File Collection
# =============================================================================
class TestCollectSourceFiles:
    """Tests for collect_source_files function."""

    def test_collect_single_file(self, temp_dir):
        """Test collecting a single file."""
        files = collect_source_files([f"{temp_dir}/sample.py"])

        assert len(files) == 1
        assert files[0].language == "python"

    def test_collect_directory(self, temp_dir):
        """Test collecting files from a directory."""
        files = collect_source_files([temp_dir])

        # Should find all supported files including nested
        assert len(files) >= 5  # py, js, ts, go, rs files

    def test_collect_directory_non_recursive(self, temp_dir):
        """Test non-recursive directory collection."""
        files = collect_source_files([temp_dir], recursive=False)

        # Should not include nested.py from subdir
        nested_files = [f for f in files if "nested" in str(f.path)]
        assert len(nested_files) == 0

    def test_collect_multiple_paths(self, temp_dir):
        """Test collecting from multiple paths."""
        files = collect_source_files([
            f"{temp_dir}/sample.py",
            f"{temp_dir}/utils.js",
        ])

        assert len(files) == 2

    def test_collect_nonexistent_path(self, temp_dir, capsys):
        """Test warning for nonexistent path."""
        files = collect_source_files([f"{temp_dir}/nonexistent.py"])

        assert len(files) == 0
        captured = capsys.readouterr()
        assert "Warning" in captured.err


# =============================================================================
# Test: LLM Providers
# =============================================================================
class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_is_available_with_key(self):
        """Test availability check with API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
            assert provider.is_available() is True

    def test_is_available_without_key(self):
        """Test availability check without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the key is not set
            os.environ.pop("OPENAI_API_KEY", None)
            provider = OpenAIProvider()
            assert provider.is_available() is False

    def test_generate_success(self, mock_openai_response):
        """Test successful generation with mocked API."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = mock_openai_response
            mock_client.chat.completions.create.return_value = mock_response

            # Create a mock module with OpenAI class
            mock_openai_module = MagicMock()
            mock_openai_module.OpenAI.return_value = mock_client

            with patch.dict(sys.modules, {"openai": mock_openai_module}):
                result = provider.generate("Test prompt")

            assert result == mock_openai_response

    def test_generate_import_error(self):
        """Test ImportError when openai not installed."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()

            with patch.dict(sys.modules, {"openai": None}):
                with pytest.raises(ImportError):
                    provider.generate("Test prompt")


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_is_available_with_key(self):
        """Test availability check with API key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider()
            assert provider.is_available() is True

    def test_is_available_without_key(self):
        """Test availability check without API key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            provider = AnthropicProvider()
            assert provider.is_available() is False

    def test_generate_success(self, mock_anthropic_response):
        """Test successful generation with mocked API."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider()

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = mock_anthropic_response
            mock_client.messages.create.return_value = mock_response

            # Create a mock module with Anthropic class
            mock_anthropic_module = MagicMock()
            mock_anthropic_module.Anthropic.return_value = mock_client

            with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}):
                result = provider.generate("Test prompt")

            assert result == mock_anthropic_response


class TestGoogleProvider:
    """Tests for Google provider."""

    def test_is_available_with_key(self):
        """Test availability check with API key."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            provider = GoogleProvider()
            assert provider.is_available() is True

    def test_is_available_without_key(self):
        """Test availability check without API key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_API_KEY", None)
            provider = GoogleProvider()
            assert provider.is_available() is False


class TestGetProvider:
    """Tests for get_provider function."""

    def test_get_openai_provider(self):
        """Test getting OpenAI provider."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            provider = get_provider()
            assert isinstance(provider, OpenAIProvider)

    def test_get_anthropic_provider(self):
        """Test getting Anthropic provider when forced."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = get_provider(force_provider="anthropic")
            assert isinstance(provider, AnthropicProvider)

    def test_get_provider_with_env_override(self):
        """Test provider selection via environment variable."""
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "DOC_GENERATOR_PROVIDER": "anthropic",
        }):
            provider = get_provider()
            assert isinstance(provider, AnthropicProvider)

    def test_get_provider_unknown(self):
        """Test error for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider(force_provider="unknown")

    def test_get_provider_no_key(self):
        """Test error when no API key available."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            with pytest.raises(RuntimeError, match="No LLM API key found"):
                get_provider()


# =============================================================================
# Test: Documentation Generator
# =============================================================================
class TestDocumentationGenerator:
    """Tests for DocumentationGenerator class."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock(spec=OpenAIProvider)
        provider.generate.return_value = "# Generated Documentation\n\nContent here."
        return provider

    @pytest.fixture
    def sample_source_file(self, temp_dir):
        """Create a sample source file."""
        return SourceFile.from_path(Path(temp_dir) / "sample.py")

    def test_generate_module_docs(self, mock_provider, sample_source_file):
        """Test module documentation generation."""
        generator = DocumentationGenerator(mock_provider)
        request = DocumentationRequest(
            source_files=[sample_source_file],
            doc_type=DocType.MODULE,
            output_format=OutputFormat.MARKDOWN,
        )

        result = generator.generate(request)

        assert isinstance(result, DocumentationResult)
        assert result.format == OutputFormat.MARKDOWN
        assert len(result.source_files) == 1
        mock_provider.generate.assert_called_once()

    def test_generate_api_reference(self, mock_provider, sample_source_file):
        """Test API reference generation."""
        generator = DocumentationGenerator(mock_provider)
        request = DocumentationRequest(
            source_files=[sample_source_file],
            doc_type=DocType.API_REFERENCE,
            output_format=OutputFormat.MARKDOWN,
        )

        result = generator.generate(request)

        assert result.format == OutputFormat.MARKDOWN
        mock_provider.generate.assert_called_once()

    def test_generate_openapi(self, mock_provider, sample_source_file):
        """Test OpenAPI spec generation."""
        mock_provider.generate.return_value = "openapi: '3.0.0'\ninfo:\n  title: API"
        generator = DocumentationGenerator(mock_provider)
        request = DocumentationRequest(
            source_files=[sample_source_file],
            doc_type=DocType.OPENAPI,
            output_format=OutputFormat.YAML,
        )

        result = generator.generate(request)

        assert result.format == OutputFormat.YAML
        assert "openapi" in result.content

    def test_generate_readme(self, mock_provider, sample_source_file):
        """Test README generation."""
        generator = DocumentationGenerator(mock_provider)
        request = DocumentationRequest(
            source_files=[sample_source_file],
            doc_type=DocType.README,
            output_format=OutputFormat.MARKDOWN,
        )

        result = generator.generate(request)

        assert result.format == OutputFormat.MARKDOWN

    def test_generate_docstrings(self, mock_provider, sample_source_file):
        """Test docstring generation."""
        mock_provider.generate.return_value = '''```python
def hello(name: str) -> str:
    """Say hello to someone.

    Args:
        name: The name to greet.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"
```'''
        generator = DocumentationGenerator(mock_provider)
        request = DocumentationRequest(
            source_files=[sample_source_file],
            doc_type=DocType.DOCSTRINGS,
            output_format=OutputFormat.MARKDOWN,
        )

        result = generator.generate(request)

        assert "def hello" in result.content

    def test_generate_with_api_error(self, mock_provider, sample_source_file):
        """Test handling of API errors."""
        mock_provider.generate.side_effect = RuntimeError("API error")
        generator = DocumentationGenerator(mock_provider)
        request = DocumentationRequest(
            source_files=[sample_source_file],
            doc_type=DocType.MODULE,
            output_format=OutputFormat.MARKDOWN,
        )

        result = generator.generate(request)

        assert len(result.warnings) > 0
        assert "Failed to process" in result.warnings[0]


# =============================================================================
# Test: Output Formatting
# =============================================================================
class TestFormatOutput:
    """Tests for format_output function."""

    def test_format_markdown(self):
        """Test markdown output formatting."""
        result = DocumentationResult(
            content="# Title\n\nContent",
            format=OutputFormat.MARKDOWN,
        )

        output = format_output(result, OutputFormat.MARKDOWN)

        assert output == "# Title\n\nContent"

    def test_format_html_conversion(self):
        """Test HTML output with markdown content."""
        result = DocumentationResult(
            content="# Title\n\nContent",
            format=OutputFormat.MARKDOWN,
        )

        output = format_output(result, OutputFormat.HTML)

        assert "<!DOCTYPE html>" in output
        assert "<html>" in output
        assert "# Title" in output


# =============================================================================
# Test: CLI Parser
# =============================================================================
class TestCLIParser:
    """Tests for CLI argument parser."""

    def test_parser_help(self):
        """Test that parser has help text."""
        parser = create_parser()
        # Check description exists
        assert parser.description is not None
        assert "Generate documentation" in parser.description

    def test_parser_basic_args(self):
        """Test parsing basic arguments."""
        parser = create_parser()
        args = parser.parse_args(["file.py"])

        assert args.paths == ["file.py"]
        assert args.format == "markdown"

    def test_parser_format_option(self):
        """Test format option parsing."""
        parser = create_parser()

        for fmt in ["markdown", "rst", "html"]:
            args = parser.parse_args(["file.py", "--format", fmt])
            assert args.format == fmt

    def test_parser_type_option(self):
        """Test type option parsing."""
        parser = create_parser()

        for doc_type in ["module", "api-reference", "openapi"]:
            args = parser.parse_args(["file.py", "--type", doc_type])
            assert args.type == doc_type

    def test_parser_readme_flag(self):
        """Test --readme flag."""
        parser = create_parser()
        args = parser.parse_args(["--readme", "--include", "src/"])

        assert args.readme is True
        assert args.include == ["src/"]

    def test_parser_docstrings_flag(self):
        """Test --docstrings flag."""
        parser = create_parser()
        args = parser.parse_args(["file.py", "--docstrings"])

        assert args.docstrings is True

    def test_parser_output_option(self):
        """Test -o/--output option."""
        parser = create_parser()
        args = parser.parse_args(["file.py", "-o", "output.md"])

        assert args.output == "output.md"

    def test_parser_provider_option(self):
        """Test --provider option."""
        parser = create_parser()
        args = parser.parse_args(["file.py", "--provider", "anthropic"])

        assert args.provider == "anthropic"

    def test_parser_verbose_flag(self):
        """Test -v/--verbose flag."""
        parser = create_parser()
        args = parser.parse_args(["file.py", "-v"])

        assert args.verbose is True

    def test_parser_dry_run_flag(self):
        """Test --dry-run flag."""
        parser = create_parser()
        args = parser.parse_args(["file.py", "--dry-run"])

        assert args.dry_run is True

    def test_parser_mutually_exclusive_type_flags(self):
        """Test that --readme and --docstrings are mutually exclusive."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["file.py", "--readme", "--docstrings"])


# =============================================================================
# Test: Main Function
# =============================================================================
class TestMain:
    """Tests for main entry point."""

    def test_main_no_args(self, capsys):
        """Test error when no arguments provided."""
        # parser.error calls sys.exit(2), so we expect SystemExit
        with pytest.raises(SystemExit) as exc_info:
            main([])

        # argparse exits with code 2 for argument errors
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()

    def test_main_file_not_found(self, capsys):
        """Test FILE_NOT_FOUND exit code."""
        result = main(["/nonexistent/path/file.py"])

        assert result == ExitCode.FILE_NOT_FOUND.value

    def test_main_dry_run(self, temp_dir, capsys):
        """Test dry run mode."""
        result = main([f"{temp_dir}/sample.py", "--dry-run"])

        assert result == ExitCode.SUCCESS.value
        captured = capsys.readouterr()
        assert "Dry run" in captured.err

    def test_main_no_api_key(self, temp_dir, capsys):
        """Test API_ERROR when no API key set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)

            result = main([f"{temp_dir}/sample.py"])

        assert result == ExitCode.API_ERROR.value

    def test_main_success_with_mocked_provider(self, temp_dir, capsys):
        """Test successful execution with mocked provider."""
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.generate.return_value = "# Documentation"

        with patch("doc_generator.get_provider", return_value=mock_provider):
            result = main([f"{temp_dir}/sample.py"])

        assert result == ExitCode.SUCCESS.value
        captured = capsys.readouterr()
        assert "Documentation" in captured.out

    def test_main_verbose_output(self, temp_dir, capsys):
        """Test verbose mode output."""
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.generate.return_value = "# Documentation"

        with patch("doc_generator.get_provider", return_value=mock_provider):
            result = main([f"{temp_dir}/sample.py", "--verbose"])

        assert result == ExitCode.SUCCESS.value
        captured = capsys.readouterr()
        assert "Found" in captured.err
        assert "source files" in captured.err

    def test_main_with_output_file(self, temp_dir, capsys):
        """Test writing output to file."""
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.generate.return_value = "# Documentation"

        output_file = Path(temp_dir) / "output.md"

        with patch("doc_generator.get_provider", return_value=mock_provider):
            result = main([f"{temp_dir}/sample.py", "-o", str(output_file)])

        assert result == ExitCode.SUCCESS.value
        assert output_file.exists()
        assert "Documentation" in output_file.read_text()

    def test_main_readme_mode(self, temp_dir, capsys):
        """Test README generation mode."""
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.generate.return_value = "# Project README"

        with patch("doc_generator.get_provider", return_value=mock_provider):
            result = main(["--readme", "--include", temp_dir])

        assert result == ExitCode.SUCCESS.value
        captured = capsys.readouterr()
        assert "README" in captured.out

    def test_main_docstrings_mode(self, temp_dir, capsys):
        """Test docstring generation mode."""
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.generate.return_value = 'def hello():\n    """Hello docstring."""\n    pass'

        with patch("doc_generator.get_provider", return_value=mock_provider):
            result = main([f"{temp_dir}/sample.py", "--docstrings"])

        assert result == ExitCode.SUCCESS.value

    def test_main_html_format(self, temp_dir, capsys):
        """Test HTML output format."""
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.generate.return_value = "# Documentation"

        with patch("doc_generator.get_provider", return_value=mock_provider):
            result = main([f"{temp_dir}/sample.py", "--format", "html"])

        assert result == ExitCode.SUCCESS.value
        captured = capsys.readouterr()
        assert "<html>" in captured.out

    def test_main_api_error_handling(self, temp_dir, capsys):
        """Test API error handling."""
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.generate.side_effect = RuntimeError("API rate limit exceeded")

        with patch("doc_generator.get_provider", return_value=mock_provider):
            result = main([f"{temp_dir}/sample.py"])

        # Should still succeed but with warnings (errors are per-file)
        # The current implementation catches errors per file
        assert result == ExitCode.SUCCESS.value


# =============================================================================
# Test: Exit Codes
# =============================================================================
class TestExitCodes:
    """Tests for exit code values."""

    def test_exit_code_values(self):
        """Test that exit codes have correct values."""
        assert ExitCode.SUCCESS.value == 0
        assert ExitCode.FILE_NOT_FOUND.value == 1
        assert ExitCode.API_ERROR.value == 2
        assert ExitCode.INVALID_ARGS.value == 3


# =============================================================================
# Test: Language Extensions
# =============================================================================
class TestLanguageExtensions:
    """Tests for language extension mappings."""

    def test_python_extensions(self):
        """Test Python file extension."""
        assert LANGUAGE_EXTENSIONS[".py"] == "python"

    def test_javascript_extensions(self):
        """Test JavaScript file extensions."""
        assert LANGUAGE_EXTENSIONS[".js"] == "javascript"
        assert LANGUAGE_EXTENSIONS[".jsx"] == "javascript"

    def test_typescript_extensions(self):
        """Test TypeScript file extensions."""
        assert LANGUAGE_EXTENSIONS[".ts"] == "typescript"
        assert LANGUAGE_EXTENSIONS[".tsx"] == "typescript"

    def test_go_extension(self):
        """Test Go file extension."""
        assert LANGUAGE_EXTENSIONS[".go"] == "go"

    def test_rust_extension(self):
        """Test Rust file extension."""
        assert LANGUAGE_EXTENSIONS[".rs"] == "rust"


# =============================================================================
# Test: Integration
# =============================================================================
class TestIntegration:
    """Integration tests."""

    def test_full_workflow_mocked(self, temp_dir, capsys):
        """Test complete workflow with mocked API."""
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.generate.return_value = """# sample.py

## Overview

This module provides utilities for greeting and calculation.

## Functions

### hello(name: str) -> str

Say hello to someone.

**Parameters:**
- `name` (str): The name to greet.

**Returns:**
- str: A greeting message.

## Classes

### Calculator

A simple calculator class.

#### Methods

##### add(a: int, b: int) -> int

Add two numbers together.
"""

        with patch("doc_generator.get_provider", return_value=mock_provider):
            result = main([
                f"{temp_dir}/sample.py",
                "--format", "markdown",
                "--type", "api-reference",
                "-v",
            ])

        assert result == ExitCode.SUCCESS.value
        captured = capsys.readouterr()

        # Check that documentation was generated
        assert "sample.py" in captured.out
        assert "Overview" in captured.out or "Functions" in captured.out


# =============================================================================
# Run Tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
