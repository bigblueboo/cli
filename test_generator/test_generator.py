#!/usr/bin/env python3
"""
test_generator - AI-powered test generation CLI tool

A self-documenting CLI tool that generates comprehensive tests from source code
using Large Language Models (LLMs). Supports multiple programming languages and
test frameworks with intelligent pattern detection.

SUPPORTED TEST FRAMEWORKS:
    Python:     pytest, unittest
    JavaScript: jest, mocha, vitest
    Go:         go test (gotest)
    Rust:       cargo test
    Java:       junit
    Ruby:       rspec, minitest

ENVIRONMENT VARIABLES:
    OPENAI_API_KEY      - OpenAI API key (GPT-4, GPT-3.5)
    ANTHROPIC_API_KEY   - Anthropic API key (Claude)
    GOOGLE_API_KEY      - Google API key (Gemini)

    At least one API key must be set. Priority: Anthropic > OpenAI > Google

EXIT CODES:
    0 - Success
    1 - File not found
    2 - API error (rate limit, auth failure, network)
    3 - Invalid arguments

EXAMPLES:
    # Generate pytest tests for a Python module
    test_generator src/utils.py --framework pytest > tests/test_utils.py

    # Generate tests with edge cases
    test_generator src/utils.py --framework pytest --edge-cases

    # Generate integration tests
    test_generator api/handlers.py --type integration

    # Generate Jest tests for JavaScript
    test_generator src/auth.js --framework jest > tests/auth.test.js

    # Analyze test coverage and suggest missing tests
    test_generator --analyze tests/ --suggest-missing

    # Generate Go tests
    test_generator func.go --framework gotest

    # Use specific LLM provider
    test_generator src/utils.py --provider anthropic --model claude-3-opus-20240229

    # Verbose output with reasoning
    test_generator src/utils.py --verbose

Author: Auto-generated CLI tool
License: MIT
"""

import argparse
import os
import sys
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import textwrap

__version__ = "1.0.0"

# Exit codes
EXIT_SUCCESS = 0
EXIT_FILE_NOT_FOUND = 1
EXIT_API_ERROR = 2
EXIT_INVALID_ARGS = 3


class TestFramework(Enum):
    """Supported test frameworks."""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    VITEST = "vitest"
    GOTEST = "gotest"
    CARGO_TEST = "cargo"
    JUNIT = "junit"
    RSPEC = "rspec"
    MINITEST = "minitest"


class TestType(Enum):
    """Types of tests that can be generated."""
    UNIT = "unit"
    INTEGRATION = "integration"
    EDGE_CASES = "edge_cases"
    PROPERTY = "property"
    SNAPSHOT = "snapshot"


class LLMProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: LLMProvider
    api_key: str
    model: str


@dataclass
class TestGenerationConfig:
    """Configuration for test generation."""
    source_file: Optional[Path]
    framework: Optional[TestFramework]
    test_type: TestType
    include_edge_cases: bool
    analyze_dir: Optional[Path]
    suggest_missing: bool
    verbose: bool
    llm_config: LLMConfig
    existing_patterns: Optional[str]


# Language detection based on file extension
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".swift": "swift",
    ".kt": "kotlin",
}

# Default frameworks per language
DEFAULT_FRAMEWORK = {
    "python": TestFramework.PYTEST,
    "javascript": TestFramework.JEST,
    "typescript": TestFramework.JEST,
    "go": TestFramework.GOTEST,
    "rust": TestFramework.CARGO_TEST,
    "java": TestFramework.JUNIT,
    "ruby": TestFramework.RSPEC,
}

# Framework to language mapping
FRAMEWORK_LANGUAGE = {
    TestFramework.PYTEST: "python",
    TestFramework.UNITTEST: "python",
    TestFramework.JEST: "javascript",
    TestFramework.MOCHA: "javascript",
    TestFramework.VITEST: "javascript",
    TestFramework.GOTEST: "go",
    TestFramework.CARGO_TEST: "rust",
    TestFramework.JUNIT: "java",
    TestFramework.RSPEC: "ruby",
    TestFramework.MINITEST: "ruby",
}


def detect_language(file_path: Path) -> Optional[str]:
    """Detect programming language from file extension."""
    return EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower())


def detect_framework(file_path: Path, language: str) -> TestFramework:
    """
    Auto-detect the appropriate test framework based on language and project structure.

    Checks for:
    - Package configuration files (package.json, pyproject.toml, etc.)
    - Existing test files and their patterns
    - Default framework for the language
    """
    parent_dir = file_path.parent

    # Check for Python frameworks
    if language == "python":
        # Check pyproject.toml or setup.cfg for pytest
        pyproject = parent_dir / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            if "pytest" in content:
                return TestFramework.PYTEST

        # Check for conftest.py (pytest indicator)
        if (parent_dir / "conftest.py").exists():
            return TestFramework.PYTEST

        # Check existing test files
        for test_file in parent_dir.glob("test_*.py"):
            content = test_file.read_text()
            if "import pytest" in content or "@pytest" in content:
                return TestFramework.PYTEST
            if "import unittest" in content or "class.*TestCase" in content:
                return TestFramework.UNITTEST

    # Check for JavaScript frameworks
    elif language in ("javascript", "typescript"):
        package_json = parent_dir / "package.json"
        if package_json.exists():
            try:
                pkg = json.loads(package_json.read_text())
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                if "jest" in deps:
                    return TestFramework.JEST
                if "mocha" in deps:
                    return TestFramework.MOCHA
                if "vitest" in deps:
                    return TestFramework.VITEST
            except json.JSONDecodeError:
                pass

    # Return default for language
    return DEFAULT_FRAMEWORK.get(language, TestFramework.PYTEST)


def detect_existing_patterns(test_dir: Path) -> str:
    """
    Analyze existing test files to detect coding patterns and conventions.

    Returns a summary of patterns found for the LLM to follow.
    """
    patterns = []

    if not test_dir.exists():
        return ""

    test_files = list(test_dir.glob("**/*test*.py")) + \
                 list(test_dir.glob("**/test_*.py")) + \
                 list(test_dir.glob("**/*.test.js")) + \
                 list(test_dir.glob("**/*.spec.js"))

    if not test_files:
        return ""

    # Analyze first few test files
    for test_file in test_files[:5]:
        try:
            content = test_file.read_text()

            # Check for fixtures
            if "@pytest.fixture" in content:
                patterns.append("Uses pytest fixtures")

            # Check for parametrize
            if "@pytest.mark.parametrize" in content:
                patterns.append("Uses parametrized tests")

            # Check for mocking
            if "unittest.mock" in content or "from mock import" in content:
                patterns.append("Uses unittest.mock for mocking")
            if "pytest-mock" in content or "mocker" in content:
                patterns.append("Uses pytest-mock (mocker fixture)")

            # Check for async tests
            if "@pytest.mark.asyncio" in content or "async def test_" in content:
                patterns.append("Uses async tests with pytest-asyncio")

            # Check naming conventions
            if re.search(r"def test_\w+_should_\w+", content):
                patterns.append("Uses 'test_X_should_Y' naming convention")
            elif re.search(r"def test_\w+_when_\w+", content):
                patterns.append("Uses 'test_X_when_Y' naming convention")

            # Check for docstrings
            if re.search(r'def test_\w+\([^)]*\):\s*"""', content):
                patterns.append("Tests include docstrings")

            # Check for arrange-act-assert pattern
            if "# Arrange" in content or "# Act" in content or "# Assert" in content:
                patterns.append("Uses Arrange-Act-Assert comments")

        except Exception:
            continue

    return "; ".join(set(patterns)) if patterns else ""


def get_llm_config() -> LLMConfig:
    """
    Get LLM configuration from environment variables.

    Priority: Anthropic > OpenAI > Google

    Raises:
        SystemExit: If no API key is found
    """
    # Check Anthropic first
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        return LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key=api_key,
            model="claude-3-5-sonnet-20241022"
        )

    # Check OpenAI
    if api_key := os.environ.get("OPENAI_API_KEY"):
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key=api_key,
            model="gpt-4-turbo-preview"
        )

    # Check Google
    if api_key := os.environ.get("GOOGLE_API_KEY"):
        return LLMConfig(
            provider=LLMProvider.GOOGLE,
            api_key=api_key,
            model="gemini-pro"
        )

    print("Error: No API key found. Set one of:", file=sys.stderr)
    print("  ANTHROPIC_API_KEY", file=sys.stderr)
    print("  OPENAI_API_KEY", file=sys.stderr)
    print("  GOOGLE_API_KEY", file=sys.stderr)
    sys.exit(EXIT_INVALID_ARGS)


def build_prompt(
    source_code: str,
    language: str,
    framework: TestFramework,
    test_type: TestType,
    include_edge_cases: bool,
    existing_patterns: str
) -> str:
    """Build the prompt for test generation."""

    framework_examples = {
        TestFramework.PYTEST: '''
import pytest
from module import function_name

class TestFunctionName:
    """Tests for function_name."""

    def test_basic_functionality(self):
        """Test basic expected behavior."""
        result = function_name(input_value)
        assert result == expected_value

    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
    ])
    def test_parametrized(self, input_val, expected):
        """Test with multiple inputs."""
        assert function_name(input_val) == expected
''',
        TestFramework.UNITTEST: '''
import unittest
from module import function_name

class TestFunctionName(unittest.TestCase):
    """Tests for function_name."""

    def test_basic_functionality(self):
        """Test basic expected behavior."""
        result = function_name(input_value)
        self.assertEqual(result, expected_value)

    def test_raises_error(self):
        """Test error handling."""
        with self.assertRaises(ValueError):
            function_name(invalid_input)
''',
        TestFramework.JEST: '''
const { functionName } = require('./module');

describe('functionName', () => {
    it('should return expected value for basic input', () => {
        expect(functionName(input)).toBe(expected);
    });

    it('should throw error for invalid input', () => {
        expect(() => functionName(invalid)).toThrow();
    });

    it('should handle edge case', () => {
        expect(functionName(edgeCase)).toBe(expectedEdge);
    });
});
''',
        TestFramework.GOTEST: '''
package mypackage

import (
    "testing"
)

func TestFunctionName(t *testing.T) {
    tests := []struct {
        name     string
        input    InputType
        expected OutputType
    }{
        {"basic case", input1, expected1},
        {"edge case", input2, expected2},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := FunctionName(tt.input)
            if result != tt.expected {
                t.Errorf("FunctionName(%v) = %v, want %v", tt.input, result, tt.expected)
            }
        })
    }
}
''',
    }

    prompt_parts = [
        f"Generate comprehensive {test_type.value} tests for the following {language} code.",
        f"\nUse the {framework.value} testing framework.",
    ]

    if existing_patterns:
        prompt_parts.append(f"\nFollow these existing patterns from the codebase: {existing_patterns}")

    if include_edge_cases:
        prompt_parts.append("""
Include edge case tests for:
- Empty inputs (empty strings, empty lists, None/null)
- Boundary values (0, -1, max int, min int)
- Invalid types
- Concurrent/async edge cases if applicable
- Error conditions and exception handling
""")

    if test_type == TestType.INTEGRATION:
        prompt_parts.append("""
For integration tests:
- Test interactions between components
- Mock external services and APIs
- Test database interactions if present
- Test API endpoints end-to-end
- Include setup and teardown for test fixtures
""")

    if framework in framework_examples:
        prompt_parts.append(f"\nExample format:\n```{language}\n{framework_examples[framework]}```")

    prompt_parts.append(f"\n\nSource code to test:\n```{language}\n{source_code}\n```")

    prompt_parts.append("""
Requirements:
1. Generate complete, runnable test code
2. Include necessary imports
3. Add descriptive test names and docstrings
4. Cover all public functions and methods
5. Include both positive and negative test cases
6. Use appropriate mocking for external dependencies
7. Output ONLY the test code, no explanations
""")

    return "\n".join(prompt_parts)


def call_anthropic(config: LLMConfig, prompt: str) -> str:
    """Call Anthropic API to generate tests."""
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)

    try:
        client = anthropic.Anthropic(api_key=config.api_key)
        message = client.messages.create(
            model=config.model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except anthropic.APIConnectionError as e:
        print(f"API Connection Error: {e}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)
    except anthropic.RateLimitError as e:
        print(f"Rate Limit Error: {e}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)
    except anthropic.APIStatusError as e:
        print(f"API Error: {e}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)


def call_openai(config: LLMConfig, prompt: str) -> str:
    """Call OpenAI API to generate tests."""
    try:
        import openai
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)

    try:
        client = openai.OpenAI(api_key=config.api_key)
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": "You are an expert test engineer. Generate comprehensive, well-structured tests."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096
        )
        return response.choices[0].message.content
    except openai.APIConnectionError as e:
        print(f"API Connection Error: {e}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)
    except openai.RateLimitError as e:
        print(f"Rate Limit Error: {e}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)
    except openai.APIStatusError as e:
        print(f"API Error: {e}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)


def call_google(config: LLMConfig, prompt: str) -> str:
    """Call Google Gemini API to generate tests."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("Error: google-generativeai package not installed. Run: pip install google-generativeai", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)

    try:
        genai.configure(api_key=config.api_key)
        model = genai.GenerativeModel(config.model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Google API Error: {e}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)


def generate_tests(config: TestGenerationConfig) -> str:
    """Generate tests using the configured LLM."""

    if config.source_file is None:
        print("Error: Source file is required", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)

    if not config.source_file.exists():
        print(f"Error: File not found: {config.source_file}", file=sys.stderr)
        sys.exit(EXIT_FILE_NOT_FOUND)

    source_code = config.source_file.read_text()
    language = detect_language(config.source_file)

    if language is None:
        print(f"Error: Unknown file type: {config.source_file.suffix}", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)

    framework = config.framework or detect_framework(config.source_file, language)

    if config.verbose:
        print(f"Detected language: {language}", file=sys.stderr)
        print(f"Using framework: {framework.value}", file=sys.stderr)
        if config.existing_patterns:
            print(f"Existing patterns: {config.existing_patterns}", file=sys.stderr)

    prompt = build_prompt(
        source_code=source_code,
        language=language,
        framework=framework,
        test_type=config.test_type,
        include_edge_cases=config.include_edge_cases,
        existing_patterns=config.existing_patterns or ""
    )

    if config.verbose:
        print("Generating tests...", file=sys.stderr)

    # Call appropriate LLM
    if config.llm_config.provider == LLMProvider.ANTHROPIC:
        result = call_anthropic(config.llm_config, prompt)
    elif config.llm_config.provider == LLMProvider.OPENAI:
        result = call_openai(config.llm_config, prompt)
    else:
        result = call_google(config.llm_config, prompt)

    # Extract code from markdown code blocks if present
    code_block_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_block_pattern, result, re.DOTALL)
    if matches:
        return "\n\n".join(matches)

    return result


def analyze_test_coverage(test_dir: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze existing tests and suggest missing coverage.

    Returns a dictionary with analysis results.
    """
    if not test_dir.exists():
        print(f"Error: Directory not found: {test_dir}", file=sys.stderr)
        sys.exit(EXIT_FILE_NOT_FOUND)

    analysis = {
        "test_files": [],
        "tested_modules": set(),
        "suggestions": [],
        "coverage_gaps": []
    }

    # Find all test files
    test_patterns = ["**/test_*.py", "**/*_test.py", "**/*.test.js", "**/*.spec.js", "**/*_test.go"]
    for pattern in test_patterns:
        for test_file in test_dir.glob(pattern):
            analysis["test_files"].append(str(test_file))

            # Extract tested module names
            content = test_file.read_text()

            # Python imports
            for match in re.finditer(r"from (\S+) import", content):
                analysis["tested_modules"].add(match.group(1))

            # JavaScript/TypeScript imports
            for match in re.finditer(r"require\(['\"]([^'\"]+)['\"]\)", content):
                analysis["tested_modules"].add(match.group(1))
            for match in re.finditer(r"from ['\"]([^'\"]+)['\"]", content):
                analysis["tested_modules"].add(match.group(1))

    # Find source files that might need tests
    parent_dir = test_dir.parent
    source_patterns = ["**/*.py", "**/*.js", "**/*.ts", "**/*.go"]

    for pattern in source_patterns:
        for source_file in parent_dir.glob(pattern):
            # Skip test files and common non-source files
            if "test" in source_file.name.lower():
                continue
            if source_file.name.startswith("__"):
                continue
            if "node_modules" in str(source_file):
                continue
            if "venv" in str(source_file) or ".env" in str(source_file):
                continue

            # Check if this module has tests
            module_name = source_file.stem
            has_tests = any(
                module_name in tested for tested in analysis["tested_modules"]
            )

            if not has_tests:
                analysis["coverage_gaps"].append(str(source_file))
                analysis["suggestions"].append(
                    f"Consider adding tests for: {source_file}"
                )

    analysis["tested_modules"] = list(analysis["tested_modules"])
    return analysis


def format_analysis_output(analysis: Dict[str, Any]) -> str:
    """Format the analysis results for display."""
    output = []

    output.append("=" * 60)
    output.append("TEST COVERAGE ANALYSIS")
    output.append("=" * 60)

    output.append(f"\nFound {len(analysis['test_files'])} test file(s)")

    if analysis["test_files"]:
        output.append("\nTest files:")
        for f in analysis["test_files"][:10]:
            output.append(f"  - {f}")
        if len(analysis["test_files"]) > 10:
            output.append(f"  ... and {len(analysis['test_files']) - 10} more")

    output.append(f"\nTested modules: {len(analysis['tested_modules'])}")

    if analysis["coverage_gaps"]:
        output.append(f"\n{'=' * 60}")
        output.append("SUGGESTED MISSING TESTS")
        output.append("=" * 60)
        for gap in analysis["coverage_gaps"][:20]:
            output.append(f"  ! {gap}")
        if len(analysis["coverage_gaps"]) > 20:
            output.append(f"  ... and {len(analysis['coverage_gaps']) - 20} more")
    else:
        output.append("\nNo obvious coverage gaps found!")

    return "\n".join(output)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser with comprehensive help."""

    parser = argparse.ArgumentParser(
        prog="test_generator",
        description=textwrap.dedent("""
            AI-powered test generation CLI tool.

            Generates comprehensive tests from source code using Large Language Models.
            Supports multiple programming languages and test frameworks with intelligent
            pattern detection.
        """),
        epilog=textwrap.dedent("""
            EXAMPLES:
              Generate pytest tests:
                %(prog)s src/utils.py --framework pytest > tests/test_utils.py

              Generate tests with edge cases:
                %(prog)s src/utils.py --framework pytest --edge-cases

              Generate integration tests:
                %(prog)s api/handlers.py --type integration

              Generate Jest tests for JavaScript:
                %(prog)s src/auth.js --framework jest > tests/auth.test.js

              Analyze and suggest missing tests:
                %(prog)s --analyze tests/ --suggest-missing

              Generate Go tests:
                %(prog)s func.go --framework gotest

            EXIT CODES:
              0 - Success
              1 - File not found
              2 - API error
              3 - Invalid arguments

            ENVIRONMENT VARIABLES:
              ANTHROPIC_API_KEY - Anthropic/Claude API key (preferred)
              OPENAI_API_KEY    - OpenAI API key
              GOOGLE_API_KEY    - Google/Gemini API key
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "source_file",
        nargs="?",
        type=Path,
        help="Source file to generate tests for"
    )

    parser.add_argument(
        "-f", "--framework",
        choices=[f.value for f in TestFramework],
        help="Test framework to use (auto-detected if not specified)"
    )

    parser.add_argument(
        "-t", "--type",
        choices=[t.value for t in TestType],
        default="unit",
        help="Type of tests to generate (default: unit)"
    )

    parser.add_argument(
        "-e", "--edge-cases",
        action="store_true",
        help="Include comprehensive edge case tests"
    )

    parser.add_argument(
        "-a", "--analyze",
        type=Path,
        metavar="DIR",
        help="Analyze test directory for coverage"
    )

    parser.add_argument(
        "-s", "--suggest-missing",
        action="store_true",
        help="Suggest files that need tests (use with --analyze)"
    )

    parser.add_argument(
        "-p", "--provider",
        choices=[p.value for p in LLMProvider],
        help="LLM provider to use (auto-detected from env vars)"
    )

    parser.add_argument(
        "-m", "--model",
        help="Specific model to use (provider-dependent)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "--patterns-from",
        type=Path,
        metavar="DIR",
        help="Directory to analyze for existing test patterns"
    )

    return parser


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle analysis mode
    if args.analyze:
        analysis = analyze_test_coverage(args.analyze, args.verbose)
        print(format_analysis_output(analysis))
        return EXIT_SUCCESS

    # Require source file for generation mode
    if not args.source_file:
        parser.print_help()
        print("\nError: source_file is required for test generation", file=sys.stderr)
        return EXIT_INVALID_ARGS

    # Get LLM configuration
    llm_config = get_llm_config()

    # Override provider if specified
    if args.provider:
        provider = LLMProvider(args.provider)
        env_var = f"{provider.value.upper()}_API_KEY"
        api_key = os.environ.get(env_var)
        if not api_key:
            print(f"Error: {env_var} not set for provider {provider.value}", file=sys.stderr)
            return EXIT_INVALID_ARGS
        llm_config = LLMConfig(provider=provider, api_key=api_key, model=llm_config.model)

    # Override model if specified
    if args.model:
        llm_config = LLMConfig(
            provider=llm_config.provider,
            api_key=llm_config.api_key,
            model=args.model
        )

    # Detect existing patterns
    existing_patterns = ""
    if args.patterns_from:
        existing_patterns = detect_existing_patterns(args.patterns_from)
    elif args.source_file:
        # Try to find test directory near source
        test_dirs = ["tests", "test", "__tests__", "spec"]
        for test_dir in test_dirs:
            potential_dir = args.source_file.parent / test_dir
            if potential_dir.exists():
                existing_patterns = detect_existing_patterns(potential_dir)
                break

    # Build configuration
    config = TestGenerationConfig(
        source_file=args.source_file,
        framework=TestFramework(args.framework) if args.framework else None,
        test_type=TestType(args.type),
        include_edge_cases=args.edge_cases,
        analyze_dir=args.analyze,
        suggest_missing=args.suggest_missing,
        verbose=args.verbose,
        llm_config=llm_config,
        existing_patterns=existing_patterns
    )

    # Generate tests
    result = generate_tests(config)
    print(result)

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
