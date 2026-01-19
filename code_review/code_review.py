#!/usr/bin/env python3
"""
code_review - AI-Powered Code Review CLI Tool

A self-documenting command-line tool that leverages Large Language Models (LLMs)
to perform intelligent code reviews. Supports multiple AI providers, various input
methods, and configurable focus areas.

SUPPORTED PROVIDERS:
    - OpenAI (GPT-4, GPT-4-turbo, GPT-3.5-turbo)
    - Anthropic (Claude-3-opus, Claude-3-sonnet, Claude-3-haiku)
    - Google (Gemini-pro, Gemini-1.5-pro)

ENVIRONMENT VARIABLES:
    OPENAI_API_KEY      - API key for OpenAI models
    ANTHROPIC_API_KEY   - API key for Anthropic Claude models
    GOOGLE_API_KEY      - API key for Google Gemini models

EXIT CODES:
    0 - Success, no issues found
    1 - Issues found in code review
    2 - API error (authentication, rate limit, etc.)
    3 - Invalid arguments or input error

EXAMPLES:
    # Review a single file for security issues
    code_review -f src/auth.py --focus security

    # Review git diff with multiple focus areas
    code_review --diff HEAD~1 --focus security,performance

    # Review multiple files with markdown output
    code_review src/*.py --output markdown > review.md

    # Review staged changes, fail on high severity issues
    code_review --staged --fail-on high

    # Review from stdin (pipe git diff)
    git diff main | code_review --stdin --focus bugs

    # Use a specific model
    code_review -f api.py --model claude-3-5-sonnet

    # Review a directory recursively
    code_review -d ./src --focus style --output json

    # Set custom severity threshold
    code_review -f main.py --min-severity medium

Author: AI Code Review Tool
Version: 1.0.0
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

# Version info
__version__ = "1.0.0"
__author__ = "AI Code Review Tool"


class ExitCode(Enum):
    """Exit codes for the CLI tool."""
    SUCCESS = 0        # No issues found
    ISSUES_FOUND = 1   # Issues found in review
    API_ERROR = 2      # API-related error
    INVALID_ARGS = 3   # Invalid arguments


class Severity(Enum):
    """Severity levels for code review issues."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_string(cls, value: str) -> "Severity":
        """Convert string to Severity enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid severity: {value}. Must be one of: {', '.join(s.value for s in cls)}")

    def __ge__(self, other: "Severity") -> bool:
        order = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return order.index(self) >= order.index(other)

    def __gt__(self, other: "Severity") -> bool:
        order = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return order.index(self) > order.index(other)


class FocusArea(Enum):
    """Focus areas for code review."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    BUGS = "bugs"
    ALL = "all"

    @classmethod
    def from_string(cls, value: str) -> "FocusArea":
        """Convert string to FocusArea enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid focus area: {value}. Must be one of: {', '.join(f.value for f in cls)}")


class OutputFormat(Enum):
    """Output format options."""
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"


@dataclass
class Issue:
    """Represents a code review issue."""
    severity: str
    category: str
    file: str
    line: Optional[int]
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert issue to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ReviewResult:
    """Represents the complete review result."""
    issues: list = field(default_factory=list)
    summary: str = ""
    files_reviewed: list = field(default_factory=list)
    model_used: str = ""
    focus_areas: list = field(default_factory=list)

    def has_issues_at_or_above(self, min_severity: Severity) -> bool:
        """Check if there are issues at or above the given severity."""
        for issue in self.issues:
            try:
                issue_severity = Severity.from_string(issue.severity)
                if issue_severity >= min_severity:
                    return True
            except ValueError:
                continue
        return False

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
            "files_reviewed": self.files_reviewed,
            "model_used": self.model_used,
            "focus_areas": self.focus_areas
        }


class LLMProvider:
    """Base class for LLM providers."""

    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model

    def review(self, code: str, focus_areas: list, context: str = "") -> ReviewResult:
        """Perform code review. Must be implemented by subclasses."""
        raise NotImplementedError

    def _build_prompt(self, code: str, focus_areas: list, context: str = "") -> str:
        """Build the review prompt."""
        focus_str = ", ".join(f.value for f in focus_areas)
        if FocusArea.ALL in focus_areas:
            focus_str = "security, performance, style, bugs"

        prompt = f"""You are an expert code reviewer. Please review the following code focusing on: {focus_str}

For each issue found, provide:
1. Severity level (info, low, medium, high, critical)
2. Category (security, performance, style, bugs)
3. File name and line number (if identifiable)
4. Clear description of the issue
5. Suggested fix or improvement

{context}

CODE TO REVIEW:
```
{code}
```

Please respond in the following JSON format:
{{
    "issues": [
        {{
            "severity": "medium",
            "category": "security",
            "file": "filename.py",
            "line": 42,
            "message": "Description of the issue",
            "suggestion": "How to fix it",
            "code_snippet": "relevant code"
        }}
    ],
    "summary": "Brief overall assessment"
}}

If no issues are found, return an empty issues array with a positive summary.
"""
        return prompt

    def _parse_response(self, response_text: str, files: list, focus_areas: list) -> ReviewResult:
        """Parse the LLM response into a ReviewResult."""
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = {"issues": [], "summary": response_text}

            issues = []
            for issue_data in data.get("issues", []):
                issues.append(Issue(
                    severity=issue_data.get("severity", "info"),
                    category=issue_data.get("category", "general"),
                    file=issue_data.get("file", "unknown"),
                    line=issue_data.get("line"),
                    message=issue_data.get("message", ""),
                    suggestion=issue_data.get("suggestion"),
                    code_snippet=issue_data.get("code_snippet")
                ))

            return ReviewResult(
                issues=issues,
                summary=data.get("summary", ""),
                files_reviewed=files,
                model_used=self.model or "unknown",
                focus_areas=[f.value for f in focus_areas]
            )
        except json.JSONDecodeError:
            # If JSON parsing fails, create a single issue with the response
            return ReviewResult(
                issues=[Issue(
                    severity="info",
                    category="general",
                    file="unknown",
                    line=None,
                    message=response_text
                )],
                summary="Review completed (non-JSON response)",
                files_reviewed=files,
                model_used=self.model or "unknown",
                focus_areas=[f.value for f in focus_areas]
            )


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    DEFAULT_MODEL = "gpt-4-turbo"
    SUPPORTED_MODELS = ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"]

    def __init__(self, api_key: str, model: Optional[str] = None):
        super().__init__(api_key, model or self.DEFAULT_MODEL)

    def review(self, code: str, focus_areas: list, context: str = "", files: list = None) -> ReviewResult:
        """Perform code review using OpenAI."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        client = openai.OpenAI(api_key=self.api_key)
        prompt = self._build_prompt(code, focus_areas, context)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert code reviewer. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4096
            )
            response_text = response.choices[0].message.content
            return self._parse_response(response_text, files or [], focus_areas)

        except openai.AuthenticationError as e:
            raise APIError(f"OpenAI authentication failed: {e}")
        except openai.RateLimitError as e:
            raise APIError(f"OpenAI rate limit exceeded: {e}")
        except openai.APIError as e:
            raise APIError(f"OpenAI API error: {e}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    DEFAULT_MODEL = "claude-3-5-sonnet-latest"
    SUPPORTED_MODELS = ["claude-3-opus-latest", "claude-3-5-sonnet-latest", "claude-3-haiku-20240307", "claude-3-5-sonnet"]

    def __init__(self, api_key: str, model: Optional[str] = None):
        # Handle model alias
        if model == "claude-3-5-sonnet":
            model = "claude-3-5-sonnet-latest"
        super().__init__(api_key, model or self.DEFAULT_MODEL)

    def review(self, code: str, focus_areas: list, context: str = "", files: list = None) -> ReviewResult:
        """Perform code review using Anthropic Claude."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)
        prompt = self._build_prompt(code, focus_areas, context)

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = response.content[0].text
            return self._parse_response(response_text, files or [], focus_areas)

        except anthropic.AuthenticationError as e:
            raise APIError(f"Anthropic authentication failed: {e}")
        except anthropic.RateLimitError as e:
            raise APIError(f"Anthropic rate limit exceeded: {e}")
        except anthropic.APIError as e:
            raise APIError(f"Anthropic API error: {e}")


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""

    DEFAULT_MODEL = "gemini-1.5-pro"
    SUPPORTED_MODELS = ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]

    def __init__(self, api_key: str, model: Optional[str] = None):
        super().__init__(api_key, model or self.DEFAULT_MODEL)

    def review(self, code: str, focus_areas: list, context: str = "", files: list = None) -> ReviewResult:
        """Perform code review using Google Gemini."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        prompt = self._build_prompt(code, focus_areas, context)

        try:
            response = model.generate_content(prompt)
            response_text = response.text
            return self._parse_response(response_text, files or [], focus_areas)

        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "authentication" in error_msg:
                raise APIError(f"Google authentication failed: {e}")
            elif "rate" in error_msg or "quota" in error_msg:
                raise APIError(f"Google rate limit exceeded: {e}")
            else:
                raise APIError(f"Google API error: {e}")


class APIError(Exception):
    """Custom exception for API errors."""
    pass


class OutputFormatter:
    """Formats review results for output."""

    @staticmethod
    def format(result: ReviewResult, output_format: OutputFormat) -> str:
        """Format the review result based on the specified format."""
        if output_format == OutputFormat.JSON:
            return OutputFormatter._format_json(result)
        elif output_format == OutputFormat.MARKDOWN:
            return OutputFormatter._format_markdown(result)
        else:
            return OutputFormatter._format_text(result)

    @staticmethod
    def _format_json(result: ReviewResult) -> str:
        """Format as JSON."""
        return json.dumps(result.to_dict(), indent=2)

    @staticmethod
    def _format_markdown(result: ReviewResult) -> str:
        """Format as Markdown."""
        lines = ["# Code Review Report\n"]
        lines.append(f"**Model:** {result.model_used}\n")
        lines.append(f"**Focus Areas:** {', '.join(result.focus_areas)}\n")
        lines.append(f"**Files Reviewed:** {', '.join(result.files_reviewed) if result.files_reviewed else 'N/A'}\n")

        lines.append("\n## Summary\n")
        lines.append(f"{result.summary}\n")

        if result.issues:
            lines.append("\n## Issues Found\n")

            # Group by severity
            severity_order = ["critical", "high", "medium", "low", "info"]
            for severity in severity_order:
                severity_issues = [i for i in result.issues if i.severity == severity]
                if severity_issues:
                    lines.append(f"\n### {severity.upper()} ({len(severity_issues)})\n")
                    for i, issue in enumerate(severity_issues, 1):
                        lines.append(f"\n#### {i}. [{issue.category}] {issue.file}")
                        if issue.line:
                            lines.append(f" (line {issue.line})")
                        lines.append(f"\n\n{issue.message}\n")
                        if issue.suggestion:
                            lines.append(f"\n**Suggestion:** {issue.suggestion}\n")
                        if issue.code_snippet:
                            lines.append(f"\n```\n{issue.code_snippet}\n```\n")
        else:
            lines.append("\n## No Issues Found\n")
            lines.append("\nGreat job! The code looks good.\n")

        return "".join(lines)

    @staticmethod
    def _format_text(result: ReviewResult) -> str:
        """Format as plain text."""
        lines = ["=" * 60]
        lines.append("CODE REVIEW REPORT")
        lines.append("=" * 60)
        lines.append(f"Model: {result.model_used}")
        lines.append(f"Focus Areas: {', '.join(result.focus_areas)}")
        files_str = ', '.join(result.files_reviewed) if result.files_reviewed else 'N/A'
        lines.append(f"Files Reviewed: {files_str}")
        lines.append("-" * 60)
        lines.append("SUMMARY")
        lines.append("-" * 60)
        lines.append(result.summary)

        if result.issues:
            lines.append("\n" + "-" * 60)
            lines.append(f"ISSUES FOUND: {len(result.issues)}")
            lines.append("-" * 60)

            for i, issue in enumerate(result.issues, 1):
                lines.append(f"\n[{i}] {issue.severity.upper()} - {issue.category}")
                lines.append(f"    File: {issue.file}" + (f" (line {issue.line})" if issue.line else ""))
                lines.append(f"    {issue.message}")
                if issue.suggestion:
                    lines.append(f"    Suggestion: {issue.suggestion}")
                if issue.code_snippet:
                    lines.append(f"    Code: {issue.code_snippet}")
        else:
            lines.append("\n" + "-" * 60)
            lines.append("NO ISSUES FOUND")
            lines.append("-" * 60)
            lines.append("Great job! The code looks good.")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class CodeCollector:
    """Collects code from various sources."""

    @staticmethod
    def from_files(file_paths: list) -> tuple:
        """Collect code from file paths."""
        code_parts = []
        valid_files = []

        for path_str in file_paths:
            path = Path(path_str)
            if path.is_file():
                try:
                    content = path.read_text()
                    code_parts.append(f"# File: {path}\n{content}")
                    valid_files.append(str(path))
                except Exception as e:
                    print(f"Warning: Could not read {path}: {e}", file=sys.stderr)
            elif path.is_dir():
                # Handle directory
                for py_file in path.rglob("*.py"):
                    try:
                        content = py_file.read_text()
                        code_parts.append(f"# File: {py_file}\n{content}")
                        valid_files.append(str(py_file))
                    except Exception as e:
                        print(f"Warning: Could not read {py_file}: {e}", file=sys.stderr)
            else:
                print(f"Warning: {path} does not exist", file=sys.stderr)

        return "\n\n".join(code_parts), valid_files

    @staticmethod
    def from_directory(dir_path: str, extensions: list = None) -> tuple:
        """Collect code from a directory."""
        if extensions is None:
            extensions = [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h"]

        code_parts = []
        valid_files = []
        path = Path(dir_path)

        if not path.is_dir():
            raise ValueError(f"Directory does not exist: {dir_path}")

        for ext in extensions:
            for file_path in path.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text()
                    code_parts.append(f"# File: {file_path}\n{content}")
                    valid_files.append(str(file_path))
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)

        return "\n\n".join(code_parts), valid_files

    @staticmethod
    def from_git_diff(ref: str = "HEAD~1") -> tuple:
        """Collect code from git diff."""
        try:
            result = subprocess.run(
                ["git", "diff", ref],
                capture_output=True,
                text=True,
                check=True
            )
            diff = result.stdout
            # Extract file names from diff
            files = []
            for line in diff.split("\n"):
                if line.startswith("+++ b/"):
                    files.append(line[6:])
            return diff, files
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Git diff failed: {e.stderr}")
        except FileNotFoundError:
            raise ValueError("git command not found")

    @staticmethod
    def from_staged() -> tuple:
        """Collect code from staged git changes."""
        try:
            result = subprocess.run(
                ["git", "diff", "--cached"],
                capture_output=True,
                text=True,
                check=True
            )
            diff = result.stdout
            files = []
            for line in diff.split("\n"):
                if line.startswith("+++ b/"):
                    files.append(line[6:])
            return diff, files
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Git diff failed: {e.stderr}")
        except FileNotFoundError:
            raise ValueError("git command not found")

    @staticmethod
    def from_stdin() -> tuple:
        """Collect code from stdin."""
        if sys.stdin.isatty():
            raise ValueError("No input provided via stdin")
        code = sys.stdin.read()
        return code, ["stdin"]


def get_provider(model: Optional[str] = None) -> LLMProvider:
    """Get the appropriate LLM provider based on environment variables and model."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    # If a model is specified, try to match it to a provider
    if model:
        model_lower = model.lower()
        if any(m in model_lower for m in ["gpt", "openai"]):
            if not openai_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
            return OpenAIProvider(openai_key, model)
        elif any(m in model_lower for m in ["claude", "anthropic"]):
            if not anthropic_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic models")
            return AnthropicProvider(anthropic_key, model)
        elif any(m in model_lower for m in ["gemini", "google"]):
            if not google_key:
                raise ValueError("GOOGLE_API_KEY environment variable is required for Google models")
            return GoogleProvider(google_key, model)

    # Default provider selection based on available keys
    if anthropic_key:
        return AnthropicProvider(anthropic_key, model)
    elif openai_key:
        return OpenAIProvider(openai_key, model)
    elif google_key:
        return GoogleProvider(google_key, model)
    else:
        raise ValueError(
            "No API key found. Please set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY"
        )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with comprehensive help."""
    parser = argparse.ArgumentParser(
        prog="code_review",
        description="""
AI-Powered Code Review CLI Tool

Uses Large Language Models to perform intelligent code reviews with
configurable focus areas, output formats, and severity thresholds.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Review a single file for security issues:
    %(prog)s -f src/auth.py --focus security

  Review git diff with multiple focus areas:
    %(prog)s --diff HEAD~1 --focus security,performance

  Review multiple files with markdown output:
    %(prog)s src/*.py --output markdown > review.md

  Review staged changes, fail on high severity issues:
    %(prog)s --staged --fail-on high

  Review from stdin (pipe git diff):
    git diff main | %(prog)s --stdin --focus bugs

  Use a specific model:
    %(prog)s -f api.py --model claude-3-5-sonnet

  Review a directory recursively:
    %(prog)s -d ./src --focus style --output json

SUPPORTED MODELS:
  OpenAI:    gpt-4, gpt-4-turbo, gpt-4o, gpt-3.5-turbo
  Anthropic: claude-3-opus, claude-3-5-sonnet, claude-3-haiku
  Google:    gemini-pro, gemini-1.5-pro, gemini-1.5-flash

ENVIRONMENT VARIABLES:
  OPENAI_API_KEY      API key for OpenAI models
  ANTHROPIC_API_KEY   API key for Anthropic Claude models
  GOOGLE_API_KEY      API key for Google Gemini models

EXIT CODES:
  0  No issues found
  1  Issues found in code review
  2  API error (authentication, rate limit, etc.)
  3  Invalid arguments or input error

For more information, visit: https://github.com/example/code-review
        """
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_argument_group("Input Sources (choose one or more)")
    input_group.add_argument(
        "-f", "--file",
        action="append",
        dest="files",
        metavar="FILE",
        help="File(s) to review. Can be specified multiple times."
    )
    input_group.add_argument(
        "-d", "--directory",
        metavar="DIR",
        help="Directory to review recursively."
    )
    input_group.add_argument(
        "--diff",
        metavar="REF",
        nargs="?",
        const="HEAD~1",
        help="Review git diff against REF (default: HEAD~1)."
    )
    input_group.add_argument(
        "--staged",
        action="store_true",
        help="Review staged git changes."
    )
    input_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read code from stdin."
    )

    # Also accept positional arguments as files
    parser.add_argument(
        "positional_files",
        nargs="*",
        metavar="FILE",
        help="Additional files to review."
    )

    # Review options
    review_group = parser.add_argument_group("Review Options")
    review_group.add_argument(
        "--focus",
        metavar="AREAS",
        default="all",
        help="""
Focus areas for the review (comma-separated).
Options: security, performance, style, bugs, all
Default: all
        """
    )
    review_group.add_argument(
        "--model",
        metavar="MODEL",
        help="""
Specific model to use. If not specified, uses the default
model for the first available API key.
        """
    )
    review_group.add_argument(
        "--min-severity",
        metavar="LEVEL",
        default="info",
        choices=["info", "low", "medium", "high", "critical"],
        help="""
Minimum severity level to report.
Options: info, low, medium, high, critical
Default: info
        """
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o", "--output",
        metavar="FORMAT",
        choices=["text", "markdown", "json"],
        default="text",
        help="""
Output format. Options: text, markdown, json
Default: text
        """
    )
    output_group.add_argument(
        "--fail-on",
        metavar="LEVEL",
        choices=["info", "low", "medium", "high", "critical"],
        help="""
Exit with code 1 if issues at or above this severity are found.
Useful for CI/CD pipelines.
        """
    )

    # General options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output."
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-essential output."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    return parser


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Combine file arguments
    all_files = []
    if args.files:
        all_files.extend(args.files)
    if args.positional_files:
        all_files.extend(args.positional_files)

    # Validate input sources
    input_count = sum([
        bool(all_files),
        bool(args.directory),
        bool(args.diff),
        args.staged,
        args.stdin
    ])

    if input_count == 0:
        parser.error("No input source specified. Use -f, -d, --diff, --staged, --stdin, or provide files.")
        return ExitCode.INVALID_ARGS.value

    try:
        # Collect code from sources
        code_parts = []
        files_reviewed = []

        if all_files:
            code, files = CodeCollector.from_files(all_files)
            if code:
                code_parts.append(code)
                files_reviewed.extend(files)

        if args.directory:
            code, files = CodeCollector.from_directory(args.directory)
            if code:
                code_parts.append(code)
                files_reviewed.extend(files)

        if args.diff:
            code, files = CodeCollector.from_git_diff(args.diff)
            if code:
                code_parts.append(f"# Git Diff ({args.diff})\n{code}")
                files_reviewed.extend(files)

        if args.staged:
            code, files = CodeCollector.from_staged()
            if code:
                code_parts.append(f"# Staged Changes\n{code}")
                files_reviewed.extend(files)

        if args.stdin:
            code, files = CodeCollector.from_stdin()
            if code:
                code_parts.append(code)
                files_reviewed.extend(files)

        if not code_parts:
            print("Error: No code to review.", file=sys.stderr)
            return ExitCode.INVALID_ARGS.value

        combined_code = "\n\n".join(code_parts)

        # Parse focus areas
        focus_areas = []
        for area in args.focus.split(","):
            try:
                focus_areas.append(FocusArea.from_string(area.strip()))
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return ExitCode.INVALID_ARGS.value

        # Get provider and perform review
        if not args.quiet:
            print("Starting code review...", file=sys.stderr)

        provider = get_provider(args.model)

        if args.verbose:
            print(f"Using provider: {provider.__class__.__name__}", file=sys.stderr)
            print(f"Model: {provider.model}", file=sys.stderr)
            print(f"Focus areas: {[f.value for f in focus_areas]}", file=sys.stderr)
            print(f"Files: {files_reviewed}", file=sys.stderr)

        result = provider.review(
            combined_code,
            focus_areas,
            files=files_reviewed
        )

        # Filter by minimum severity
        min_severity = Severity.from_string(args.min_severity)
        result.issues = [
            i for i in result.issues
            if Severity.from_string(i.severity) >= min_severity
        ]

        # Format and output
        output_format = OutputFormat(args.output)
        output = OutputFormatter.format(result, output_format)
        print(output)

        # Determine exit code
        if args.fail_on:
            fail_severity = Severity.from_string(args.fail_on)
            if result.has_issues_at_or_above(fail_severity):
                return ExitCode.ISSUES_FOUND.value

        if result.issues:
            return ExitCode.ISSUES_FOUND.value

        return ExitCode.SUCCESS.value

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return ExitCode.INVALID_ARGS.value
    except APIError as e:
        print(f"API Error: {e}", file=sys.stderr)
        return ExitCode.API_ERROR.value
    except ImportError as e:
        print(f"Import Error: {e}", file=sys.stderr)
        return ExitCode.API_ERROR.value
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return ExitCode.INVALID_ARGS.value
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return ExitCode.API_ERROR.value


if __name__ == "__main__":
    sys.exit(main())
