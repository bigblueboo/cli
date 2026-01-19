#!/usr/bin/env python3
"""
doc_generator - A self-documenting CLI tool for generating documentation from code using LLMs.

This tool analyzes source code files and generates comprehensive documentation in various
formats (Markdown, RST, HTML) using Large Language Models from OpenAI, Anthropic, or Google.

Supported Languages:
    - Python (.py)
    - JavaScript (.js, .jsx)
    - TypeScript (.ts, .tsx)
    - Go (.go)
    - Rust (.rs)

Documentation Types:
    - README generation: Creates project README files with overview, installation, usage
    - API documentation: Generates API reference documentation for modules
    - Docstring generation: Adds/updates docstrings in source code
    - OpenAPI specs: Generates OpenAPI/Swagger documentation for API endpoints

Environment Variables:
    OPENAI_API_KEY: API key for OpenAI (GPT models)
    ANTHROPIC_API_KEY: API key for Anthropic (Claude models)
    GOOGLE_API_KEY: API key for Google (Gemini models)
    DOC_GENERATOR_MODEL: Override default model selection
    DOC_GENERATOR_PROVIDER: Force specific provider (openai, anthropic, google)

Exit Codes:
    0: Success
    1: File not found or path error
    2: API error (authentication, rate limit, network)
    3: Invalid arguments or configuration

Examples:
    # Generate markdown documentation for a single file
    $ doc_generator src/utils.py --format markdown > docs/utils.md

    # Generate documentation for entire directory
    $ doc_generator src/ --output docs/ --format markdown

    # Generate OpenAPI specification
    $ doc_generator api.py --type openapi > openapi.yaml

    # Generate project README
    $ doc_generator --readme --include src/ tests/ > README.md

    # Add docstrings to code
    $ doc_generator module.py --docstrings

    # Generate HTML API reference
    $ doc_generator src/ --type api-reference --format html > api.html

Author: Auto-generated CLI tool
Version: 1.0.0
"""

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional


# =============================================================================
# Exit Codes
# =============================================================================
class ExitCode(Enum):
    """Exit codes for the CLI tool."""
    SUCCESS = 0
    FILE_NOT_FOUND = 1
    API_ERROR = 2
    INVALID_ARGS = 3


# =============================================================================
# Output Formats
# =============================================================================
class OutputFormat(Enum):
    """Supported output formats for documentation."""
    MARKDOWN = "markdown"
    RST = "rst"
    HTML = "html"
    YAML = "yaml"  # For OpenAPI output


# =============================================================================
# Documentation Types
# =============================================================================
class DocType(Enum):
    """Types of documentation that can be generated."""
    MODULE = "module"  # Default: module-level documentation
    API_REFERENCE = "api-reference"  # Detailed API reference
    OPENAPI = "openapi"  # OpenAPI/Swagger spec
    README = "readme"  # Project README
    DOCSTRINGS = "docstrings"  # Inline docstring generation


# =============================================================================
# Supported Languages
# =============================================================================
LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class SourceFile:
    """Represents a source code file to be documented."""
    path: Path
    content: str
    language: str

    @classmethod
    def from_path(cls, path: Path) -> "SourceFile":
        """Create a SourceFile from a file path."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        language = LANGUAGE_EXTENSIONS.get(ext)
        if not language:
            raise ValueError(f"Unsupported file type: {ext}")

        content = path.read_text(encoding="utf-8")
        return cls(path=path, content=content, language=language)


@dataclass
class DocumentationRequest:
    """Request for documentation generation."""
    source_files: list[SourceFile]
    doc_type: DocType
    output_format: OutputFormat
    include_examples: bool = True
    include_types: bool = True
    verbose: bool = False


@dataclass
class DocumentationResult:
    """Result of documentation generation."""
    content: str
    format: OutputFormat
    source_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# LLM Provider Interface
# =============================================================================
class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text using the LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (API key set)."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text using OpenAI API."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text using Anthropic API."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt if system_prompt else "You are a documentation generator.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(self, model: str = "gemini-pro"):
        self.model = model
        self.api_key = os.environ.get("GOOGLE_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text using Google Gemini API."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

            model = genai.GenerativeModel(self.model)
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = model.generate_content(full_prompt)
            return response.text
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise RuntimeError(f"Google API error: {e}")


def get_provider(force_provider: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
    """Get the first available LLM provider."""
    provider_env = os.environ.get("DOC_GENERATOR_PROVIDER", force_provider)
    model_env = os.environ.get("DOC_GENERATOR_MODEL", model)

    providers = {
        "openai": lambda: OpenAIProvider(model=model_env or "gpt-4"),
        "anthropic": lambda: AnthropicProvider(model=model_env or "claude-sonnet-4-20250514"),
        "google": lambda: GoogleProvider(model=model_env or "gemini-pro"),
    }

    if provider_env:
        if provider_env not in providers:
            raise ValueError(f"Unknown provider: {provider_env}")
        provider = providers[provider_env]()
        if not provider.is_available():
            raise RuntimeError(f"Provider {provider_env} API key not set")
        return provider

    # Try providers in order
    for name, factory in providers.items():
        provider = factory()
        if provider.is_available():
            return provider

    raise RuntimeError(
        "No LLM API key found. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY"
    )


# =============================================================================
# Prompt Templates
# =============================================================================
SYSTEM_PROMPT = """You are a technical documentation expert. Generate clear, accurate, and
comprehensive documentation for source code. Follow best practices for the requested
documentation format. Be concise but thorough."""

MODULE_DOC_PROMPT = """Generate {format} documentation for the following {language} code.

Include:
- Module/file overview
- All public classes, functions, and constants
- Parameter descriptions and types
- Return value descriptions
- Usage examples where appropriate

Code:
```{language}
{code}
```

Generate documentation in {format} format:"""

API_REFERENCE_PROMPT = """Generate detailed API reference documentation in {format} format for the following {language} code.

Include:
- Complete function/method signatures
- Type annotations
- Parameter descriptions
- Return types and descriptions
- Exceptions/errors that may be raised
- Code examples for each public API

Code:
```{language}
{code}
```

Generate API reference in {format} format:"""

OPENAPI_PROMPT = """Analyze the following {language} code and generate an OpenAPI 3.0 specification in YAML format.

Extract:
- API endpoints (paths, methods)
- Request/response schemas
- Parameters (path, query, body)
- Response codes
- Authentication requirements if apparent

Code:
```{language}
{code}
```

Generate OpenAPI 3.0 specification in YAML:"""

README_PROMPT = """Generate a comprehensive README.md for a project with the following source files.

Include:
- Project title and description
- Features
- Installation instructions
- Quick start / usage examples
- API overview (if applicable)
- Contributing guidelines placeholder
- License placeholder

Source files:
{files}

Generate README in Markdown format:"""

DOCSTRING_PROMPT = """Add comprehensive docstrings to the following {language} code.

Requirements:
- Use the standard docstring format for {language}
- Document all public functions, classes, and methods
- Include parameter descriptions and types
- Include return value descriptions
- Preserve all existing code exactly
- Only add/update docstrings

Code:
```{language}
{code}
```

Return the complete code with docstrings added:"""


# =============================================================================
# Documentation Generator
# =============================================================================
class DocumentationGenerator:
    """Main documentation generator class."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, request: DocumentationRequest) -> DocumentationResult:
        """Generate documentation based on the request."""
        if request.doc_type == DocType.README:
            return self._generate_readme(request)
        elif request.doc_type == DocType.DOCSTRINGS:
            return self._generate_docstrings(request)
        elif request.doc_type == DocType.OPENAPI:
            return self._generate_openapi(request)
        elif request.doc_type == DocType.API_REFERENCE:
            return self._generate_api_reference(request)
        else:
            return self._generate_module_docs(request)

    def _generate_module_docs(self, request: DocumentationRequest) -> DocumentationResult:
        """Generate module-level documentation."""
        results = []
        warnings = []

        for source_file in request.source_files:
            prompt = MODULE_DOC_PROMPT.format(
                format=request.output_format.value,
                language=source_file.language,
                code=source_file.content,
            )
            try:
                doc = self.provider.generate(prompt, SYSTEM_PROMPT)
                results.append(f"# {source_file.path.name}\n\n{doc}")
            except Exception as e:
                warnings.append(f"Failed to process {source_file.path}: {e}")

        return DocumentationResult(
            content="\n\n---\n\n".join(results),
            format=OutputFormat.MARKDOWN,  # LLMs generate markdown, conversion happens in format_output
            source_files=[str(f.path) for f in request.source_files],
            warnings=warnings,
        )

    def _generate_api_reference(self, request: DocumentationRequest) -> DocumentationResult:
        """Generate API reference documentation."""
        results = []
        warnings = []

        for source_file in request.source_files:
            prompt = API_REFERENCE_PROMPT.format(
                format=request.output_format.value,
                language=source_file.language,
                code=source_file.content,
            )
            try:
                doc = self.provider.generate(prompt, SYSTEM_PROMPT)
                results.append(doc)
            except Exception as e:
                warnings.append(f"Failed to process {source_file.path}: {e}")

        return DocumentationResult(
            content="\n\n".join(results),
            format=OutputFormat.MARKDOWN,  # LLMs generate markdown, conversion happens in format_output
            source_files=[str(f.path) for f in request.source_files],
            warnings=warnings,
        )

    def _generate_openapi(self, request: DocumentationRequest) -> DocumentationResult:
        """Generate OpenAPI specification."""
        combined_code = "\n\n".join(
            f"# File: {f.path.name}\n{f.content}"
            for f in request.source_files
        )

        # Use the first file's language or default to the most common
        language = request.source_files[0].language if request.source_files else "python"

        prompt = OPENAPI_PROMPT.format(
            language=language,
            code=combined_code,
        )

        doc = self.provider.generate(prompt, SYSTEM_PROMPT)

        return DocumentationResult(
            content=doc,
            format=OutputFormat.YAML,
            source_files=[str(f.path) for f in request.source_files],
        )

    def _generate_readme(self, request: DocumentationRequest) -> DocumentationResult:
        """Generate project README."""
        files_summary = "\n\n".join(
            f"## {f.path.name} ({f.language})\n```{f.language}\n{f.content[:2000]}...\n```"
            if len(f.content) > 2000 else
            f"## {f.path.name} ({f.language})\n```{f.language}\n{f.content}\n```"
            for f in request.source_files
        )

        prompt = README_PROMPT.format(files=files_summary)
        doc = self.provider.generate(prompt, SYSTEM_PROMPT)

        return DocumentationResult(
            content=doc,
            format=OutputFormat.MARKDOWN,
            source_files=[str(f.path) for f in request.source_files],
        )

    def _generate_docstrings(self, request: DocumentationRequest) -> DocumentationResult:
        """Generate code with added docstrings."""
        results = []
        warnings = []

        for source_file in request.source_files:
            prompt = DOCSTRING_PROMPT.format(
                language=source_file.language,
                code=source_file.content,
            )
            try:
                doc = self.provider.generate(prompt, SYSTEM_PROMPT)
                # Clean up potential markdown code blocks in response
                if doc.startswith("```"):
                    lines = doc.split("\n")
                    doc = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                results.append(f"# {source_file.path}\n{doc}")
            except Exception as e:
                warnings.append(f"Failed to process {source_file.path}: {e}")

        return DocumentationResult(
            content="\n\n".join(results),
            format=OutputFormat.MARKDOWN,  # LLMs generate code with docstrings, treated as markdown
            source_files=[str(f.path) for f in request.source_files],
            warnings=warnings,
        )


# =============================================================================
# File Collection
# =============================================================================
def collect_source_files(paths: list[str], recursive: bool = True) -> list[SourceFile]:
    """Collect source files from paths (files or directories)."""
    source_files = []

    for path_str in paths:
        path = Path(path_str).resolve()

        if path.is_file():
            try:
                source_files.append(SourceFile.from_path(path))
            except (ValueError, FileNotFoundError) as e:
                print(f"Warning: Skipping {path}: {e}", file=sys.stderr)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            for ext in LANGUAGE_EXTENSIONS:
                for file_path in path.glob(f"{pattern}{ext}"):
                    try:
                        source_files.append(SourceFile.from_path(file_path))
                    except (ValueError, FileNotFoundError) as e:
                        print(f"Warning: Skipping {file_path}: {e}", file=sys.stderr)
        else:
            print(f"Warning: Path not found: {path}", file=sys.stderr)

    return source_files


# =============================================================================
# Output Formatting
# =============================================================================
def format_output(result: DocumentationResult, output_format: OutputFormat) -> str:
    """Format the documentation result for output."""
    content = result.content

    if output_format == OutputFormat.HTML and result.format != OutputFormat.HTML:
        # Convert markdown to basic HTML if needed
        content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Documentation</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        pre {{ background: #f4f4f4; padding: 16px; overflow-x: auto; border-radius: 4px; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        h1, h2, h3 {{ border-bottom: 1px solid #eee; padding-bottom: 8px; }}
    </style>
</head>
<body>
<pre>{content}</pre>
</body>
</html>"""

    return content


# =============================================================================
# CLI Argument Parser
# =============================================================================
def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="doc_generator",
        description="""
Generate documentation from source code using LLMs.

Supports Python, JavaScript, TypeScript, Go, and Rust.
Outputs Markdown, RST, or HTML documentation.

Environment Variables:
  OPENAI_API_KEY      API key for OpenAI (GPT models)
  ANTHROPIC_API_KEY   API key for Anthropic (Claude models)
  GOOGLE_API_KEY      API key for Google (Gemini models)
  DOC_GENERATOR_MODEL     Override default model
  DOC_GENERATOR_PROVIDER  Force provider (openai/anthropic/google)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0  Success
  1  File not found or path error
  2  API error (authentication, rate limit, network)
  3  Invalid arguments or configuration

Examples:
  # Generate markdown docs for a file
  %(prog)s src/utils.py --format markdown > docs/utils.md

  # Generate docs for a directory
  %(prog)s src/ --output docs/ --format markdown

  # Generate OpenAPI spec
  %(prog)s api.py --type openapi > openapi.yaml

  # Generate project README
  %(prog)s --readme --include src/ tests/ > README.md

  # Add docstrings to code
  %(prog)s module.py --docstrings

  # Generate HTML API reference
  %(prog)s src/ --type api-reference --format html > api.html

  # Use specific provider
  %(prog)s src/ --provider anthropic --model claude-sonnet-4-20250514

For more information, see: https://github.com/example/doc_generator
        """,
    )

    # Positional arguments
    parser.add_argument(
        "paths",
        nargs="*",
        help="Source files or directories to document",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-f", "--format",
        choices=["markdown", "rst", "html"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    output_group.add_argument(
        "-o", "--output",
        metavar="PATH",
        help="Output file or directory (default: stdout)",
    )
    output_group.add_argument(
        "--no-examples",
        action="store_true",
        help="Exclude usage examples from documentation",
    )

    # Documentation type options
    type_group = parser.add_argument_group("Documentation Type")
    type_mutex = type_group.add_mutually_exclusive_group()
    type_mutex.add_argument(
        "-t", "--type",
        choices=["module", "api-reference", "openapi"],
        default="module",
        help="Type of documentation to generate (default: module)",
    )
    type_mutex.add_argument(
        "--readme",
        action="store_true",
        help="Generate a project README file",
    )
    type_mutex.add_argument(
        "--docstrings",
        action="store_true",
        help="Generate code with added/updated docstrings",
    )

    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "-i", "--include",
        nargs="+",
        metavar="PATH",
        help="Additional paths to include (used with --readme)",
    )
    input_group.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search directories recursively",
    )

    # Provider options
    provider_group = parser.add_argument_group("LLM Provider Options")
    provider_group.add_argument(
        "--provider",
        choices=["openai", "anthropic", "google"],
        help="Force specific LLM provider",
    )
    provider_group.add_argument(
        "--model",
        metavar="MODEL",
        help="Override default model for provider",
    )

    # Other options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without calling the API",
    )

    return parser


# =============================================================================
# Main Entry Point
# =============================================================================
def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0=success, 1=file not found, 2=API error, 3=invalid args)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Validate arguments
    all_paths = list(args.paths) if args.paths else []
    if args.include:
        all_paths.extend(args.include)

    if not all_paths and not args.readme:
        parser.error("No input files specified. Provide paths or use --include with --readme")
        return ExitCode.INVALID_ARGS.value

    # Determine documentation type
    if args.readme:
        doc_type = DocType.README
    elif args.docstrings:
        doc_type = DocType.DOCSTRINGS
    else:
        doc_type = DocType(args.type)

    # Determine output format
    output_format = OutputFormat(args.format)
    if doc_type == DocType.OPENAPI:
        output_format = OutputFormat.YAML

    # Collect source files
    try:
        source_files = collect_source_files(all_paths, recursive=not args.no_recursive)
    except Exception as e:
        print(f"Error collecting files: {e}", file=sys.stderr)
        return ExitCode.FILE_NOT_FOUND.value

    if not source_files:
        print("Error: No supported source files found", file=sys.stderr)
        return ExitCode.FILE_NOT_FOUND.value

    if args.verbose:
        print(f"Found {len(source_files)} source files:", file=sys.stderr)
        for sf in source_files:
            print(f"  - {sf.path} ({sf.language})", file=sys.stderr)

    # Dry run mode
    if args.dry_run:
        print("Dry run mode - would process:", file=sys.stderr)
        for sf in source_files:
            print(f"  {sf.path}", file=sys.stderr)
        print(f"\nDocumentation type: {doc_type.value}", file=sys.stderr)
        print(f"Output format: {output_format.value}", file=sys.stderr)
        return ExitCode.SUCCESS.value

    # Get LLM provider
    try:
        provider = get_provider(args.provider, args.model)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return ExitCode.API_ERROR.value

    if args.verbose:
        print(f"Using provider: {provider.__class__.__name__}", file=sys.stderr)

    # Create documentation request
    request = DocumentationRequest(
        source_files=source_files,
        doc_type=doc_type,
        output_format=output_format,
        include_examples=not args.no_examples,
        verbose=args.verbose,
    )

    # Generate documentation
    try:
        generator = DocumentationGenerator(provider)
        result = generator.generate(request)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return ExitCode.API_ERROR.value
    except RuntimeError as e:
        print(f"API Error: {e}", file=sys.stderr)
        return ExitCode.API_ERROR.value
    except Exception as e:
        print(f"Error generating documentation: {e}", file=sys.stderr)
        return ExitCode.API_ERROR.value

    # Print warnings
    for warning in result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    # Format and output
    output_content = format_output(result, output_format)

    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir():
            # Generate individual files
            for source_file in source_files:
                ext = ".md" if output_format == OutputFormat.MARKDOWN else f".{output_format.value}"
                out_file = output_path / f"{source_file.path.stem}{ext}"
                out_file.write_text(output_content, encoding="utf-8")
                if args.verbose:
                    print(f"Wrote: {out_file}", file=sys.stderr)
        else:
            output_path.write_text(output_content, encoding="utf-8")
            if args.verbose:
                print(f"Wrote: {output_path}", file=sys.stderr)
    else:
        print(output_content)

    return ExitCode.SUCCESS.value


if __name__ == "__main__":
    sys.exit(main())
