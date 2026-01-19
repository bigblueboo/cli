#!/usr/bin/env python3
"""
data_extractor - Extract structured data from unstructured text using LLMs

A self-documenting CLI tool that leverages Large Language Models to extract
structured data from various input sources (files, stdin, URLs) and validates
the output against JSON Schema.

Exit Codes:
    0 - Success
    1 - File not found
    2 - API error
    3 - Validation error
    4 - Invalid arguments

Environment Variables:
    OPENAI_API_KEY    - OpenAI API key (uses gpt-4o model)
    ANTHROPIC_API_KEY - Anthropic API key (uses claude-3-5-sonnet)
    GOOGLE_API_KEY    - Google API key (uses gemini-pro)

Examples:
    # Extract invoice data using a JSON schema file
    data_extractor invoice.pdf --schema invoice-schema.json

    # Quick extraction with field names
    data_extractor email.txt --extract "sender,date,subject,action_items"

    # Output as JSON with specific fields
    data_extractor report.txt --to-json --fields "metrics,recommendations,risks"

    # Batch process multiple files
    data_extractor contracts/*.pdf --schema contract.json --output results/

    # Read from stdin
    cat article.html | data_extractor --extract "title,author,publish_date,summary"

    # Inline JSON schema
    data_extractor resume.pdf --schema '{"name": "string", "skills": ["string"], "experience_years": "number"}'

    # Extract from URL
    data_extractor https://example.com/article.html --extract "title,summary"

Author: Data Extractor CLI
Version: 1.0.0
"""

import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

# Exit codes
EXIT_SUCCESS = 0
EXIT_FILE_NOT_FOUND = 1
EXIT_API_ERROR = 2
EXIT_VALIDATION_ERROR = 3
EXIT_INVALID_ARGS = 4

# Version
__version__ = "1.0.0"


def get_api_key() -> tuple[str, str]:
    """
    Get the API key from environment variables.

    Returns:
        tuple: (provider_name, api_key)

    Raises:
        SystemExit: If no API key is found
    """
    providers = [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("google", "GOOGLE_API_KEY"),
    ]

    for provider, env_var in providers:
        key = os.environ.get(env_var)
        if key:
            return provider, key

    print(
        "Error: No API key found. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY",
        file=sys.stderr
    )
    sys.exit(EXIT_INVALID_ARGS)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def extract(self, text: str, schema: Dict[str, Any], fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract structured data from text."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for data extraction."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
            sys.exit(EXIT_INVALID_ARGS)

    def extract(self, text: str, schema: Dict[str, Any], fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract structured data using OpenAI."""
        prompt = self._build_prompt(text, schema, fields)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Extract structured data from the provided text and return valid JSON only. No explanations."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"OpenAI API Error: {e}", file=sys.stderr)
            sys.exit(EXIT_API_ERROR)

    def _build_prompt(self, text: str, schema: Dict[str, Any], fields: Optional[List[str]]) -> str:
        """Build the extraction prompt."""
        if fields:
            schema_desc = f"Extract these fields: {', '.join(fields)}"
        else:
            schema_desc = f"Extract data according to this JSON schema:\n{json.dumps(schema, indent=2)}"

        return f"""{schema_desc}

Text to extract from:
---
{text}
---

Return the extracted data as valid JSON."""


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider for data extraction."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("Error: anthropic package not installed. Run: pip install anthropic", file=sys.stderr)
            sys.exit(EXIT_INVALID_ARGS)

    def extract(self, text: str, schema: Dict[str, Any], fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract structured data using Anthropic Claude."""
        prompt = self._build_prompt(text, schema, fields)

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are a data extraction assistant. Extract structured data from the provided text and return valid JSON only. No explanations or markdown formatting."
            )
            content = response.content[0].text
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            print(f"Error parsing Anthropic response as JSON: {e}", file=sys.stderr)
            sys.exit(EXIT_API_ERROR)
        except Exception as e:
            print(f"Anthropic API Error: {e}", file=sys.stderr)
            sys.exit(EXIT_API_ERROR)

    def _build_prompt(self, text: str, schema: Dict[str, Any], fields: Optional[List[str]]) -> str:
        """Build the extraction prompt."""
        if fields:
            schema_desc = f"Extract these fields: {', '.join(fields)}"
        else:
            schema_desc = f"Extract data according to this JSON schema:\n{json.dumps(schema, indent=2)}"

        return f"""{schema_desc}

Text to extract from:
---
{text}
---

Return only the extracted data as valid JSON, no other text."""


class GoogleProvider(LLMProvider):
    """Google Gemini provider for data extraction."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        except ImportError:
            print("Error: google-generativeai package not installed. Run: pip install google-generativeai", file=sys.stderr)
            sys.exit(EXIT_INVALID_ARGS)

    def extract(self, text: str, schema: Dict[str, Any], fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract structured data using Google Gemini."""
        prompt = self._build_prompt(text, schema, fields)

        try:
            response = self.model.generate_content(prompt)
            content = response.text
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            print(f"Error parsing Google response as JSON: {e}", file=sys.stderr)
            sys.exit(EXIT_API_ERROR)
        except Exception as e:
            print(f"Google API Error: {e}", file=sys.stderr)
            sys.exit(EXIT_API_ERROR)

    def _build_prompt(self, text: str, schema: Dict[str, Any], fields: Optional[List[str]]) -> str:
        """Build the extraction prompt."""
        if fields:
            schema_desc = f"Extract these fields: {', '.join(fields)}"
        else:
            schema_desc = f"Extract data according to this JSON schema:\n{json.dumps(schema, indent=2)}"

        return f"""You are a data extraction assistant. Extract structured data from the provided text and return valid JSON only.

{schema_desc}

Text to extract from:
---
{text}
---

Return only the extracted data as valid JSON, no other text or explanation."""


def get_provider(provider_name: str, api_key: str) -> LLMProvider:
    """Get the appropriate LLM provider."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
    }
    return providers[provider_name](api_key)


def read_file(file_path: str) -> str:
    """
    Read content from a file.

    Args:
        file_path: Path to the file to read

    Returns:
        str: File contents as text

    Raises:
        SystemExit: If file not found or cannot be read
    """
    path = Path(file_path)

    if not path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(EXIT_FILE_NOT_FOUND)

    # Handle different file types
    suffix = path.suffix.lower()

    if suffix == '.pdf':
        return read_pdf(path)
    elif suffix in ['.html', '.htm']:
        return read_html(path)
    else:
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                return path.read_text(encoding='latin-1')
            except Exception as e:
                print(f"Error reading file {file_path}: {e}", file=sys.stderr)
                sys.exit(EXIT_FILE_NOT_FOUND)


def read_pdf(path: Path) -> str:
    """
    Read text content from a PDF file.

    Args:
        path: Path to the PDF file

    Returns:
        str: Extracted text from PDF
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        return "\n".join(text)
    except ImportError:
        print("Warning: pypdf not installed. Install with: pip install pypdf", file=sys.stderr)
        print("Attempting to read PDF as text...", file=sys.stderr)
        return path.read_bytes().decode('utf-8', errors='ignore')


def read_html(path: Path) -> str:
    """
    Read and extract text from HTML file.

    Args:
        path: Path to the HTML file

    Returns:
        str: Extracted text from HTML
    """
    try:
        from bs4 import BeautifulSoup
        html_content = path.read_text(encoding='utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator='\n', strip=True)
    except ImportError:
        # Fallback: simple regex-based HTML tag removal
        html_content = path.read_text(encoding='utf-8')
        clean = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
        clean = re.sub(r'<style[^>]*>.*?</style>', '', clean, flags=re.DOTALL)
        clean = re.sub(r'<[^>]+>', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()


def read_url(url: str) -> str:
    """
    Fetch and read content from a URL.

    Args:
        url: URL to fetch

    Returns:
        str: Content from URL

    Raises:
        SystemExit: If URL cannot be fetched
    """
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'data_extractor/1.0'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode('utf-8', errors='ignore')

            # If HTML, extract text
            if 'text/html' in response.headers.get('Content-Type', ''):
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(content, 'html.parser')
                    for script in soup(["script", "style"]):
                        script.decompose()
                    return soup.get_text(separator='\n', strip=True)
                except ImportError:
                    clean = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
                    clean = re.sub(r'<style[^>]*>.*?</style>', '', clean, flags=re.DOTALL)
                    clean = re.sub(r'<[^>]+>', ' ', clean)
                    return re.sub(r'\s+', ' ', clean).strip()

            return content
    except urllib.error.URLError as e:
        print(f"Error fetching URL {url}: {e}", file=sys.stderr)
        sys.exit(EXIT_FILE_NOT_FOUND)
    except Exception as e:
        print(f"Error processing URL {url}: {e}", file=sys.stderr)
        sys.exit(EXIT_FILE_NOT_FOUND)


def parse_schema(schema_arg: str) -> Dict[str, Any]:
    """
    Parse schema from file path, JSON string, or simple format.

    Args:
        schema_arg: Schema as file path, JSON string, or simple format

    Returns:
        dict: Parsed JSON schema

    Raises:
        SystemExit: If schema cannot be parsed
    """
    # Check if it's a file path
    if os.path.isfile(schema_arg):
        try:
            with open(schema_arg, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing schema file: {e}", file=sys.stderr)
            sys.exit(EXIT_INVALID_ARGS)
        except FileNotFoundError:
            print(f"Schema file not found: {schema_arg}", file=sys.stderr)
            sys.exit(EXIT_FILE_NOT_FOUND)

    # Try to parse as JSON
    try:
        return json.loads(schema_arg)
    except json.JSONDecodeError:
        # Not valid JSON, try simple format
        print(f"Error: Invalid schema format: {schema_arg}", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)


def fields_to_schema(fields: str) -> Dict[str, Any]:
    """
    Convert comma-separated field names to a simple schema.

    Args:
        fields: Comma-separated field names

    Returns:
        dict: Simple schema with field names as keys
    """
    field_list = [f.strip() for f in fields.split(',')]
    return {
        "type": "object",
        "properties": {field: {"type": "string"} for field in field_list},
        "required": field_list
    }


def validate_output(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate extracted data against JSON schema.

    Args:
        data: Extracted data to validate
        schema: JSON schema to validate against

    Returns:
        bool: True if valid

    Raises:
        SystemExit: If validation fails
    """
    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=schema)
        return True
    except ImportError:
        # jsonschema not installed, skip validation
        print("Warning: jsonschema not installed, skipping validation", file=sys.stderr)
        return True
    except jsonschema.ValidationError as e:
        print(f"Validation Error: {e.message}", file=sys.stderr)
        sys.exit(EXIT_VALIDATION_ERROR)
    except jsonschema.SchemaError as e:
        print(f"Schema Error: {e.message}", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)


def process_input(
    source: str,
    schema: Dict[str, Any],
    fields: Optional[List[str]],
    provider: LLMProvider,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Process a single input source and extract data.

    Args:
        source: Input source (file path, URL, or '-' for stdin)
        schema: JSON schema for output
        fields: Optional list of fields to extract
        provider: LLM provider to use
        validate: Whether to validate output

    Returns:
        dict: Extracted data
    """
    # Read content
    if source == '-' or source is None:
        text = sys.stdin.read()
    elif source.startswith(('http://', 'https://')):
        text = read_url(source)
    else:
        text = read_file(source)

    if not text.strip():
        print(f"Warning: Empty input from {source or 'stdin'}", file=sys.stderr)
        return {}

    # Extract data
    result = provider.extract(text, schema, fields)

    # Validate if requested and schema provided
    if validate and schema and schema.get('type'):
        validate_output(result, schema)

    return result


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser
    """
    parser = argparse.ArgumentParser(
        prog='data_extractor',
        description='''
Extract structured data from unstructured text using Large Language Models.

This tool reads text from files, URLs, or stdin and uses AI to extract
structured data according to a specified schema or field list.
        ''',
        epilog='''
EXAMPLES:
  Extract invoice data using a JSON schema file:
    %(prog)s invoice.pdf --schema invoice-schema.json

  Quick extraction with field names:
    %(prog)s email.txt --extract "sender,date,subject,action_items"

  Output as JSON with specific fields:
    %(prog)s report.txt --to-json --fields "metrics,recommendations,risks"

  Batch process multiple files:
    %(prog)s contracts/*.pdf --schema contract.json --output results/

  Read from stdin:
    cat article.html | %(prog)s --extract "title,author,publish_date,summary"

  Inline JSON schema:
    %(prog)s resume.pdf --schema '{"type": "object", "properties": {"name": {"type": "string"}, "skills": {"type": "array", "items": {"type": "string"}}}}'

  Extract from URL:
    %(prog)s https://example.com/article.html --extract "title,summary"

EXIT CODES:
  0  Success
  1  File not found
  2  API error
  3  Validation error
  4  Invalid arguments

ENVIRONMENT VARIABLES:
  OPENAI_API_KEY     OpenAI API key (uses gpt-4o model)
  ANTHROPIC_API_KEY  Anthropic API key (uses claude-3-5-sonnet)
  GOOGLE_API_KEY     Google API key (uses gemini-pro)

  At least one API key must be set. The tool uses the first available key
  in the order listed above.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'inputs',
        nargs='*',
        metavar='INPUT',
        help='Input files, URLs, or "-" for stdin. Supports glob patterns.'
    )

    schema_group = parser.add_argument_group('Schema Options')
    schema_group.add_argument(
        '--schema', '-s',
        metavar='SCHEMA',
        help='JSON schema file path or inline JSON schema string'
    )
    schema_group.add_argument(
        '--extract', '-e',
        metavar='FIELDS',
        help='Comma-separated field names to extract (simple mode)'
    )
    schema_group.add_argument(
        '--fields', '-f',
        metavar='FIELDS',
        help='Alias for --extract: comma-separated field names'
    )

    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output', '-o',
        metavar='PATH',
        help='Output directory for batch processing, or file path for single input'
    )
    output_group.add_argument(
        '--to-json',
        action='store_true',
        help='Output as formatted JSON (default)'
    )
    output_group.add_argument(
        '--compact',
        action='store_true',
        help='Output as compact JSON (no formatting)'
    )
    output_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )

    validation_group = parser.add_argument_group('Validation Options')
    validation_group.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip schema validation of output'
    )
    validation_group.add_argument(
        '--strict',
        action='store_true',
        help='Fail on any validation warning'
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    parser.add_argument(
        '--provider', '-p',
        choices=['openai', 'anthropic', 'google'],
        help='Force specific LLM provider (default: auto-detect from env vars)'
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        int: Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Determine schema
    schema = {}
    fields = None

    if args.schema:
        schema = parse_schema(args.schema)
    elif args.extract:
        fields = [f.strip() for f in args.extract.split(',')]
        schema = fields_to_schema(args.extract)
    elif args.fields:
        fields = [f.strip() for f in args.fields.split(',')]
        schema = fields_to_schema(args.fields)
    else:
        # No schema specified
        if not args.inputs and sys.stdin.isatty():
            parser.print_help()
            return EXIT_SUCCESS
        schema = {"type": "object"}  # Generic object schema

    # Get API key and provider
    if args.provider:
        provider_name = args.provider
        env_var = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': 'GOOGLE_API_KEY'
        }[provider_name]
        api_key = os.environ.get(env_var)
        if not api_key:
            print(f"Error: {env_var} not set", file=sys.stderr)
            return EXIT_INVALID_ARGS
    else:
        provider_name, api_key = get_api_key()

    provider = get_provider(provider_name, api_key)

    # Determine inputs
    inputs = args.inputs if args.inputs else ['-']

    # Process each input
    results = []
    for i, source in enumerate(inputs):
        if not args.quiet and len(inputs) > 1:
            print(f"Processing {source}...", file=sys.stderr)

        result = process_input(
            source,
            schema,
            fields,
            provider,
            validate=not args.no_validate
        )

        if len(inputs) > 1:
            results.append({"source": source, "data": result})
        else:
            results = result

    # Output results
    if args.output:
        output_path = Path(args.output)

        if len(inputs) > 1:
            # Batch mode: output to directory
            output_path.mkdir(parents=True, exist_ok=True)
            for item in results:
                source_name = Path(item["source"]).stem if item["source"] != '-' else 'stdin'
                out_file = output_path / f"{source_name}.json"
                with open(out_file, 'w') as f:
                    json.dump(item["data"], f, indent=2 if not args.compact else None)
                if not args.quiet:
                    print(f"Written: {out_file}", file=sys.stderr)
        else:
            # Single file: output to file
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2 if not args.compact else None)
            if not args.quiet:
                print(f"Written: {output_path}", file=sys.stderr)
    else:
        # Output to stdout
        indent = None if args.compact else 2
        print(json.dumps(results, indent=indent))

    return EXIT_SUCCESS


if __name__ == '__main__':
    sys.exit(main())
