#!/usr/bin/env python3
"""
llm_query - A unified CLI tool for querying multiple LLM providers.

Supports OpenAI, Anthropic, and Google GenAI models with streaming output,
JSON schema validation, and comprehensive error handling.

Exit Codes:
    0 - Success
    1 - Missing API key
    2 - API error
    3 - Validation error (JSON schema)
    4 - Invalid arguments

Environment Variables:
    OPENAI_API_KEY    - API key for OpenAI models (gpt-4o, etc.)
    ANTHROPIC_API_KEY - API key for Anthropic models (claude-3-5-sonnet, etc.)
    GOOGLE_API_KEY    - API key for Google GenAI models (gemini-2.5-flash, etc.)

Examples:
    # Basic query with OpenAI
    llm_query -m gpt-4o -p "Summarize this" < document.txt

    # Query with file input
    llm_query -m claude-3-5-sonnet -p "Explain this code" -f code.py

    # Read from stdin explicitly
    llm_query -m gemini-2.5-flash -p "Translate to Spanish" --stdin

    # With JSON schema validation
    llm_query -m gpt-4o -p "Extract entities" --json-schema schema.json < text.txt

    # With system prompt
    llm_query -m claude-3-5-sonnet -p "Help me" --system "You are a coding assistant"

    # With system prompt from file
    llm_query -m gpt-4o -p "Review this" --system-file system_prompt.txt -f code.py
"""

import argparse
import json
import os
import sys
from typing import Any, Generator, Optional

# Exit codes
EXIT_SUCCESS = 0
EXIT_MISSING_API_KEY = 1
EXIT_API_ERROR = 2
EXIT_VALIDATION_ERROR = 3
EXIT_INVALID_ARGS = 4

# Provider detection based on model name prefixes/patterns
OPENAI_MODELS = ('gpt-', 'o1-', 'o3-', 'chatgpt-', 'text-')
ANTHROPIC_MODELS = ('claude-',)
GOOGLE_MODELS = ('gemini-',)


def detect_provider(model: str) -> str:
    """Detect the LLM provider based on model name.

    Args:
        model: The model identifier string.

    Returns:
        Provider name: 'openai', 'anthropic', or 'google'.

    Raises:
        ValueError: If the model cannot be matched to a known provider.
    """
    model_lower = model.lower()

    if any(model_lower.startswith(prefix) for prefix in OPENAI_MODELS):
        return 'openai'
    elif any(model_lower.startswith(prefix) for prefix in ANTHROPIC_MODELS):
        return 'anthropic'
    elif any(model_lower.startswith(prefix) for prefix in GOOGLE_MODELS):
        return 'google'
    else:
        raise ValueError(
            f"Unknown model '{model}'. Supported prefixes: "
            f"OpenAI ({', '.join(OPENAI_MODELS)}), "
            f"Anthropic ({', '.join(ANTHROPIC_MODELS)}), "
            f"Google ({', '.join(GOOGLE_MODELS)})"
        )


def get_api_key(provider: str) -> str:
    """Get API key for the specified provider from environment.

    Args:
        provider: The provider name ('openai', 'anthropic', or 'google').

    Returns:
        The API key string.

    Raises:
        SystemExit: If the required API key is not set.
    """
    env_vars = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'google': 'GOOGLE_API_KEY',
    }

    env_var = env_vars.get(provider)
    if not env_var:
        print(f"Error: Unknown provider '{provider}'", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)

    api_key = os.environ.get(env_var)
    if not api_key:
        print(f"Error: {env_var} environment variable is not set.", file=sys.stderr)
        print(f"Please set it with: export {env_var}=your_api_key", file=sys.stderr)
        sys.exit(EXIT_MISSING_API_KEY)

    return api_key


def load_json_schema(schema_path: str) -> dict:
    """Load and parse a JSON schema file.

    Args:
        schema_path: Path to the JSON schema file.

    Returns:
        The parsed JSON schema as a dictionary.

    Raises:
        SystemExit: If the file cannot be read or parsed.
    """
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Schema file not found: {schema_path}", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in schema file: {e}", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)


def validate_json_output(output: str, schema: dict) -> bool:
    """Validate that output conforms to a JSON schema.

    Args:
        output: The string output from the LLM.
        schema: The JSON schema to validate against.

    Returns:
        True if validation passes.

    Raises:
        SystemExit: If validation fails.
    """
    try:
        # Try to parse the output as JSON
        parsed = json.loads(output)
    except json.JSONDecodeError as e:
        print(f"\nError: Output is not valid JSON: {e}", file=sys.stderr)
        sys.exit(EXIT_VALIDATION_ERROR)

    # Basic schema validation (type checking)
    # For full JSON Schema validation, consider using jsonschema library
    schema_type = schema.get('type')
    if schema_type:
        type_map = {
            'object': dict,
            'array': list,
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'null': type(None),
        }
        expected_type = type_map.get(schema_type)
        if expected_type and not isinstance(parsed, expected_type):
            print(
                f"\nError: Output type mismatch. Expected {schema_type}, "
                f"got {type(parsed).__name__}",
                file=sys.stderr
            )
            sys.exit(EXIT_VALIDATION_ERROR)

    # Check required properties for objects
    if schema_type == 'object' and 'required' in schema:
        if isinstance(parsed, dict):
            missing = [key for key in schema['required'] if key not in parsed]
            if missing:
                print(
                    f"\nError: Missing required properties: {', '.join(missing)}",
                    file=sys.stderr
                )
                sys.exit(EXIT_VALIDATION_ERROR)

    return True


def stream_openai(
    model: str,
    prompt: str,
    content: Optional[str],
    system_prompt: Optional[str],
    json_schema: Optional[dict],
    api_key: str,
) -> Generator[str, None, None]:
    """Stream response from OpenAI API.

    Args:
        model: The model identifier (e.g., 'gpt-4o').
        prompt: The user prompt.
        content: Optional additional content to include.
        system_prompt: Optional system prompt.
        json_schema: Optional JSON schema for structured output.
        api_key: The OpenAI API key.

    Yields:
        Chunks of the response text.

    Raises:
        SystemExit: On API errors.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)

    client = OpenAI(api_key=api_key)

    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = prompt
    if content:
        user_content = f"{prompt}\n\n{content}"

    messages.append({"role": "user", "content": user_content})

    try:
        # Build request parameters
        params = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        # Add JSON schema if provided (structured outputs)
        if json_schema:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": json_schema,
                    "strict": True,
                }
            }

        stream = client.chat.completions.create(**params)

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        print(f"\nError from OpenAI API: {e}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)


def stream_anthropic(
    model: str,
    prompt: str,
    content: Optional[str],
    system_prompt: Optional[str],
    json_schema: Optional[dict],
    api_key: str,
) -> Generator[str, None, None]:
    """Stream response from Anthropic API.

    Args:
        model: The model identifier (e.g., 'claude-3-5-sonnet-20241022').
        prompt: The user prompt.
        content: Optional additional content to include.
        system_prompt: Optional system prompt.
        json_schema: Optional JSON schema for structured output.
        api_key: The Anthropic API key.

    Yields:
        Chunks of the response text.

    Raises:
        SystemExit: On API errors.
    """
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)

    client = anthropic.Anthropic(api_key=api_key)

    # Build user message content
    user_content = prompt
    if content:
        user_content = f"{prompt}\n\n{content}"

    messages = [{"role": "user", "content": user_content}]

    try:
        # Build request parameters
        params = {
            "model": model,
            "max_tokens": 4096,
            "messages": messages,
        }

        if system_prompt:
            # If JSON schema requested, add instructions to system prompt
            if json_schema:
                schema_instruction = (
                    f"\n\nYou must respond with valid JSON that conforms to this schema:\n"
                    f"{json.dumps(json_schema, indent=2)}"
                )
                params["system"] = system_prompt + schema_instruction
            else:
                params["system"] = system_prompt
        elif json_schema:
            params["system"] = (
                f"You must respond with valid JSON that conforms to this schema:\n"
                f"{json.dumps(json_schema, indent=2)}"
            )

        # Use high-level streaming with context manager
        with client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text

    except anthropic.APIError as e:
        print(f"\nError from Anthropic API: {e}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)
    except Exception as e:
        print(f"\nError from Anthropic API: {e}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)


def stream_google(
    model: str,
    prompt: str,
    content: Optional[str],
    system_prompt: Optional[str],
    json_schema: Optional[dict],
    api_key: str,
) -> Generator[str, None, None]:
    """Stream response from Google GenAI API.

    Args:
        model: The model identifier (e.g., 'gemini-2.5-flash').
        prompt: The user prompt.
        content: Optional additional content to include.
        system_prompt: Optional system prompt.
        json_schema: Optional JSON schema for structured output.
        api_key: The Google API key.

    Yields:
        Chunks of the response text.

    Raises:
        SystemExit: On API errors.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: google-genai package not installed. Run: pip install google-genai", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)

    client = genai.Client(api_key=api_key)

    # Build the prompt with content
    full_prompt = prompt
    if content:
        full_prompt = f"{prompt}\n\n{content}"

    # Build contents with optional system instruction
    contents = []
    if system_prompt:
        # Google GenAI uses system_instruction in config, not as a message
        pass

    contents.append(full_prompt)

    try:
        # Build config
        config = {}

        if system_prompt:
            config["system_instruction"] = system_prompt

        # Add JSON schema if provided (structured outputs)
        if json_schema:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = json_schema

        # Use streaming
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config if config else None,
        ):
            if chunk.text:
                yield chunk.text

    except Exception as e:
        print(f"\nError from Google GenAI API: {e}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)


def query_llm(
    model: str,
    prompt: str,
    content: Optional[str] = None,
    system_prompt: Optional[str] = None,
    json_schema: Optional[dict] = None,
) -> str:
    """Query an LLM and stream the response to stdout.

    Args:
        model: The model identifier.
        prompt: The user prompt.
        content: Optional additional content.
        system_prompt: Optional system prompt.
        json_schema: Optional JSON schema for validation.

    Returns:
        The complete response text.

    Raises:
        SystemExit: On any errors.
    """
    # Detect provider and get API key
    try:
        provider = detect_provider(model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)

    api_key = get_api_key(provider)

    # Select the appropriate streaming function
    stream_funcs = {
        'openai': stream_openai,
        'anthropic': stream_anthropic,
        'google': stream_google,
    }

    stream_func = stream_funcs[provider]

    # Stream and collect response
    full_response = []
    for chunk in stream_func(model, prompt, content, system_prompt, json_schema, api_key):
        print(chunk, end='', flush=True)
        full_response.append(chunk)

    # Print newline at end
    print()

    return ''.join(full_response)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog='llm_query',
        description='Query multiple LLM providers with a unified interface.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query with OpenAI
  %(prog)s -m gpt-4o -p "Summarize this" < document.txt

  # Query with file input
  %(prog)s -m claude-3-5-sonnet -p "Explain this code" -f code.py

  # Read from stdin explicitly
  %(prog)s -m gemini-2.5-flash -p "Translate to Spanish" --stdin

  # With JSON schema validation
  %(prog)s -m gpt-4o -p "Extract entities" --json-schema schema.json < text.txt

  # With system prompt
  %(prog)s -m claude-3-5-sonnet -p "Help me" --system "You are a coding assistant"

  # With system prompt from file
  %(prog)s -m gpt-4o -p "Review this" --system-file system_prompt.txt -f code.py

  # Query with temperature control (using provider-native temperature)
  %(prog)s -m gpt-4o -p "Be creative" --temperature 0.9

Supported Models:
  OpenAI:    gpt-4o, gpt-4o-mini, gpt-4-turbo, o1-preview, o1-mini, etc.
  Anthropic: claude-3-5-sonnet-20241022, claude-3-opus-20240229, claude-3-haiku-20240307, etc.
  Google:    gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash, etc.

Environment Variables:
  OPENAI_API_KEY     API key for OpenAI models
  ANTHROPIC_API_KEY  API key for Anthropic models
  GOOGLE_API_KEY     API key for Google GenAI models

Exit Codes:
  0  Success
  1  Missing API key
  2  API error
  3  Validation error (JSON schema)
  4  Invalid arguments
""",
    )

    # Required arguments
    parser.add_argument(
        '-m', '--model',
        required=True,
        help='Model to use (e.g., gpt-4o, claude-3-5-sonnet, gemini-2.5-flash)',
    )

    parser.add_argument(
        '-p', '--prompt',
        required=True,
        help='The prompt to send to the LLM',
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()

    input_group.add_argument(
        '-f', '--file',
        help='Read input content from a file',
    )

    input_group.add_argument(
        '--stdin',
        action='store_true',
        help='Read input content from stdin',
    )

    # System prompt options (mutually exclusive)
    system_group = parser.add_mutually_exclusive_group()

    system_group.add_argument(
        '--system', '-s',
        dest='system_prompt',
        help='System prompt to set context for the LLM',
    )

    system_group.add_argument(
        '--system-file',
        dest='system_file',
        help='Read system prompt from a file',
    )

    # JSON schema validation
    parser.add_argument(
        '--json-schema',
        dest='json_schema',
        help='Path to JSON schema file for output validation',
    )

    # Additional options
    parser.add_argument(
        '--temperature',
        type=float,
        help='Temperature for response generation (0.0-2.0)',
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        dest='max_tokens',
        help='Maximum tokens in response',
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (show provider, model, etc.)',
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0',
    )

    return parser


def main(args: Optional[list] = None) -> int:
    """Main entry point for the CLI.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code.
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    # Get content from file or stdin
    content = None

    if parsed.file:
        try:
            with open(parsed.file, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {parsed.file}", file=sys.stderr)
            return EXIT_INVALID_ARGS
        except IOError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            return EXIT_INVALID_ARGS
    elif parsed.stdin:
        if sys.stdin.isatty():
            print("Error: --stdin specified but no input provided via pipe", file=sys.stderr)
            return EXIT_INVALID_ARGS
        content = sys.stdin.read()
    elif not sys.stdin.isatty():
        # Auto-detect piped input
        content = sys.stdin.read()

    # Get system prompt
    system_prompt = None

    if parsed.system_prompt:
        system_prompt = parsed.system_prompt
    elif parsed.system_file:
        try:
            with open(parsed.system_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        except FileNotFoundError:
            print(f"Error: System prompt file not found: {parsed.system_file}", file=sys.stderr)
            return EXIT_INVALID_ARGS
        except IOError as e:
            print(f"Error reading system prompt file: {e}", file=sys.stderr)
            return EXIT_INVALID_ARGS

    # Load JSON schema if provided
    json_schema = None
    if parsed.json_schema:
        json_schema = load_json_schema(parsed.json_schema)

    # Verbose output
    if parsed.verbose:
        try:
            provider = detect_provider(parsed.model)
            print(f"Provider: {provider}", file=sys.stderr)
            print(f"Model: {parsed.model}", file=sys.stderr)
            if system_prompt:
                print(f"System prompt: {system_prompt[:50]}...", file=sys.stderr)
            if json_schema:
                print(f"JSON schema: {parsed.json_schema}", file=sys.stderr)
            print("---", file=sys.stderr)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return EXIT_INVALID_ARGS

    # Query the LLM
    response = query_llm(
        model=parsed.model,
        prompt=parsed.prompt,
        content=content,
        system_prompt=system_prompt,
        json_schema=json_schema,
    )

    # Validate output against schema if provided
    if json_schema:
        validate_json_output(response, json_schema)

    return EXIT_SUCCESS


if __name__ == '__main__':
    sys.exit(main())
