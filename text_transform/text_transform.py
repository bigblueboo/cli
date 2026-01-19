#!/usr/bin/env python3
"""
text_transform - A versatile CLI tool for text and data transformation.

This tool provides format conversion, encoding/decoding, JSON querying,
and template rendering capabilities.

Exit Codes:
    0 - Success
    1 - Parse error (invalid input data)
    2 - Transform error (transformation failed)
    3 - Invalid arguments

Examples:
    # Format conversions
    text_transform json-to-yaml < config.json > config.yaml
    text_transform yaml-to-json < config.yaml
    text_transform json-to-toml < config.json
    text_transform toml-to-json < config.toml
    text_transform csv-to-json < data.csv --headers
    text_transform json-to-csv < data.json --headers

    # JQ-style queries (using JMESPath syntax)
    text_transform jq "data[].name" < response.json
    text_transform jq "users[?age > `18`].name" < users.json

    # Encoding/Decoding
    text_transform base64-encode < file.bin
    text_transform base64-decode < encoded.txt
    text_transform url-encode "hello world"
    text_transform url-decode "hello%20world"

    # Template rendering
    text_transform template template.j2 --vars '{"name": "World"}'
    text_transform template template.j2 --vars-file vars.json
"""

import argparse
import base64
import csv
import io
import json
import sys
import urllib.parse
from typing import Any, Dict, List, Optional, TextIO

try:
    import yaml
except ImportError:
    yaml = None

try:
    import toml
except ImportError:
    toml = None

try:
    import jmespath
except ImportError:
    jmespath = None

try:
    import jinja2
except ImportError:
    jinja2 = None

# Exit codes
EXIT_SUCCESS = 0
EXIT_PARSE_ERROR = 1
EXIT_TRANSFORM_ERROR = 2
EXIT_INVALID_ARGS = 3


def check_dependency(module: Any, name: str) -> None:
    """Check if a required module is available."""
    if module is None:
        print(f"Error: {name} is required for this operation. Install with: pip install {name}", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)


def read_input(input_file: Optional[str] = None) -> str:
    """Read input from file or stdin."""
    try:
        if input_file:
            with open(input_file, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return sys.stdin.read()
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        sys.exit(EXIT_PARSE_ERROR)
    except IOError as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(EXIT_PARSE_ERROR)


def write_output(data: str, output_file: Optional[str] = None) -> None:
    """Write output to file or stdout."""
    try:
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(data)
        else:
            print(data, end='')
    except IOError as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(EXIT_TRANSFORM_ERROR)


def parse_json(data: str) -> Any:
    """Parse JSON data."""
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(EXIT_PARSE_ERROR)


def parse_yaml(data: str) -> Any:
    """Parse YAML data."""
    check_dependency(yaml, 'pyyaml')
    try:
        return yaml.safe_load(data)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}", file=sys.stderr)
        sys.exit(EXIT_PARSE_ERROR)


def parse_toml(data: str) -> Any:
    """Parse TOML data."""
    check_dependency(toml, 'toml')
    try:
        return toml.loads(data)
    except toml.TomlDecodeError as e:
        print(f"Error parsing TOML: {e}", file=sys.stderr)
        sys.exit(EXIT_PARSE_ERROR)


def parse_csv(data: str, has_headers: bool = True) -> List[Dict[str, Any]]:
    """Parse CSV data into a list of dictionaries or lists."""
    try:
        reader = csv.reader(io.StringIO(data))
        rows = list(reader)

        if not rows:
            return []

        if has_headers:
            headers = rows[0]
            return [dict(zip(headers, row)) for row in rows[1:]]
        else:
            return [{"row": row} for row in rows]
    except csv.Error as e:
        print(f"Error parsing CSV: {e}", file=sys.stderr)
        sys.exit(EXIT_PARSE_ERROR)


def to_json(data: Any, indent: int = 2) -> str:
    """Convert data to JSON."""
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        print(f"Error converting to JSON: {e}", file=sys.stderr)
        sys.exit(EXIT_TRANSFORM_ERROR)


def to_yaml(data: Any) -> str:
    """Convert data to YAML."""
    check_dependency(yaml, 'pyyaml')
    try:
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except yaml.YAMLError as e:
        print(f"Error converting to YAML: {e}", file=sys.stderr)
        sys.exit(EXIT_TRANSFORM_ERROR)


def to_toml(data: Any) -> str:
    """Convert data to TOML."""
    check_dependency(toml, 'toml')
    try:
        return toml.dumps(data)
    except (TypeError, ValueError) as e:
        print(f"Error converting to TOML: {e}", file=sys.stderr)
        sys.exit(EXIT_TRANSFORM_ERROR)


def to_csv(data: Any, include_headers: bool = True) -> str:
    """Convert data to CSV."""
    try:
        if not isinstance(data, list):
            data = [data]

        if not data:
            return ""

        output = io.StringIO()

        if isinstance(data[0], dict):
            headers = list(data[0].keys())
            writer = csv.DictWriter(output, fieldnames=headers)
            if include_headers:
                writer.writeheader()
            writer.writerows(data)
        else:
            writer = csv.writer(output)
            for row in data:
                if isinstance(row, (list, tuple)):
                    writer.writerow(row)
                else:
                    writer.writerow([row])

        return output.getvalue()
    except (TypeError, ValueError) as e:
        print(f"Error converting to CSV: {e}", file=sys.stderr)
        sys.exit(EXIT_TRANSFORM_ERROR)


# Command handlers

def cmd_json_to_yaml(args: argparse.Namespace) -> None:
    """Convert JSON to YAML."""
    data = parse_json(read_input(args.input))
    write_output(to_yaml(data), args.output)


def cmd_yaml_to_json(args: argparse.Namespace) -> None:
    """Convert YAML to JSON."""
    data = parse_yaml(read_input(args.input))
    write_output(to_json(data) + '\n', args.output)


def cmd_json_to_toml(args: argparse.Namespace) -> None:
    """Convert JSON to TOML."""
    data = parse_json(read_input(args.input))
    write_output(to_toml(data), args.output)


def cmd_toml_to_json(args: argparse.Namespace) -> None:
    """Convert TOML to JSON."""
    data = parse_toml(read_input(args.input))
    write_output(to_json(data) + '\n', args.output)


def cmd_yaml_to_toml(args: argparse.Namespace) -> None:
    """Convert YAML to TOML."""
    data = parse_yaml(read_input(args.input))
    write_output(to_toml(data), args.output)


def cmd_toml_to_yaml(args: argparse.Namespace) -> None:
    """Convert TOML to YAML."""
    data = parse_toml(read_input(args.input))
    write_output(to_yaml(data), args.output)


def cmd_csv_to_json(args: argparse.Namespace) -> None:
    """Convert CSV to JSON."""
    data = parse_csv(read_input(args.input), args.headers)
    write_output(to_json(data) + '\n', args.output)


def cmd_json_to_csv(args: argparse.Namespace) -> None:
    """Convert JSON to CSV."""
    data = parse_json(read_input(args.input))
    write_output(to_csv(data, args.headers), args.output)


def cmd_jq(args: argparse.Namespace) -> None:
    """Query JSON using JMESPath."""
    check_dependency(jmespath, 'jmespath')

    data = parse_json(read_input(args.input))

    try:
        result = jmespath.search(args.query, data)
        if result is None:
            write_output('null\n', args.output)
        elif isinstance(result, (dict, list)):
            write_output(to_json(result) + '\n', args.output)
        elif isinstance(result, bool):
            write_output(str(result).lower() + '\n', args.output)
        else:
            write_output(str(result) + '\n', args.output)
    except jmespath.exceptions.JMESPathError as e:
        print(f"Error in JMESPath query: {e}", file=sys.stderr)
        sys.exit(EXIT_TRANSFORM_ERROR)


def cmd_base64_encode(args: argparse.Namespace) -> None:
    """Encode input as base64."""
    try:
        if args.input:
            with open(args.input, 'rb') as f:
                data = f.read()
        else:
            data = sys.stdin.buffer.read()

        encoded = base64.b64encode(data).decode('ascii')
        write_output(encoded + '\n', args.output)
    except IOError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(EXIT_PARSE_ERROR)


def cmd_base64_decode(args: argparse.Namespace) -> None:
    """Decode base64 input."""
    try:
        data = read_input(args.input).strip()
        decoded = base64.b64decode(data)

        if args.output:
            with open(args.output, 'wb') as f:
                f.write(decoded)
        else:
            sys.stdout.buffer.write(decoded)
    except base64.binascii.Error as e:
        print(f"Error decoding base64: {e}", file=sys.stderr)
        sys.exit(EXIT_PARSE_ERROR)
    except IOError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(EXIT_TRANSFORM_ERROR)


def cmd_url_encode(args: argparse.Namespace) -> None:
    """URL encode input."""
    if args.text:
        data = args.text
    else:
        data = read_input(args.input).rstrip('\n')

    encoded = urllib.parse.quote(data, safe='')
    write_output(encoded + '\n', args.output)


def cmd_url_decode(args: argparse.Namespace) -> None:
    """URL decode input."""
    if args.text:
        data = args.text
    else:
        data = read_input(args.input).rstrip('\n')

    try:
        decoded = urllib.parse.unquote(data)
        write_output(decoded + '\n', args.output)
    except Exception as e:
        print(f"Error decoding URL: {e}", file=sys.stderr)
        sys.exit(EXIT_PARSE_ERROR)


def cmd_template(args: argparse.Namespace) -> None:
    """Render a Jinja2 template."""
    check_dependency(jinja2, 'jinja2')

    # Load variables
    variables: Dict[str, Any] = {}

    if args.vars:
        try:
            variables = json.loads(args.vars)
        except json.JSONDecodeError as e:
            print(f"Error parsing --vars JSON: {e}", file=sys.stderr)
            sys.exit(EXIT_INVALID_ARGS)

    if args.vars_file:
        try:
            with open(args.vars_file, 'r', encoding='utf-8') as f:
                file_vars = json.load(f)
                variables.update(file_vars)
        except FileNotFoundError:
            print(f"Error: Variables file not found: {args.vars_file}", file=sys.stderr)
            sys.exit(EXIT_INVALID_ARGS)
        except json.JSONDecodeError as e:
            print(f"Error parsing variables file: {e}", file=sys.stderr)
            sys.exit(EXIT_PARSE_ERROR)

    # Load template
    try:
        with open(args.template, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except FileNotFoundError:
        print(f"Error: Template file not found: {args.template}", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)
    except IOError as e:
        print(f"Error reading template: {e}", file=sys.stderr)
        sys.exit(EXIT_PARSE_ERROR)

    # Render template
    try:
        env = jinja2.Environment(
            undefined=jinja2.StrictUndefined,
            autoescape=False
        )
        template = env.from_string(template_content)
        result = template.render(**variables)
        write_output(result, args.output)
    except jinja2.TemplateError as e:
        print(f"Error rendering template: {e}", file=sys.stderr)
        sys.exit(EXIT_TRANSFORM_ERROR)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""

    # Main parser
    parser = argparse.ArgumentParser(
        prog='text_transform',
        description='A versatile CLI tool for text and data transformation.',
        epilog='''
Exit Codes:
  0  Success
  1  Parse error (invalid input data)
  2  Transform error (transformation failed)
  3  Invalid arguments

Examples:
  %(prog)s json-to-yaml < config.json > config.yaml
  %(prog)s yaml-to-json < config.yaml
  %(prog)s csv-to-json < data.csv --headers
  %(prog)s jq "data[].name" < response.json
  %(prog)s base64-encode < file.bin
  %(prog)s url-encode "hello world"
  %(prog)s template template.j2 --vars '{"name": "World"}'

For more help on a specific command:
  %(prog)s <command> --help
''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Common arguments for input/output
    def add_io_args(p: argparse.ArgumentParser) -> None:
        p.add_argument('-i', '--input', metavar='FILE', help='Input file (default: stdin)')
        p.add_argument('-o', '--output', metavar='FILE', help='Output file (default: stdout)')

    # json-to-yaml
    p = subparsers.add_parser('json-to-yaml', help='Convert JSON to YAML',
        description='Convert JSON input to YAML format.',
        epilog='Example: %(prog)s < config.json > config.yaml')
    add_io_args(p)
    p.set_defaults(func=cmd_json_to_yaml)

    # yaml-to-json
    p = subparsers.add_parser('yaml-to-json', help='Convert YAML to JSON',
        description='Convert YAML input to JSON format.',
        epilog='Example: %(prog)s < config.yaml > config.json')
    add_io_args(p)
    p.set_defaults(func=cmd_yaml_to_json)

    # json-to-toml
    p = subparsers.add_parser('json-to-toml', help='Convert JSON to TOML',
        description='Convert JSON input to TOML format.',
        epilog='Example: %(prog)s < config.json > config.toml')
    add_io_args(p)
    p.set_defaults(func=cmd_json_to_toml)

    # toml-to-json
    p = subparsers.add_parser('toml-to-json', help='Convert TOML to JSON',
        description='Convert TOML input to JSON format.',
        epilog='Example: %(prog)s < config.toml > config.json')
    add_io_args(p)
    p.set_defaults(func=cmd_toml_to_json)

    # yaml-to-toml
    p = subparsers.add_parser('yaml-to-toml', help='Convert YAML to TOML',
        description='Convert YAML input to TOML format.',
        epilog='Example: %(prog)s < config.yaml > config.toml')
    add_io_args(p)
    p.set_defaults(func=cmd_yaml_to_toml)

    # toml-to-yaml
    p = subparsers.add_parser('toml-to-yaml', help='Convert TOML to YAML',
        description='Convert TOML input to YAML format.',
        epilog='Example: %(prog)s < config.toml > config.yaml')
    add_io_args(p)
    p.set_defaults(func=cmd_toml_to_yaml)

    # csv-to-json
    p = subparsers.add_parser('csv-to-json', help='Convert CSV to JSON',
        description='Convert CSV input to JSON format.',
        epilog='''
Examples:
  %(prog)s --headers < data.csv
  %(prog)s --no-headers < raw.csv
''')
    add_io_args(p)
    p.add_argument('--headers', dest='headers', action='store_true', default=True,
        help='First row contains headers (default)')
    p.add_argument('--no-headers', dest='headers', action='store_false',
        help='First row is data, not headers')
    p.set_defaults(func=cmd_csv_to_json)

    # json-to-csv
    p = subparsers.add_parser('json-to-csv', help='Convert JSON to CSV',
        description='Convert JSON array input to CSV format.',
        epilog='Example: %(prog)s < data.json > data.csv')
    add_io_args(p)
    p.add_argument('--headers', dest='headers', action='store_true', default=True,
        help='Include header row (default)')
    p.add_argument('--no-headers', dest='headers', action='store_false',
        help='Omit header row')
    p.set_defaults(func=cmd_json_to_csv)

    # jq (JMESPath)
    p = subparsers.add_parser('jq', help='Query JSON using JMESPath syntax',
        description='''
Query JSON data using JMESPath expressions.
See https://jmespath.org/ for syntax documentation.
''',
        epilog='''
Examples:
  %(prog)s "data[].name" < response.json
  %(prog)s "users[?age > `18`].name" < users.json
  %(prog)s "keys(@)" < object.json
  %(prog)s "length(items)" < data.json
''',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('query', help='JMESPath query expression')
    add_io_args(p)
    p.set_defaults(func=cmd_jq)

    # base64-encode
    p = subparsers.add_parser('base64-encode', help='Encode input as base64',
        description='Encode binary or text input as base64.',
        epilog='Example: %(prog)s < file.bin > encoded.txt')
    add_io_args(p)
    p.set_defaults(func=cmd_base64_encode)

    # base64-decode
    p = subparsers.add_parser('base64-decode', help='Decode base64 input',
        description='Decode base64 encoded input.',
        epilog='Example: %(prog)s < encoded.txt > file.bin')
    add_io_args(p)
    p.set_defaults(func=cmd_base64_decode)

    # url-encode
    p = subparsers.add_parser('url-encode', help='URL encode input',
        description='URL encode (percent-encode) input text.',
        epilog='''
Examples:
  %(prog)s "hello world"
  echo "hello world" | %(prog)s
''')
    p.add_argument('text', nargs='?', help='Text to encode (reads from stdin if omitted)')
    add_io_args(p)
    p.set_defaults(func=cmd_url_encode)

    # url-decode
    p = subparsers.add_parser('url-decode', help='URL decode input',
        description='Decode URL encoded (percent-encoded) input.',
        epilog='''
Examples:
  %(prog)s "hello%%20world"
  echo "hello%%20world" | %(prog)s
''')
    p.add_argument('text', nargs='?', help='Text to decode (reads from stdin if omitted)')
    add_io_args(p)
    p.set_defaults(func=cmd_url_decode)

    # template
    p = subparsers.add_parser('template', help='Render a Jinja2 template',
        description='''
Render a Jinja2 template with provided variables.
Variables can be provided as JSON via --vars or from a JSON file via --vars-file.
''',
        epilog='''
Examples:
  %(prog)s template.j2 --vars '{"name": "World"}'
  %(prog)s template.j2 --vars-file vars.json
  %(prog)s template.j2 --vars '{"a": 1}' --vars-file more.json
''',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('template', help='Template file path')
    p.add_argument('--vars', metavar='JSON', help='Variables as JSON string')
    p.add_argument('--vars-file', metavar='FILE', help='Variables from JSON file')
    p.add_argument('-o', '--output', metavar='FILE', help='Output file (default: stdout)')
    p.set_defaults(func=cmd_template)

    return parser


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(EXIT_INVALID_ARGS)

    try:
        args.func(args)
        sys.exit(EXIT_SUCCESS)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(EXIT_TRANSFORM_ERROR)


if __name__ == '__main__':
    main()
