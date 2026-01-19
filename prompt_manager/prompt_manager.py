#!/usr/bin/env python3
"""
Prompt Manager - A CLI tool for storing and managing reusable prompt templates.

This tool provides CRUD operations for prompt templates with support for:
- Variable substitution using Jinja2-style syntax ({{ variable }})
- Categories and tags for organization
- Import/export collections as YAML or JSON
- Configurable storage location via PROMPT_MANAGER_DIR environment variable

Exit Codes:
    0 - Success
    1 - Not found (prompt, category, or tag doesn't exist)
    2 - File error (read/write/permission issues)
    3 - Invalid arguments (bad syntax, missing required args)

Storage:
    Prompts are stored in ~/.prompt_manager/ by default.
    Override with PROMPT_MANAGER_DIR environment variable.

Examples:
    # Add a new prompt template
    prompt_manager add "code-review" --template "Review this {{language}} code:\\n{{code}}"

    # Add with category and tags
    prompt_manager add "summarize" --template "Summarize: {{text}}" --category writing --tags "short,utility"

    # List all prompts
    prompt_manager list

    # List prompts by category
    prompt_manager list --category coding

    # List prompts by tag
    prompt_manager list --tag utility

    # Get a specific prompt
    prompt_manager get "code-review"

    # Render a prompt with variables
    prompt_manager render "code-review" --vars '{"language": "Python", "code": "def foo():", "focus": "bugs"}'

    # Update a prompt
    prompt_manager update "code-review" --template "New template: {{code}}"

    # Export all prompts to YAML
    prompt_manager export --format yaml > prompts.yaml

    # Export to JSON
    prompt_manager export --format json > prompts.json

    # Import prompts from file
    prompt_manager import prompts.yaml

    # Delete a prompt
    prompt_manager delete "code-review"

Author: Prompt Manager CLI
License: MIT
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from jinja2 import Template, TemplateError, UndefinedError, StrictUndefined
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


# Exit codes
EXIT_SUCCESS = 0
EXIT_NOT_FOUND = 1
EXIT_FILE_ERROR = 2
EXIT_INVALID_ARGS = 3


def get_storage_dir() -> Path:
    """
    Get the storage directory for prompt templates.

    Returns the path specified by PROMPT_MANAGER_DIR environment variable,
    or ~/.prompt_manager/ if not set.

    Returns:
        Path: The storage directory path.
    """
    env_dir = os.environ.get('PROMPT_MANAGER_DIR')
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / '.prompt_manager'


def get_prompts_file() -> Path:
    """
    Get the path to the prompts storage file.

    Returns:
        Path: The path to prompts.json file.
    """
    return get_storage_dir() / 'prompts.json'


def ensure_storage_exists() -> bool:
    """
    Ensure the storage directory and file exist.

    Creates the storage directory and an empty prompts file if they don't exist.

    Returns:
        bool: True if successful, False on error.
    """
    try:
        storage_dir = get_storage_dir()
        storage_dir.mkdir(parents=True, exist_ok=True)

        prompts_file = get_prompts_file()
        if not prompts_file.exists():
            prompts_file.write_text('{}')
        return True
    except (OSError, PermissionError) as e:
        print(f"Error: Could not create storage directory: {e}", file=sys.stderr)
        return False


def load_prompts() -> Optional[Dict[str, Any]]:
    """
    Load all prompts from the storage file.

    Returns:
        Optional[Dict[str, Any]]: Dictionary of prompts, or None on error.
        Each prompt has: template, category, tags, created_at, updated_at
    """
    if not ensure_storage_exists():
        return None

    try:
        prompts_file = get_prompts_file()
        content = prompts_file.read_text()
        return json.loads(content) if content.strip() else {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in prompts file: {e}", file=sys.stderr)
        return None
    except (OSError, PermissionError) as e:
        print(f"Error: Could not read prompts file: {e}", file=sys.stderr)
        return None


def save_prompts(prompts: Dict[str, Any]) -> bool:
    """
    Save prompts to the storage file.

    Args:
        prompts: Dictionary of prompts to save.

    Returns:
        bool: True if successful, False on error.
    """
    if not ensure_storage_exists():
        return False

    try:
        prompts_file = get_prompts_file()
        prompts_file.write_text(json.dumps(prompts, indent=2, sort_keys=True))
        return True
    except (OSError, PermissionError) as e:
        print(f"Error: Could not write prompts file: {e}", file=sys.stderr)
        return False


def cmd_add(args: argparse.Namespace) -> int:
    """
    Add a new prompt template.

    Creates a new prompt with the given name, template, optional category, and tags.
    Fails if a prompt with the same name already exists.

    Args:
        args: Parsed command line arguments with name, template, category, tags.

    Returns:
        int: Exit code (0 for success, 2 for file error, 3 for already exists).
    """
    prompts = load_prompts()
    if prompts is None:
        return EXIT_FILE_ERROR

    if args.name in prompts:
        print(f"Error: Prompt '{args.name}' already exists. Use 'update' to modify.", file=sys.stderr)
        return EXIT_INVALID_ARGS

    tags = []
    if args.tags:
        tags = [t.strip() for t in args.tags.split(',') if t.strip()]

    now = datetime.now(timezone.utc).isoformat()
    prompts[args.name] = {
        'template': args.template,
        'category': args.category or '',
        'tags': tags,
        'created_at': now,
        'updated_at': now
    }

    if not save_prompts(prompts):
        return EXIT_FILE_ERROR

    print(f"Added prompt '{args.name}'")
    return EXIT_SUCCESS


def cmd_list(args: argparse.Namespace) -> int:
    """
    List all prompts, optionally filtered by category or tag.

    Displays prompt names with their categories and tags.
    Can filter by --category or --tag options.

    Args:
        args: Parsed command line arguments with optional category and tag filters.

    Returns:
        int: Exit code (0 for success, 1 if no matches, 2 for file error).
    """
    prompts = load_prompts()
    if prompts is None:
        return EXIT_FILE_ERROR

    if not prompts:
        print("No prompts found.")
        return EXIT_SUCCESS

    # Filter by category if specified
    filtered = prompts
    if args.category:
        filtered = {k: v for k, v in filtered.items() if v.get('category') == args.category}

    # Filter by tag if specified
    if args.tag:
        filtered = {k: v for k, v in filtered.items() if args.tag in v.get('tags', [])}

    if not filtered:
        if args.category and args.tag:
            print(f"No prompts found with category '{args.category}' and tag '{args.tag}'.")
        elif args.category:
            print(f"No prompts found with category '{args.category}'.")
        elif args.tag:
            print(f"No prompts found with tag '{args.tag}'.")
        return EXIT_NOT_FOUND

    # Display prompts
    for name, data in sorted(filtered.items()):
        category = data.get('category', '')
        tags = data.get('tags', [])

        line = f"  {name}"
        if category:
            line += f"  [category: {category}]"
        if tags:
            line += f"  [tags: {', '.join(tags)}]"
        print(line)

    print(f"\nTotal: {len(filtered)} prompt(s)")
    return EXIT_SUCCESS


def cmd_get(args: argparse.Namespace) -> int:
    """
    Get and display a specific prompt template.

    Shows the full details of a prompt including template, category, tags,
    and timestamps.

    Args:
        args: Parsed command line arguments with name of prompt to get.

    Returns:
        int: Exit code (0 for success, 1 if not found, 2 for file error).
    """
    prompts = load_prompts()
    if prompts is None:
        return EXIT_FILE_ERROR

    if args.name not in prompts:
        print(f"Error: Prompt '{args.name}' not found.", file=sys.stderr)
        return EXIT_NOT_FOUND

    prompt = prompts[args.name]

    print(f"Name: {args.name}")
    print(f"Category: {prompt.get('category', '') or '(none)'}")
    print(f"Tags: {', '.join(prompt.get('tags', [])) or '(none)'}")
    print(f"Created: {prompt.get('created_at', 'unknown')}")
    print(f"Updated: {prompt.get('updated_at', 'unknown')}")
    print(f"\nTemplate:\n{prompt.get('template', '')}")

    return EXIT_SUCCESS


def cmd_update(args: argparse.Namespace) -> int:
    """
    Update an existing prompt template.

    Updates the specified fields of an existing prompt.
    At least one of --template, --category, or --tags must be provided.

    Args:
        args: Parsed command line arguments with name and fields to update.

    Returns:
        int: Exit code (0 for success, 1 if not found, 2 for file error, 3 for invalid args).
    """
    prompts = load_prompts()
    if prompts is None:
        return EXIT_FILE_ERROR

    if args.name not in prompts:
        print(f"Error: Prompt '{args.name}' not found.", file=sys.stderr)
        return EXIT_NOT_FOUND

    if not any([args.template, args.category is not None, args.tags is not None]):
        print("Error: At least one of --template, --category, or --tags must be provided.", file=sys.stderr)
        return EXIT_INVALID_ARGS

    prompt = prompts[args.name]

    if args.template:
        prompt['template'] = args.template
    if args.category is not None:
        prompt['category'] = args.category
    if args.tags is not None:
        prompt['tags'] = [t.strip() for t in args.tags.split(',') if t.strip()] if args.tags else []

    prompt['updated_at'] = datetime.now(timezone.utc).isoformat()

    if not save_prompts(prompts):
        return EXIT_FILE_ERROR

    print(f"Updated prompt '{args.name}'")
    return EXIT_SUCCESS


def cmd_delete(args: argparse.Namespace) -> int:
    """
    Delete a prompt template.

    Permanently removes a prompt from storage.

    Args:
        args: Parsed command line arguments with name of prompt to delete.

    Returns:
        int: Exit code (0 for success, 1 if not found, 2 for file error).
    """
    prompts = load_prompts()
    if prompts is None:
        return EXIT_FILE_ERROR

    if args.name not in prompts:
        print(f"Error: Prompt '{args.name}' not found.", file=sys.stderr)
        return EXIT_NOT_FOUND

    del prompts[args.name]

    if not save_prompts(prompts):
        return EXIT_FILE_ERROR

    print(f"Deleted prompt '{args.name}'")
    return EXIT_SUCCESS


def cmd_render(args: argparse.Namespace) -> int:
    """
    Render a prompt template with variable substitution.

    Uses Jinja2 templating to substitute variables in the prompt template.
    Variables are provided as a JSON object via --vars.

    Args:
        args: Parsed command line arguments with name and vars.

    Returns:
        int: Exit code (0 for success, 1 if not found, 2 for file error, 3 for invalid args).

    Example:
        prompt_manager render "code-review" --vars '{"language": "Python", "code": "def foo():"}'
    """
    if not JINJA2_AVAILABLE:
        print("Error: jinja2 is required for rendering. Install with: pip install jinja2", file=sys.stderr)
        return EXIT_INVALID_ARGS

    prompts = load_prompts()
    if prompts is None:
        return EXIT_FILE_ERROR

    if args.name not in prompts:
        print(f"Error: Prompt '{args.name}' not found.", file=sys.stderr)
        return EXIT_NOT_FOUND

    template_str = prompts[args.name].get('template', '')

    # Parse variables from JSON
    variables = {}
    if args.vars:
        try:
            variables = json.loads(args.vars)
            if not isinstance(variables, dict):
                print("Error: --vars must be a JSON object.", file=sys.stderr)
                return EXIT_INVALID_ARGS
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in --vars: {e}", file=sys.stderr)
            return EXIT_INVALID_ARGS

    # Render the template
    try:
        template = Template(template_str, undefined=StrictUndefined)
        rendered = template.render(**variables)
        print(rendered)
        return EXIT_SUCCESS
    except UndefinedError as e:
        print(f"Error: Missing variable in template: {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS
    except TemplateError as e:
        print(f"Error: Template error: {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS


def cmd_export(args: argparse.Namespace) -> int:
    """
    Export all prompts to YAML or JSON format.

    Outputs all prompts to stdout in the specified format.
    Redirect to a file to save: prompt_manager export --format yaml > prompts.yaml

    Args:
        args: Parsed command line arguments with format (yaml or json).

    Returns:
        int: Exit code (0 for success, 2 for file error, 3 for invalid args).
    """
    prompts = load_prompts()
    if prompts is None:
        return EXIT_FILE_ERROR

    fmt = args.format.lower()

    if fmt == 'yaml':
        if not YAML_AVAILABLE:
            print("Error: pyyaml is required for YAML export. Install with: pip install pyyaml", file=sys.stderr)
            return EXIT_INVALID_ARGS
        print(yaml.dump(prompts, default_flow_style=False, sort_keys=True, allow_unicode=True))
    elif fmt == 'json':
        print(json.dumps(prompts, indent=2, sort_keys=True))
    else:
        print(f"Error: Unknown format '{fmt}'. Use 'yaml' or 'json'.", file=sys.stderr)
        return EXIT_INVALID_ARGS

    return EXIT_SUCCESS


def cmd_import(args: argparse.Namespace) -> int:
    """
    Import prompts from a YAML or JSON file.

    Reads prompts from a file and merges them with existing prompts.
    Existing prompts with the same name will be overwritten unless --no-overwrite is set.

    Args:
        args: Parsed command line arguments with file path and options.

    Returns:
        int: Exit code (0 for success, 2 for file error, 3 for invalid args).
    """
    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: File '{args.file}' not found.", file=sys.stderr)
        return EXIT_FILE_ERROR

    try:
        content = file_path.read_text()
    except (OSError, PermissionError) as e:
        print(f"Error: Could not read file: {e}", file=sys.stderr)
        return EXIT_FILE_ERROR

    # Try to parse as YAML first (which also handles JSON), then JSON
    imported = None
    try:
        if YAML_AVAILABLE:
            imported = yaml.safe_load(content)
        else:
            imported = json.loads(content)
    except (yaml.YAMLError if YAML_AVAILABLE else Exception, json.JSONDecodeError) as e:
        print(f"Error: Could not parse file: {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS

    if not isinstance(imported, dict):
        print("Error: Import file must contain a dictionary of prompts.", file=sys.stderr)
        return EXIT_INVALID_ARGS

    prompts = load_prompts()
    if prompts is None:
        return EXIT_FILE_ERROR

    # Merge prompts
    added = 0
    updated = 0
    skipped = 0

    for name, data in imported.items():
        if not isinstance(data, dict):
            print(f"Warning: Skipping invalid prompt '{name}'", file=sys.stderr)
            skipped += 1
            continue

        # Ensure required fields
        if 'template' not in data:
            print(f"Warning: Skipping prompt '{name}' without template", file=sys.stderr)
            skipped += 1
            continue

        if name in prompts:
            if args.no_overwrite:
                skipped += 1
                continue
            updated += 1
        else:
            added += 1

        # Normalize the prompt data
        now = datetime.now(timezone.utc).isoformat()
        prompts[name] = {
            'template': data.get('template', ''),
            'category': data.get('category', ''),
            'tags': data.get('tags', []) if isinstance(data.get('tags'), list) else [],
            'created_at': data.get('created_at', now),
            'updated_at': now
        }

    if not save_prompts(prompts):
        return EXIT_FILE_ERROR

    print(f"Import complete: {added} added, {updated} updated, {skipped} skipped")
    return EXIT_SUCCESS


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser with all subcommands.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog='prompt_manager',
        description='''
Prompt Manager - Store and manage reusable prompt templates.

Supports variable substitution using Jinja2-style syntax ({{ variable }}),
categories and tags for organization, and import/export as YAML or JSON.

Storage location: ~/.prompt_manager/ (override with PROMPT_MANAGER_DIR env var)
        ''',
        epilog='''
Exit Codes:
  0  Success
  1  Not found (prompt, category, or tag doesn't exist)
  2  File error (read/write/permission issues)
  3  Invalid arguments (bad syntax, missing required args)

Examples:
  %(prog)s add "code-review" --template "Review this {{language}} code:\\n{{code}}"
  %(prog)s add "summarize" --template "Summarize: {{text}}" --category writing --tags "short,utility"
  %(prog)s list
  %(prog)s list --category coding
  %(prog)s get "code-review"
  %(prog)s render "code-review" --vars '{"language": "Python", "code": "def foo():"}'
  %(prog)s update "code-review" --template "New template"
  %(prog)s export --format yaml > prompts.yaml
  %(prog)s import prompts.yaml
  %(prog)s delete "old-prompt"
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Add command
    add_parser = subparsers.add_parser(
        'add',
        help='Add a new prompt template',
        description='Add a new prompt template with optional category and tags.',
        epilog='''
Examples:
  %(prog)s "code-review" --template "Review this {{language}} code:\\n{{code}}"
  %(prog)s "summarize" --template "Summarize: {{text}}" --category writing --tags "short,utility"
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_parser.add_argument('name', help='Unique name for the prompt')
    add_parser.add_argument('--template', '-t', required=True, help='The prompt template text (use {{var}} for variables)')
    add_parser.add_argument('--category', '-c', help='Category for organization (e.g., "coding", "writing")')
    add_parser.add_argument('--tags', help='Comma-separated tags (e.g., "short,utility,python")')
    add_parser.set_defaults(func=cmd_add)

    # List command
    list_parser = subparsers.add_parser(
        'list',
        help='List all prompts',
        description='List all prompts, optionally filtered by category or tag.',
        epilog='''
Examples:
  %(prog)s                    # List all prompts
  %(prog)s --category coding  # List prompts in "coding" category
  %(prog)s --tag utility      # List prompts with "utility" tag
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    list_parser.add_argument('--category', '-c', help='Filter by category')
    list_parser.add_argument('--tag', '-t', help='Filter by tag')
    list_parser.set_defaults(func=cmd_list)

    # Get command
    get_parser = subparsers.add_parser(
        'get',
        help='Get a specific prompt',
        description='Display the full details of a prompt template.',
        epilog='''
Example:
  %(prog)s "code-review"
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser.add_argument('name', help='Name of the prompt to retrieve')
    get_parser.set_defaults(func=cmd_get)

    # Update command
    update_parser = subparsers.add_parser(
        'update',
        help='Update an existing prompt',
        description='Update one or more fields of an existing prompt.',
        epilog='''
Examples:
  %(prog)s "code-review" --template "New template"
  %(prog)s "code-review" --category "new-category"
  %(prog)s "code-review" --tags "new,tags"
  %(prog)s "code-review" --tags ""  # Clear all tags
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    update_parser.add_argument('name', help='Name of the prompt to update')
    update_parser.add_argument('--template', '-t', help='New template text')
    update_parser.add_argument('--category', '-c', help='New category (use "" to clear)')
    update_parser.add_argument('--tags', help='New comma-separated tags (use "" to clear)')
    update_parser.set_defaults(func=cmd_update)

    # Delete command
    delete_parser = subparsers.add_parser(
        'delete',
        help='Delete a prompt',
        description='Permanently delete a prompt template.',
        epilog='''
Example:
  %(prog)s "old-prompt"
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    delete_parser.add_argument('name', help='Name of the prompt to delete')
    delete_parser.set_defaults(func=cmd_delete)

    # Render command
    render_parser = subparsers.add_parser(
        'render',
        help='Render a prompt with variables',
        description='Render a prompt template by substituting variables using Jinja2.',
        epilog='''
Examples:
  %(prog)s "code-review" --vars '{"language": "Python", "code": "def foo():"}'
  %(prog)s "summarize" --vars '{"text": "Long article text here..."}'
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    render_parser.add_argument('name', help='Name of the prompt to render')
    render_parser.add_argument('--vars', '-v', help='JSON object with variable values')
    render_parser.set_defaults(func=cmd_render)

    # Export command
    export_parser = subparsers.add_parser(
        'export',
        help='Export prompts to YAML or JSON',
        description='Export all prompts to stdout in YAML or JSON format.',
        epilog='''
Examples:
  %(prog)s --format yaml > prompts.yaml
  %(prog)s --format json > prompts.json
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    export_parser.add_argument('--format', '-f', default='yaml', choices=['yaml', 'json'], help='Output format (default: yaml)')
    export_parser.set_defaults(func=cmd_export)

    # Import command
    import_parser = subparsers.add_parser(
        'import',
        help='Import prompts from file',
        description='Import prompts from a YAML or JSON file.',
        epilog='''
Examples:
  %(prog)s prompts.yaml
  %(prog)s prompts.json --no-overwrite
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    import_parser.add_argument('file', help='Path to YAML or JSON file to import')
    import_parser.add_argument('--no-overwrite', action='store_true', help='Skip prompts that already exist')
    import_parser.set_defaults(func=cmd_import)

    return parser


def main() -> int:
    """
    Main entry point for the prompt_manager CLI.

    Parses command line arguments and dispatches to the appropriate command handler.

    Returns:
        int: Exit code from the command handler.
    """
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return EXIT_SUCCESS

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
