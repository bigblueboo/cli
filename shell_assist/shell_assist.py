#!/usr/bin/env python3
"""
shell_assist - A self-documenting CLI tool for shell command assistance.

This tool provides:
- Natural language to shell command conversion
- Explanation of existing shell commands
- Command improvement suggestions and fixes
- Optional command execution with confirmation
- Automatic shell detection (bash, zsh, fish)

Exit Codes:
    0 - Success
    1 - API error (authentication, network, rate limit)
    2 - Execution error (command failed to run)
    3 - Invalid arguments

Environment Variables:
    OPENAI_API_KEY    - OpenAI API key (checked first)
    ANTHROPIC_API_KEY - Anthropic API key (checked second)
    GOOGLE_API_KEY    - Google API key (checked third)

Examples:
    # Convert natural language to shell commands
    shell_assist "find all python files modified in the last week"
    shell_assist "compress all log files older than 30 days"
    shell_assist "count lines in all js files" --shell zsh

    # Explain existing commands
    shell_assist explain "find . -name '*.py' -mtime -7 -exec wc -l {} +"
    shell_assist explain "awk '{sum+=$1} END {print sum}'"

    # Execute commands with confirmation
    shell_assist "list docker containers" --execute

    # Get improvement suggestions
    shell_assist fix "grep -r pattern"

Author: shell_assist
Version: 1.0.0
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import Optional, Tuple

# Version information
__version__ = "1.0.0"

# Exit codes
EXIT_SUCCESS = 0
EXIT_API_ERROR = 1
EXIT_EXECUTION_ERROR = 2
EXIT_INVALID_ARGS = 3

# Destructive commands that require safety warnings
DESTRUCTIVE_COMMANDS = {
    'rm': 'Removes files or directories permanently',
    'rmdir': 'Removes directories',
    'dd': 'Low-level data copy that can overwrite disks',
    'mkfs': 'Formats a filesystem, destroying all data',
    'fdisk': 'Disk partitioning tool that can destroy data',
    'parted': 'Disk partitioning tool that can destroy data',
    'shred': 'Securely deletes files by overwriting',
    'truncate': 'Shrinks or extends file size, potentially losing data',
    'mv': 'Moves or renames files, can overwrite existing files',
    'cp': 'Copies files, can overwrite existing files',
    'chmod': 'Changes file permissions',
    'chown': 'Changes file ownership',
    'kill': 'Terminates processes',
    'killall': 'Terminates processes by name',
    'pkill': 'Terminates processes by pattern',
    'reboot': 'Reboots the system',
    'shutdown': 'Shuts down the system',
    'halt': 'Halts the system',
    'poweroff': 'Powers off the system',
    'init': 'Changes system runlevel',
    'systemctl': 'Controls system services (start/stop/restart)',
    'service': 'Controls system services',
    ':>': 'Truncates file to zero length',
    '>': 'Redirects output, can overwrite files',
}

# Dangerous patterns in commands
DANGEROUS_PATTERNS = [
    (r'rm\s+(-[rf]+\s+)*/', 'Recursive removal from root directory'),
    (r'rm\s+(-[rf]+\s+)*\*', 'Recursive removal with wildcard'),
    (r'dd\s+.*of=/dev/', 'Writing directly to a device'),
    (r'mkfs', 'Formatting a filesystem'),
    (r'>\s*/dev/', 'Writing directly to a device'),
    (r'chmod\s+(-R\s+)?777', 'Setting overly permissive permissions'),
    (r'curl.*\|\s*(sudo\s+)?(ba)?sh', 'Piping remote script to shell'),
    (r'wget.*\|\s*(sudo\s+)?(ba)?sh', 'Piping remote script to shell'),
    (r':\(\)\s*\{\s*:\|:&\s*\}\s*;', 'Fork bomb detected'),
]


def detect_shell() -> str:
    """
    Detect the current shell being used.

    Returns:
        str: The detected shell name ('bash', 'zsh', 'fish', or 'sh')

    Examples:
        >>> shell = detect_shell()
        >>> shell in ['bash', 'zsh', 'fish', 'sh']
        True
    """
    # Check SHELL environment variable
    shell_path = os.environ.get('SHELL', '')
    if shell_path:
        shell_name = os.path.basename(shell_path)
        if shell_name in ['bash', 'zsh', 'fish', 'sh', 'dash', 'ksh', 'tcsh', 'csh']:
            return shell_name

    # Try to detect from parent process
    try:
        ppid = os.getppid()
        with open(f'/proc/{ppid}/comm', 'r') as f:
            comm = f.read().strip()
            if comm in ['bash', 'zsh', 'fish', 'sh', 'dash', 'ksh', 'tcsh', 'csh']:
                return comm
    except (FileNotFoundError, PermissionError, OSError):
        pass

    # Default to bash
    return 'bash'


def get_api_client() -> Tuple[str, object]:
    """
    Get the appropriate API client based on available environment variables.

    Checks for API keys in order: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY

    Returns:
        Tuple[str, object]: A tuple of (provider_name, client_instance)

    Raises:
        EnvironmentError: If no API key is found in environment variables

    Examples:
        >>> os.environ['OPENAI_API_KEY'] = 'test-key'
        >>> provider, client = get_api_client()
        >>> provider
        'openai'
    """
    # Check for OpenAI API key
    if os.environ.get('OPENAI_API_KEY'):
        try:
            from openai import OpenAI
            return 'openai', OpenAI()
        except ImportError:
            pass

    # Check for Anthropic API key
    if os.environ.get('ANTHROPIC_API_KEY'):
        try:
            from anthropic import Anthropic
            return 'anthropic', Anthropic()
        except ImportError:
            pass

    # Check for Google API key
    if os.environ.get('GOOGLE_API_KEY'):
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            return 'google', genai
        except ImportError:
            pass

    raise EnvironmentError(
        "No API key found. Please set one of the following environment variables:\n"
        "  - OPENAI_API_KEY\n"
        "  - ANTHROPIC_API_KEY\n"
        "  - GOOGLE_API_KEY"
    )


def call_llm(provider: str, client: object, prompt: str, system_prompt: str) -> str:
    """
    Call the LLM API with the given prompt.

    Args:
        provider: The API provider name ('openai', 'anthropic', 'google')
        client: The API client instance
        prompt: The user prompt to send
        system_prompt: The system prompt for context

    Returns:
        str: The LLM response text

    Raises:
        Exception: If the API call fails
    """
    if provider == 'openai':
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    elif provider == 'anthropic':
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip()

    elif provider == 'google':
        model = client.GenerativeModel('gemini-pro')
        full_prompt = f"{system_prompt}\n\nUser request: {prompt}"
        response = model.generate_content(full_prompt)
        return response.text.strip()

    raise ValueError(f"Unknown provider: {provider}")


def check_destructive_commands(command: str) -> list:
    """
    Check if a command contains potentially destructive operations.

    Args:
        command: The shell command to check

    Returns:
        list: List of warning messages for detected dangers

    Examples:
        >>> warnings = check_destructive_commands("rm -rf /")
        >>> len(warnings) > 0
        True
        >>> "rm" in warnings[0].lower() or "root" in warnings[0].lower()
        True
    """
    warnings = []

    # Check for dangerous patterns first (more specific)
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            warnings.append(f"DANGER: {description}")

    # Check for destructive commands
    for cmd, description in DESTRUCTIVE_COMMANDS.items():
        # Match command at start of string or after pipe/semicolon/&&/||
        pattern = rf'(^|[|;&]\s*){re.escape(cmd)}(\s|$)'
        if re.search(pattern, command):
            warnings.append(f"WARNING: '{cmd}' - {description}")

    return warnings


def generate_command(query: str, shell: str) -> str:
    """
    Convert natural language to a shell command.

    Args:
        query: Natural language description of desired command
        shell: Target shell (bash, zsh, fish)

    Returns:
        str: The generated shell command

    Examples:
        >>> # This would call the LLM in real usage
        >>> cmd = "find . -name '*.py' -mtime -7"
        >>> isinstance(cmd, str)
        True
    """
    provider, client = get_api_client()

    system_prompt = f"""You are a shell command expert. Convert natural language requests to {shell} commands.

Rules:
1. Output ONLY the command, no explanations
2. Use portable commands when possible
3. For {shell}-specific features, use {shell} syntax
4. Prefer safe options (e.g., -i for interactive rm when appropriate)
5. Use quotes to handle filenames with spaces
6. For complex operations, use single-line commands with pipes or && when appropriate

Examples:
- "find python files" -> find . -name "*.py"
- "list files by size" -> ls -lhS
- "count lines in all js files" -> find . -name "*.js" -exec wc -l {{}} +
"""

    prompt = f"Convert to {shell} command: {query}"

    return call_llm(provider, client, prompt, system_prompt)


def explain_command(command: str, shell: str) -> str:
    """
    Explain what a shell command does.

    Args:
        command: The shell command to explain
        shell: The shell context (bash, zsh, fish)

    Returns:
        str: Detailed explanation of the command

    Examples:
        >>> explanation = "This command finds Python files..."
        >>> isinstance(explanation, str)
        True
    """
    provider, client = get_api_client()

    system_prompt = f"""You are a shell command expert. Explain shell commands in detail.

Format your response as:
1. Brief summary (one line)
2. Breakdown of each part/flag
3. What the command does step by step
4. Any potential risks or side effects

Context: {shell} shell
"""

    prompt = f"Explain this command:\n{command}"

    return call_llm(provider, client, prompt, system_prompt)


def fix_command(command: str, shell: str) -> str:
    """
    Suggest improvements or fixes for a shell command.

    Args:
        command: The shell command to improve
        shell: The shell context (bash, zsh, fish)

    Returns:
        str: Suggestions for improving the command

    Examples:
        >>> suggestion = "Consider using -r for recursive..."
        >>> isinstance(suggestion, str)
        True
    """
    provider, client = get_api_client()

    system_prompt = f"""You are a shell command expert. Analyze and suggest improvements for shell commands.

Consider:
1. Correctness - Is the command correct?
2. Safety - Are there safer alternatives?
3. Efficiency - Can it be more efficient?
4. Portability - Will it work across systems?
5. Best practices - Modern alternatives?

Format:
1. Issues found (if any)
2. Suggested improved command
3. Explanation of improvements

Context: {shell} shell
"""

    prompt = f"Analyze and improve this command:\n{command}"

    return call_llm(provider, client, prompt, system_prompt)


def execute_command(command: str, shell: str) -> Tuple[int, str, str]:
    """
    Execute a shell command and return results.

    Args:
        command: The command to execute
        shell: The shell to use for execution

    Returns:
        Tuple[int, str, str]: (return_code, stdout, stderr)

    Examples:
        >>> code, out, err = execute_command("echo hello", "bash")
        >>> code
        0
        >>> "hello" in out
        True
    """
    shell_path = shutil.which(shell) or shutil.which('sh')

    try:
        result = subprocess.run(
            [shell_path, '-c', command],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, '', 'Command timed out after 60 seconds'
    except Exception as e:
        return 1, '', str(e)


def confirm_execution(command: str) -> bool:
    """
    Ask user to confirm command execution.

    Args:
        command: The command to be executed

    Returns:
        bool: True if user confirms, False otherwise
    """
    print(f"\nCommand to execute:\n  {command}\n")

    # Check for destructive commands
    warnings = check_destructive_commands(command)
    if warnings:
        print("\n".join(warnings))
        print()

    try:
        response = input("Execute this command? [y/N]: ").strip().lower()
        return response in ('y', 'yes')
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog='shell_assist',
        description='''
shell_assist - AI-powered shell command assistant

Convert natural language to shell commands, explain existing commands,
and get suggestions for improvements.
        ''',
        epilog='''
Examples:
  Natural language to command:
    %(prog)s "find all python files modified in the last week"
    %(prog)s "compress all log files older than 30 days"
    %(prog)s "count lines in all js files" --shell zsh

  Explain a command:
    %(prog)s explain "find . -name '*.py' -mtime -7 -exec wc -l {} +"
    %(prog)s explain "awk '{sum+=$1} END {print sum}'"

  Execute with confirmation:
    %(prog)s "list docker containers" --execute

  Get improvement suggestions:
    %(prog)s fix "grep -r pattern"

Exit Codes:
  0 - Success
  1 - API error (authentication, network, rate limit)
  2 - Execution error (command failed)
  3 - Invalid arguments

Environment Variables:
  OPENAI_API_KEY    - OpenAI API key (checked first)
  ANTHROPIC_API_KEY - Anthropic API key (checked second)
  GOOGLE_API_KEY    - Google API key (checked third)

For more information, visit: https://github.com/example/shell_assist
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'mode',
        nargs='?',
        choices=['explain', 'fix'],
        help='Operation mode: explain (explain a command) or fix (suggest improvements)'
    )

    parser.add_argument(
        'query',
        nargs='?',
        help='Natural language query or command to process'
    )

    parser.add_argument(
        '--shell', '-s',
        choices=['bash', 'zsh', 'fish', 'sh'],
        default=None,
        help='Target shell (default: auto-detect)'
    )

    parser.add_argument(
        '--execute', '-x',
        action='store_true',
        help='Execute the generated command after confirmation'
    )

    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt (use with caution!)'
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    parser.add_argument(
        '--no-warnings',
        action='store_true',
        help='Suppress safety warnings for destructive commands'
    )

    return parser


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for shell_assist.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        int: Exit code (0=success, 1=API error, 2=exec error, 3=invalid args)

    Examples:
        >>> main(['--help'])  # doctest: +SKIP
        0
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    # Handle the case where mode is actually the query
    if parsed.mode and parsed.mode not in ['explain', 'fix']:
        # Mode is actually part of the query
        if parsed.query:
            query = f"{parsed.mode} {parsed.query}"
        else:
            query = parsed.mode
        mode = 'generate'
    elif parsed.mode in ['explain', 'fix']:
        mode = parsed.mode
        query = parsed.query
    else:
        query = parsed.query
        mode = 'generate'

    # Validate query
    if not query:
        parser.print_help()
        return EXIT_INVALID_ARGS

    # Detect or use specified shell
    shell = parsed.shell or detect_shell()

    try:
        if mode == 'explain':
            print(f"Explaining command for {shell}...\n")
            result = explain_command(query, shell)
            print(result)
            return EXIT_SUCCESS

        elif mode == 'fix':
            print(f"Analyzing command for {shell}...\n")
            result = fix_command(query, shell)
            print(result)
            return EXIT_SUCCESS

        else:  # generate
            print(f"Generating {shell} command...\n")
            command = generate_command(query, shell)

            # Clean up the command (remove markdown code blocks if present)
            command = re.sub(r'^```\w*\n?', '', command)
            command = re.sub(r'\n?```$', '', command)
            command = command.strip()

            print(f"Command: {command}")

            # Show safety warnings unless suppressed
            if not parsed.no_warnings:
                warnings = check_destructive_commands(command)
                if warnings:
                    print()
                    for warning in warnings:
                        print(f"  {warning}")

            # Execute if requested
            if parsed.execute:
                if parsed.yes or confirm_execution(command):
                    print("\nExecuting...\n")
                    code, stdout, stderr = execute_command(command, shell)

                    if stdout:
                        print(stdout)
                    if stderr:
                        print(stderr, file=sys.stderr)

                    if code != 0:
                        print(f"\nCommand exited with code {code}")
                        return EXIT_EXECUTION_ERROR
                else:
                    print("Execution cancelled.")

            return EXIT_SUCCESS

    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_API_ERROR

    except Exception as e:
        print(f"API Error: {e}", file=sys.stderr)
        return EXIT_API_ERROR


if __name__ == '__main__':
    sys.exit(main())
