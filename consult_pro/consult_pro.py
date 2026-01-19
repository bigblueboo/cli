#!/usr/bin/env python3
"""
consult_pro - Query OpenAI GPT-5.2 Pro with background mode for long-running tasks

A CLI tool for AI agents to submit complex prompts to GPT-5.2 Pro and wait
for responses that may take 5-50+ minutes to complete. Uses OpenAI's
Responses API with background mode for reliable long-running inference.

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (required)

Models:
    gpt-5.2-pro (default) - Most advanced reasoning model
    gpt-5.2              - Fast, capable model
    o3                   - Advanced reasoning
    o3-pro               - Extended reasoning

Reasoning Effort (for pro models):
    medium  - Balanced speed/quality
    high    - Higher quality (default for pro)
    xhigh   - Maximum reasoning effort

Usage:
    consult_pro "Solve this complex problem..."
    consult_pro -f problem.txt --effort xhigh
    consult_pro "Deep analysis needed" --model gpt-5.2-pro --wait
    echo "Complex query" | consult_pro --stdin

Exit Codes:
    0: Success
    1: Missing API key
    2: API error
    3: Task failed/cancelled
    4: Invalid arguments
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Generator

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for pro models."""
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class TaskStatus(str, Enum):
    """Possible status values for background tasks."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"


@dataclass
class TaskResult:
    """Result of a completed background task."""
    id: str
    status: str
    model: str
    output_text: Optional[str]
    usage: Optional[dict]
    created_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    error: Optional[str] = None


# Default model
DEFAULT_MODEL = "gpt-5.2-pro"

# Models that support reasoning effort
REASONING_MODELS = {"gpt-5.2-pro", "o3-pro", "o3"}

# Poll interval settings
INITIAL_POLL_INTERVAL = 5  # seconds
MAX_POLL_INTERVAL = 30  # seconds
POLL_BACKOFF_FACTOR = 1.5


def get_api_key() -> str:
    """
    Get OpenAI API key from environment.

    Returns:
        API key string

    Raises:
        SystemExit: If OPENAI_API_KEY not set
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        print("\nGet your API key from: https://platform.openai.com/api-keys", file=sys.stderr)
        sys.exit(1)
    return api_key


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_background_task(
    client: OpenAI,
    prompt: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Response:
    """
    Create a background task for long-running inference.

    Args:
        client: OpenAI client
        prompt: The user prompt
        model: Model to use
        reasoning_effort: Reasoning effort level (for pro models)
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        Response object with task ID and initial status
    """
    # Build the input
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Build request parameters
    params = {
        "model": model,
        "input": messages,
        "background": True,
        "store": True,  # Required for background mode
    }

    # Add reasoning effort for supported models
    if reasoning_effort and model in REASONING_MODELS:
        params["reasoning"] = {"effort": reasoning_effort}

    if temperature is not None:
        params["temperature"] = temperature

    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens

    response = client.responses.create(**params)
    return response


def poll_task(
    client: OpenAI,
    task_id: str,
    verbose: bool = False,
    quiet: bool = False,
) -> Generator[Response, None, Response]:
    """
    Poll a background task until completion.

    Args:
        client: OpenAI client
        task_id: The task/response ID to poll
        verbose: Print detailed status updates
        quiet: Suppress all status output

    Yields:
        Response objects during polling

    Returns:
        Final Response object
    """
    poll_interval = INITIAL_POLL_INTERVAL
    start_time = time.time()
    last_status = None

    while True:
        response = client.responses.retrieve(task_id)
        status = response.status

        # Yield for progress tracking
        yield response

        # Check if we've reached a terminal state
        if status not in (TaskStatus.QUEUED.value, TaskStatus.IN_PROGRESS.value):
            return response

        # Print status update if changed
        if not quiet and status != last_status:
            elapsed = format_duration(time.time() - start_time)
            if verbose:
                print(f"[{elapsed}] Status: {status}", file=sys.stderr)
            else:
                print(f"Status: {status} (elapsed: {elapsed})", file=sys.stderr)
            last_status = status

        # Wait before next poll with exponential backoff
        time.sleep(poll_interval)
        poll_interval = min(poll_interval * POLL_BACKOFF_FACTOR, MAX_POLL_INTERVAL)


def run_task_and_wait(
    client: OpenAI,
    prompt: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> TaskResult:
    """
    Submit a task and wait for completion.

    Args:
        client: OpenAI client
        prompt: The user prompt
        model: Model to use
        reasoning_effort: Reasoning effort level
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        verbose: Print detailed status
        quiet: Suppress status output

    Returns:
        TaskResult with the completed response
    """
    start_time = datetime.now()

    # Create the background task
    if not quiet:
        print(f"Submitting task to {model}...", file=sys.stderr)

    response = create_background_task(
        client=client,
        prompt=prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    task_id = response.id

    if not quiet:
        print(f"Task ID: {task_id}", file=sys.stderr)
        print(f"Initial status: {response.status}", file=sys.stderr)

    # If already complete (unlikely but possible)
    if response.status not in (TaskStatus.QUEUED.value, TaskStatus.IN_PROGRESS.value):
        final_response = response
    else:
        # Poll until completion
        if not quiet:
            print("Waiting for completion...", file=sys.stderr)

        for resp in poll_task(client, task_id, verbose=verbose, quiet=quiet):
            final_response = resp

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Extract output text
    output_text = None
    if hasattr(final_response, 'output_text'):
        output_text = final_response.output_text
    elif hasattr(final_response, 'output') and final_response.output:
        # Handle output array format
        for item in final_response.output:
            if hasattr(item, 'content'):
                for content in item.content:
                    if hasattr(content, 'text'):
                        output_text = content.text
                        break

    # Extract usage info
    usage = None
    if hasattr(final_response, 'usage') and final_response.usage:
        usage = {
            "input_tokens": getattr(final_response.usage, 'input_tokens', None),
            "output_tokens": getattr(final_response.usage, 'output_tokens', None),
            "total_tokens": getattr(final_response.usage, 'total_tokens', None),
        }

    # Check for errors
    error = None
    if final_response.status == TaskStatus.FAILED.value:
        if hasattr(final_response, 'error') and final_response.error:
            error = str(final_response.error)
        else:
            error = "Task failed without error details"

    return TaskResult(
        id=task_id,
        status=final_response.status,
        model=model,
        output_text=output_text,
        usage=usage,
        created_at=start_time,
        completed_at=end_time,
        duration_seconds=duration,
        error=error,
    )


def stream_background_task(
    client: OpenAI,
    prompt: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Generator[str, None, None]:
    """
    Stream a background task's output as it becomes available.

    Args:
        client: OpenAI client
        prompt: The user prompt
        model: Model to use
        reasoning_effort: Reasoning effort level
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens

    Yields:
        Text chunks as they become available
    """
    # Build the input
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Build request parameters
    params = {
        "model": model,
        "input": messages,
        "background": True,
        "stream": True,
        "store": True,
    }

    if reasoning_effort and model in REASONING_MODELS:
        params["reasoning"] = {"effort": reasoning_effort}

    if temperature is not None:
        params["temperature"] = temperature

    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens

    stream = client.responses.create(**params)

    for event in stream:
        if hasattr(event, 'delta') and event.delta:
            yield event.delta
        elif hasattr(event, 'text') and event.text:
            yield event.text


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="consult_pro",
        description="Query OpenAI GPT-5.2 Pro with background mode for long-running tasks. "
                    "Designed for complex prompts that may take 5-50+ minutes to complete.",
        epilog="""
Examples:
  %(prog)s "Analyze this complex problem in depth..."
  %(prog)s -f problem.txt --effort xhigh
  %(prog)s "Write a comprehensive report" --model gpt-5.2-pro
  %(prog)s "Deep analysis needed" --system "You are an expert analyst"
  echo "Complex query" | %(prog)s --stdin
  %(prog)s "Quick task" --no-wait  # Fire and forget, returns task ID

Models:
  gpt-5.2-pro  Most advanced reasoning (default)
  gpt-5.2      Fast, capable model
  o3           Advanced reasoning
  o3-pro       Extended reasoning

Reasoning Effort (pro models only):
  medium   Balanced speed/quality
  high     Higher quality (default)
  xhigh    Maximum reasoning effort

Environment Variables:
  OPENAI_API_KEY  Your OpenAI API key
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input options
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to send to the model"
    )

    parser.add_argument(
        "-f", "--file",
        type=Path,
        help="Read prompt from file"
    )

    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read prompt from stdin"
    )

    # Model options
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )

    parser.add_argument(
        "-e", "--effort",
        choices=["medium", "high", "xhigh"],
        default="high",
        help="Reasoning effort level for pro models (default: high)"
    )

    parser.add_argument(
        "-s", "--system",
        help="System prompt"
    )

    parser.add_argument(
        "--system-file",
        type=Path,
        help="Read system prompt from file"
    )

    parser.add_argument(
        "-t", "--temperature",
        type=float,
        help="Sampling temperature (0.0-2.0)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens in response"
    )

    # Execution options
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit task and return immediately with task ID (don't wait for completion)"
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output as it becomes available"
    )

    parser.add_argument(
        "--poll-id",
        help="Poll an existing task by ID instead of creating new one"
    )

    # Output options
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Write output to file"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON (includes metadata)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress status messages, only output result"
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Check for openai availability after parsing (so --help works)
    if not OPENAI_AVAILABLE:
        print("Error: openai package not installed. Run: pip install openai>=1.50.0", file=sys.stderr)
        return 1

    # Get API key
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)

    # Handle polling existing task
    if args.poll_id:
        if not args.quiet:
            print(f"Polling task: {args.poll_id}", file=sys.stderr)

        try:
            for resp in poll_task(client, args.poll_id, verbose=args.verbose, quiet=args.quiet):
                final = resp

            if final.status == TaskStatus.COMPLETED.value:
                output_text = getattr(final, 'output_text', None) or ""
                print(output_text)
                return 0
            else:
                print(f"Task ended with status: {final.status}", file=sys.stderr)
                return 3

        except Exception as e:
            print(f"Error polling task: {e}", file=sys.stderr)
            return 2

    # Get prompt from various sources
    prompt = None

    if args.stdin or (not args.prompt and not args.file and not sys.stdin.isatty()):
        prompt = sys.stdin.read().strip()
    elif args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 4
        prompt = args.file.read_text().strip()
    elif args.prompt:
        prompt = args.prompt

    if not prompt:
        print("Error: No prompt provided. Use positional argument, -f FILE, or --stdin", file=sys.stderr)
        return 4

    # Get system prompt
    system_prompt = None
    if args.system_file:
        if not args.system_file.exists():
            print(f"Error: System prompt file not found: {args.system_file}", file=sys.stderr)
            return 4
        system_prompt = args.system_file.read_text().strip()
    elif args.system:
        system_prompt = args.system

    try:
        # Handle streaming mode
        if args.stream:
            if not args.quiet:
                print(f"Streaming from {args.model}...", file=sys.stderr)

            output_parts = []
            for chunk in stream_background_task(
                client=client,
                prompt=prompt,
                model=args.model,
                reasoning_effort=args.effort,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ):
                print(chunk, end="", flush=True)
                output_parts.append(chunk)

            print()  # Final newline

            if args.output:
                args.output.write_text("".join(output_parts))

            return 0

        # Handle fire-and-forget mode
        if args.no_wait:
            response = create_background_task(
                client=client,
                prompt=prompt,
                model=args.model,
                reasoning_effort=args.effort,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            if args.json:
                result = {
                    "id": response.id,
                    "status": response.status,
                    "model": args.model,
                    "message": "Task submitted. Use --poll-id to check status."
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"Task ID: {response.id}")
                print(f"Status: {response.status}")
                print(f"\nTo check status: consult_pro --poll-id {response.id}")

            return 0

        # Standard mode: submit and wait
        result = run_task_and_wait(
            client=client,
            prompt=prompt,
            model=args.model,
            reasoning_effort=args.effort,
            system_prompt=system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            quiet=args.quiet,
        )

        # Handle output
        if result.status == TaskStatus.COMPLETED.value:
            if args.json:
                output = {
                    "id": result.id,
                    "status": result.status,
                    "model": result.model,
                    "output": result.output_text,
                    "usage": result.usage,
                    "duration_seconds": result.duration_seconds,
                    "created_at": result.created_at.isoformat(),
                    "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                }
                output_str = json.dumps(output, indent=2)
            else:
                output_str = result.output_text or ""

            if args.output:
                args.output.write_text(output_str)
                if not args.quiet:
                    print(f"Output written to: {args.output}", file=sys.stderr)
            else:
                print(output_str)

            if not args.quiet and not args.json:
                print(f"\n[Completed in {format_duration(result.duration_seconds)}]", file=sys.stderr)
                if result.usage:
                    print(f"[Tokens: {result.usage.get('total_tokens', 'N/A')}]", file=sys.stderr)

            return 0

        else:
            print(f"Task failed with status: {result.status}", file=sys.stderr)
            if result.error:
                print(f"Error: {result.error}", file=sys.stderr)
            return 3

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
