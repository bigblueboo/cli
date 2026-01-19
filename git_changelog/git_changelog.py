#!/usr/bin/env python3
"""
git_changelog - Generate changelogs from git commit history

A self-documenting CLI tool that parses conventional commits and generates
formatted changelogs. Supports multiple output formats and grouping options.

Conventional Commit Format:
    type(scope): description

    Types: feat, fix, docs, style, refactor, perf, test, chore

Exit Codes:
    0 - Success
    1 - Not a git repository
    2 - Invalid version/commit range
    3 - Git error
    4 - Invalid arguments

Examples:
    # Generate changelog between two tags
    git_changelog --from v1.0.0 --to v1.1.0

    # Generate changelog for recent commits
    git_changelog --since "2 weeks ago" --format markdown

    # Group commits by type
    git_changelog --from HEAD~20 --group-by type

    # Group by author and output as JSON
    git_changelog --from v1.0.0 --group-by author --format json

    # Generate changelog for all commits
    git_changelog --all --format plain

Author: Generated CLI Tool
License: MIT
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

try:
    import git
    from git.exc import InvalidGitRepositoryError, GitCommandError, BadName
except ImportError:
    print("Error: gitpython is required. Install with: pip install gitpython", file=sys.stderr)
    sys.exit(4)


# Exit codes
EXIT_SUCCESS = 0
EXIT_NOT_GIT_REPO = 1
EXIT_INVALID_RANGE = 2
EXIT_GIT_ERROR = 3
EXIT_INVALID_ARGS = 4

# Conventional commit types with descriptions
COMMIT_TYPES = {
    'feat': 'Features',
    'fix': 'Bug Fixes',
    'docs': 'Documentation',
    'style': 'Styles',
    'refactor': 'Code Refactoring',
    'perf': 'Performance Improvements',
    'test': 'Tests',
    'chore': 'Chores',
    'build': 'Build System',
    'ci': 'Continuous Integration',
    'revert': 'Reverts',
}

# Regex pattern for conventional commits: type(scope): description
CONVENTIONAL_COMMIT_PATTERN = re.compile(
    r'^(?P<type>\w+)(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?\s*:\s*(?P<description>.+)$',
    re.MULTILINE
)


class OutputFormat(Enum):
    """Supported output formats."""
    MARKDOWN = 'markdown'
    JSON = 'json'
    PLAIN = 'plain'


class GroupBy(Enum):
    """Grouping options for commits."""
    NONE = 'none'
    TYPE = 'type'
    AUTHOR = 'author'
    SCOPE = 'scope'


@dataclass
class ParsedCommit:
    """Represents a parsed conventional commit."""
    sha: str
    short_sha: str
    type: str
    scope: Optional[str]
    description: str
    breaking: bool
    author: str
    author_email: str
    date: datetime
    full_message: str
    is_conventional: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'sha': self.sha,
            'short_sha': self.short_sha,
            'type': self.type,
            'scope': self.scope,
            'description': self.description,
            'breaking': self.breaking,
            'author': self.author,
            'author_email': self.author_email,
            'date': self.date.isoformat(),
            'full_message': self.full_message,
            'is_conventional': self.is_conventional,
        }


class GitChangelog:
    """Main class for generating changelogs from git history."""

    def __init__(self, repo_path: str = '.'):
        """
        Initialize the changelog generator.

        Args:
            repo_path: Path to the git repository (default: current directory)

        Raises:
            SystemExit: If the path is not a git repository
        """
        try:
            self.repo = git.Repo(repo_path, search_parent_directories=True)
        except InvalidGitRepositoryError:
            print(f"Error: '{repo_path}' is not a git repository", file=sys.stderr)
            sys.exit(EXIT_NOT_GIT_REPO)

    def parse_commit(self, commit: git.Commit) -> ParsedCommit:
        """
        Parse a git commit into a ParsedCommit object.

        Args:
            commit: A gitpython Commit object

        Returns:
            ParsedCommit object with parsed conventional commit data
        """
        message = commit.message.strip()
        first_line = message.split('\n')[0]

        match = CONVENTIONAL_COMMIT_PATTERN.match(first_line)

        if match:
            return ParsedCommit(
                sha=commit.hexsha,
                short_sha=commit.hexsha[:7],
                type=match.group('type'),
                scope=match.group('scope'),
                description=match.group('description'),
                breaking=match.group('breaking') == '!',
                author=commit.author.name,
                author_email=commit.author.email,
                date=datetime.fromtimestamp(commit.committed_date),
                full_message=message,
                is_conventional=True,
            )
        else:
            return ParsedCommit(
                sha=commit.hexsha,
                short_sha=commit.hexsha[:7],
                type='other',
                scope=None,
                description=first_line,
                breaking=False,
                author=commit.author.name,
                author_email=commit.author.email,
                date=datetime.fromtimestamp(commit.committed_date),
                full_message=message,
                is_conventional=False,
            )

    def get_commits(
        self,
        from_ref: Optional[str] = None,
        to_ref: Optional[str] = None,
        since: Optional[str] = None,
        all_commits: bool = False,
    ) -> List[ParsedCommit]:
        """
        Get commits from the repository within the specified range.

        Args:
            from_ref: Starting reference (tag, branch, or commit)
            to_ref: Ending reference (default: HEAD)
            since: Time-based filter (e.g., "2 weeks ago")
            all_commits: If True, get all commits

        Returns:
            List of ParsedCommit objects

        Raises:
            SystemExit: On invalid range or git errors
        """
        try:
            if all_commits:
                commits = list(self.repo.iter_commits())
            elif since:
                commits = list(self.repo.iter_commits(since=since))
            elif from_ref:
                to_ref = to_ref or 'HEAD'
                # Build revision range
                rev_range = f"{from_ref}..{to_ref}"
                commits = list(self.repo.iter_commits(rev_range))
            else:
                # Default to last 10 commits
                commits = list(self.repo.iter_commits(max_count=10))

            return [self.parse_commit(c) for c in commits]

        except BadName as e:
            print(f"Error: Invalid reference - {e}", file=sys.stderr)
            sys.exit(EXIT_INVALID_RANGE)
        except GitCommandError as e:
            print(f"Error: Git command failed - {e}", file=sys.stderr)
            sys.exit(EXIT_GIT_ERROR)

    def group_commits(
        self,
        commits: List[ParsedCommit],
        group_by: GroupBy,
    ) -> Dict[str, List[ParsedCommit]]:
        """
        Group commits by the specified attribute.

        Args:
            commits: List of ParsedCommit objects
            group_by: Grouping option (type, author, scope, or none)

        Returns:
            Dictionary mapping group names to lists of commits
        """
        if group_by == GroupBy.NONE:
            return {'All Commits': commits}

        grouped = defaultdict(list)

        for commit in commits:
            if group_by == GroupBy.TYPE:
                key = COMMIT_TYPES.get(commit.type, f"Other ({commit.type})")
            elif group_by == GroupBy.AUTHOR:
                key = commit.author
            elif group_by == GroupBy.SCOPE:
                key = commit.scope or 'No Scope'
            else:
                key = 'All Commits'

            grouped[key].append(commit)

        # Sort groups for consistent output
        if group_by == GroupBy.TYPE:
            # Order by conventional commit type order
            type_order = list(COMMIT_TYPES.values()) + ['Other']
            sorted_groups = {}
            for type_name in type_order:
                for key in grouped:
                    if key.startswith(type_name.split()[0]) or key == type_name:
                        sorted_groups[key] = grouped[key]
            # Add any remaining groups
            for key in grouped:
                if key not in sorted_groups:
                    sorted_groups[key] = grouped[key]
            return sorted_groups

        return dict(sorted(grouped.items()))

    def format_markdown(
        self,
        grouped_commits: Dict[str, List[ParsedCommit]],
        title: Optional[str] = None,
    ) -> str:
        """
        Format commits as Markdown.

        Args:
            grouped_commits: Dictionary of grouped commits
            title: Optional title for the changelog

        Returns:
            Markdown formatted string
        """
        lines = []

        if title:
            lines.append(f"# {title}")
            lines.append("")

        for group_name, commits in grouped_commits.items():
            if not commits:
                continue

            lines.append(f"## {group_name}")
            lines.append("")

            for commit in commits:
                scope_str = f"**{commit.scope}**: " if commit.scope else ""
                breaking_str = "**BREAKING** " if commit.breaking else ""
                lines.append(
                    f"- {breaking_str}{scope_str}{commit.description} "
                    f"([{commit.short_sha}](commit/{commit.sha}))"
                )

            lines.append("")

        return '\n'.join(lines)

    def format_plain(
        self,
        grouped_commits: Dict[str, List[ParsedCommit]],
        title: Optional[str] = None,
    ) -> str:
        """
        Format commits as plain text.

        Args:
            grouped_commits: Dictionary of grouped commits
            title: Optional title for the changelog

        Returns:
            Plain text formatted string
        """
        lines = []

        if title:
            lines.append(title)
            lines.append("=" * len(title))
            lines.append("")

        for group_name, commits in grouped_commits.items():
            if not commits:
                continue

            lines.append(group_name)
            lines.append("-" * len(group_name))

            for commit in commits:
                scope_str = f"[{commit.scope}] " if commit.scope else ""
                breaking_str = "[BREAKING] " if commit.breaking else ""
                lines.append(f"  * {breaking_str}{scope_str}{commit.description} ({commit.short_sha})")

            lines.append("")

        return '\n'.join(lines)

    def format_json(
        self,
        grouped_commits: Dict[str, List[ParsedCommit]],
        title: Optional[str] = None,
    ) -> str:
        """
        Format commits as JSON.

        Args:
            grouped_commits: Dictionary of grouped commits
            title: Optional title for the changelog

        Returns:
            JSON formatted string
        """
        output = {
            'title': title,
            'generated_at': datetime.now().isoformat(),
            'groups': {
                name: [c.to_dict() for c in commits]
                for name, commits in grouped_commits.items()
            },
            'total_commits': sum(len(c) for c in grouped_commits.values()),
        }
        return json.dumps(output, indent=2)

    def generate(
        self,
        from_ref: Optional[str] = None,
        to_ref: Optional[str] = None,
        since: Optional[str] = None,
        all_commits: bool = False,
        group_by: GroupBy = GroupBy.NONE,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        title: Optional[str] = None,
    ) -> str:
        """
        Generate a changelog with the specified options.

        Args:
            from_ref: Starting reference
            to_ref: Ending reference
            since: Time-based filter
            all_commits: If True, include all commits
            group_by: Grouping option
            output_format: Output format (markdown, json, plain)
            title: Optional title for the changelog

        Returns:
            Formatted changelog string
        """
        commits = self.get_commits(
            from_ref=from_ref,
            to_ref=to_ref,
            since=since,
            all_commits=all_commits,
        )

        if not commits:
            return "No commits found in the specified range."

        grouped = self.group_commits(commits, group_by)

        if output_format == OutputFormat.MARKDOWN:
            return self.format_markdown(grouped, title)
        elif output_format == OutputFormat.JSON:
            return self.format_json(grouped, title)
        else:
            return self.format_plain(grouped, title)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser with comprehensive help.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog='git_changelog',
        description='''
Generate changelogs from git commit history.

This tool parses conventional commits and generates formatted changelogs.
It supports multiple output formats and various grouping options.

CONVENTIONAL COMMIT FORMAT:
    type(scope): description

    Supported types:
        feat     - New features
        fix      - Bug fixes
        docs     - Documentation changes
        style    - Code style changes (formatting, etc.)
        refactor - Code refactoring
        perf     - Performance improvements
        test     - Adding or updating tests
        chore    - Maintenance tasks
        build    - Build system changes
        ci       - CI configuration changes
        revert   - Reverting previous commits

EXIT CODES:
    0 - Success
    1 - Not a git repository
    2 - Invalid version/commit range
    3 - Git error
    4 - Invalid arguments
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
EXAMPLES:
    Generate changelog between two tags:
        %(prog)s --from v1.0.0 --to v1.1.0

    Generate changelog for commits in the last 2 weeks:
        %(prog)s --since "2 weeks ago" --format markdown

    Group commits by type (features, fixes, etc.):
        %(prog)s --from HEAD~20 --group-by type

    Group by author and output as JSON:
        %(prog)s --from v1.0.0 --group-by author --format json

    Generate changelog for all commits:
        %(prog)s --all --format plain

    Add a custom title:
        %(prog)s --from v1.0.0 --to v1.1.0 --title "Release Notes v1.1.0"

    Generate changelog from a specific directory:
        %(prog)s --repo /path/to/repo --from v1.0.0

For more information, visit: https://github.com/example/git_changelog
        ''',
    )

    # Version range options
    range_group = parser.add_argument_group(
        'Range Selection',
        'Options for specifying the commit range'
    )
    range_group.add_argument(
        '--from', '-f',
        dest='from_ref',
        metavar='REF',
        help='Starting reference (tag, branch, or commit SHA). '
             'Example: v1.0.0, main, HEAD~10, abc1234'
    )
    range_group.add_argument(
        '--to', '-t',
        dest='to_ref',
        metavar='REF',
        default='HEAD',
        help='Ending reference (default: HEAD). '
             'Example: v1.1.0, develop, HEAD'
    )
    range_group.add_argument(
        '--since', '-s',
        metavar='DATE',
        help='Include commits more recent than DATE. '
             'Example: "2 weeks ago", "2024-01-01", "yesterday"'
    )
    range_group.add_argument(
        '--all', '-a',
        dest='all_commits',
        action='store_true',
        help='Include all commits in the repository'
    )

    # Output options
    output_group = parser.add_argument_group(
        'Output Options',
        'Options for controlling the output format'
    )
    output_group.add_argument(
        '--format', '-F',
        dest='output_format',
        choices=['markdown', 'json', 'plain'],
        default='markdown',
        help='Output format (default: markdown). '
             'markdown - GitHub-flavored Markdown, '
             'json - Structured JSON, '
             'plain - Plain text'
    )
    output_group.add_argument(
        '--group-by', '-g',
        dest='group_by',
        choices=['none', 'type', 'author', 'scope'],
        default='none',
        help='Group commits by attribute (default: none). '
             'type - Group by commit type (feat, fix, etc.), '
             'author - Group by commit author, '
             'scope - Group by commit scope'
    )
    output_group.add_argument(
        '--title', '-T',
        metavar='TITLE',
        help='Add a title to the changelog. '
             'Example: "Release Notes v1.1.0"'
    )
    output_group.add_argument(
        '--output', '-o',
        metavar='FILE',
        help='Write output to FILE instead of stdout'
    )

    # Repository options
    repo_group = parser.add_argument_group(
        'Repository Options',
        'Options for specifying the repository'
    )
    repo_group.add_argument(
        '--repo', '-r',
        dest='repo_path',
        default='.',
        metavar='PATH',
        help='Path to the git repository (default: current directory)'
    )

    # Additional options
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='%(prog)s 1.0.0',
        help='Show program version and exit'
    )
    parser.add_argument(
        '--verbose', '-V',
        action='store_true',
        help='Enable verbose output (show additional information)'
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.

    Args:
        args: Parsed arguments namespace

    Raises:
        SystemExit: On invalid argument combinations
    """
    # Check for conflicting options
    if args.since and args.from_ref:
        print(
            "Error: --since and --from are mutually exclusive. "
            "Use one or the other.",
            file=sys.stderr
        )
        sys.exit(EXIT_INVALID_ARGS)

    if args.all_commits and (args.from_ref or args.since):
        print(
            "Error: --all cannot be used with --from or --since.",
            file=sys.stderr
        )
        sys.exit(EXIT_INVALID_ARGS)


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    # Create changelog generator
    changelog = GitChangelog(args.repo_path)

    # Generate changelog
    result = changelog.generate(
        from_ref=args.from_ref,
        to_ref=args.to_ref,
        since=args.since,
        all_commits=args.all_commits,
        group_by=GroupBy(args.group_by),
        output_format=OutputFormat(args.output_format),
        title=args.title,
    )

    # Output result
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(result)
            if args.verbose:
                print(f"Changelog written to {args.output}", file=sys.stderr)
        except IOError as e:
            print(f"Error writing to file: {e}", file=sys.stderr)
            return EXIT_GIT_ERROR
    else:
        print(result)

    return EXIT_SUCCESS


if __name__ == '__main__':
    sys.exit(main())
