#!/usr/bin/env python3
"""
github_ops - A self-documenting CLI tool for GitHub operations.

This tool provides a command-line interface for common GitHub operations including:
- Creating, reading, listing, and updating issues
- Commenting on pull requests
- Creating releases with notes
- Triggering repository dispatch events

Authentication:
    Set the GITHUB_TOKEN environment variable with a personal access token
    that has appropriate permissions (repo scope for most operations).

Exit Codes:
    0 - Success
    1 - Missing GITHUB_TOKEN
    2 - Resource not found (404)
    3 - API error (other HTTP errors)
    4 - Invalid arguments

Examples:
    # List open issues in a repository
    github_ops issue list --repo owner/repo --state open

    # Create a new issue
    github_ops issue create --repo owner/repo --title "Bug report" --body "Description"

    # Get details of a specific issue
    github_ops issue get --repo owner/repo --number 42

    # Update an existing issue
    github_ops issue update --repo owner/repo --number 42 --state closed

    # Comment on a pull request
    github_ops pr comment --repo owner/repo --pr 123 --body "LGTM!"

    # Create a release
    github_ops release create --repo owner/repo --tag v1.0.0 --notes "Release notes"

    # Trigger a repository dispatch event
    github_ops dispatch --repo owner/repo --event-type deploy --payload '{"env": "prod"}'
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import requests

# Exit codes
EXIT_SUCCESS = 0
EXIT_MISSING_TOKEN = 1
EXIT_NOT_FOUND = 2
EXIT_API_ERROR = 3
EXIT_INVALID_ARGS = 4

# GitHub API base URL
GITHUB_API_URL = "https://api.github.com"


class GitHubAPIError(Exception):
    """Exception raised for GitHub API errors."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class GitHubOps:
    """Client for GitHub REST API operations."""

    def __init__(self, token: str):
        """
        Initialize the GitHub operations client.

        Args:
            token: GitHub personal access token
        """
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def _parse_repo(self, repo: str) -> tuple:
        """Parse owner/repo format into (owner, repo) tuple."""
        parts = repo.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid repo format: {repo}. Expected 'owner/repo'")
        return parts[0], parts[1]

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make a request to the GitHub API.

        Args:
            method: HTTP method (GET, POST, PATCH, etc.)
            endpoint: API endpoint (e.g., /repos/owner/repo/issues)
            params: Query parameters
            json_data: JSON body data

        Returns:
            Response data as dictionary

        Raises:
            GitHubAPIError: If the API returns an error
        """
        url = f"{GITHUB_API_URL}{endpoint}"
        response = self.session.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
        )

        if response.status_code == 404:
            raise GitHubAPIError("Resource not found", 404)
        elif response.status_code == 401:
            raise GitHubAPIError("Authentication failed. Check your GITHUB_TOKEN", 401)
        elif response.status_code == 403:
            raise GitHubAPIError("Permission denied or rate limited", 403)
        elif response.status_code >= 400:
            error_msg = response.json().get("message", "Unknown error")
            raise GitHubAPIError(f"API error: {error_msg}", response.status_code)

        if response.status_code == 204:
            return {}

        return response.json()

    # Issue operations
    def list_issues(
        self,
        repo: str,
        state: str = "open",
        labels: Optional[str] = None,
        sort: str = "created",
        direction: str = "desc",
        per_page: int = 30,
        page: int = 1,
    ) -> list:
        """
        List issues in a repository.

        Args:
            repo: Repository in 'owner/repo' format
            state: Filter by state ('open', 'closed', 'all')
            labels: Comma-separated list of label names
            sort: Sort by ('created', 'updated', 'comments')
            direction: Sort direction ('asc', 'desc')
            per_page: Number of results per page (max 100)
            page: Page number

        Returns:
            List of issues
        """
        owner, repo_name = self._parse_repo(repo)
        params = {
            "state": state,
            "sort": sort,
            "direction": direction,
            "per_page": per_page,
            "page": page,
        }
        if labels:
            params["labels"] = labels

        return self._request("GET", f"/repos/{owner}/{repo_name}/issues", params=params)

    def get_issue(self, repo: str, issue_number: int) -> Dict:
        """
        Get a single issue.

        Args:
            repo: Repository in 'owner/repo' format
            issue_number: Issue number

        Returns:
            Issue data
        """
        owner, repo_name = self._parse_repo(repo)
        return self._request("GET", f"/repos/{owner}/{repo_name}/issues/{issue_number}")

    def create_issue(
        self,
        repo: str,
        title: str,
        body: Optional[str] = None,
        labels: Optional[list] = None,
        assignees: Optional[list] = None,
        milestone: Optional[int] = None,
    ) -> Dict:
        """
        Create a new issue.

        Args:
            repo: Repository in 'owner/repo' format
            title: Issue title
            body: Issue body/description
            labels: List of label names
            assignees: List of usernames to assign
            milestone: Milestone number

        Returns:
            Created issue data
        """
        owner, repo_name = self._parse_repo(repo)
        data = {"title": title}
        if body:
            data["body"] = body
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees
        if milestone:
            data["milestone"] = milestone

        return self._request("POST", f"/repos/{owner}/{repo_name}/issues", json_data=data)

    def update_issue(
        self,
        repo: str,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        labels: Optional[list] = None,
        assignees: Optional[list] = None,
        milestone: Optional[int] = None,
    ) -> Dict:
        """
        Update an existing issue.

        Args:
            repo: Repository in 'owner/repo' format
            issue_number: Issue number to update
            title: New title
            body: New body/description
            state: New state ('open' or 'closed')
            labels: List of label names
            assignees: List of usernames to assign
            milestone: Milestone number

        Returns:
            Updated issue data
        """
        owner, repo_name = self._parse_repo(repo)
        data = {}
        if title is not None:
            data["title"] = title
        if body is not None:
            data["body"] = body
        if state is not None:
            data["state"] = state
        if labels is not None:
            data["labels"] = labels
        if assignees is not None:
            data["assignees"] = assignees
        if milestone is not None:
            data["milestone"] = milestone

        if not data:
            raise ValueError("At least one field must be provided for update")

        return self._request(
            "PATCH", f"/repos/{owner}/{repo_name}/issues/{issue_number}", json_data=data
        )

    # Pull Request comment operations
    def create_pr_comment(self, repo: str, pr_number: int, body: str) -> Dict:
        """
        Create a comment on a pull request.

        Note: Pull requests are treated as issues in the GitHub API for comments.

        Args:
            repo: Repository in 'owner/repo' format
            pr_number: Pull request number
            body: Comment body

        Returns:
            Created comment data
        """
        owner, repo_name = self._parse_repo(repo)
        data = {"body": body}
        return self._request(
            "POST", f"/repos/{owner}/{repo_name}/issues/{pr_number}/comments", json_data=data
        )

    # Release operations
    def create_release(
        self,
        repo: str,
        tag_name: str,
        name: Optional[str] = None,
        body: Optional[str] = None,
        target_commitish: Optional[str] = None,
        draft: bool = False,
        prerelease: bool = False,
        generate_release_notes: bool = False,
    ) -> Dict:
        """
        Create a new release.

        Args:
            repo: Repository in 'owner/repo' format
            tag_name: The name of the tag for the release
            name: Release name/title
            body: Release notes/description
            target_commitish: Branch or commit SHA for the tag (defaults to main branch)
            draft: True to create a draft release
            prerelease: True to mark as a prerelease
            generate_release_notes: True to auto-generate release notes

        Returns:
            Created release data
        """
        owner, repo_name = self._parse_repo(repo)
        data = {"tag_name": tag_name}
        if name:
            data["name"] = name
        if body:
            data["body"] = body
        if target_commitish:
            data["target_commitish"] = target_commitish
        if draft:
            data["draft"] = draft
        if prerelease:
            data["prerelease"] = prerelease
        if generate_release_notes:
            data["generate_release_notes"] = generate_release_notes

        return self._request("POST", f"/repos/{owner}/{repo_name}/releases", json_data=data)

    def list_releases(self, repo: str, per_page: int = 30, page: int = 1) -> list:
        """
        List releases for a repository.

        Args:
            repo: Repository in 'owner/repo' format
            per_page: Number of results per page (max 100)
            page: Page number

        Returns:
            List of releases
        """
        owner, repo_name = self._parse_repo(repo)
        params = {"per_page": per_page, "page": page}
        return self._request("GET", f"/repos/{owner}/{repo_name}/releases", params=params)

    # Repository dispatch operations
    def create_dispatch_event(
        self,
        repo: str,
        event_type: str,
        client_payload: Optional[Dict] = None,
    ) -> Dict:
        """
        Create a repository dispatch event.

        This triggers the 'repository_dispatch' webhook event, which can be used
        to trigger GitHub Actions workflows.

        Args:
            repo: Repository in 'owner/repo' format
            event_type: Custom webhook event name
            client_payload: JSON payload to send with the event (max 10 properties)

        Returns:
            Empty dict on success (API returns 204)
        """
        owner, repo_name = self._parse_repo(repo)
        data = {"event_type": event_type}
        if client_payload:
            if len(client_payload) > 10:
                raise ValueError("client_payload cannot have more than 10 top-level properties")
            data["client_payload"] = client_payload

        return self._request("POST", f"/repos/{owner}/{repo_name}/dispatches", json_data=data)


def get_token() -> str:
    """Get GitHub token from environment variable."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable is not set.", file=sys.stderr)
        print(
            "\nTo use this tool, set your GitHub personal access token:",
            file=sys.stderr,
        )
        print("  export GITHUB_TOKEN=your_token_here", file=sys.stderr)
        sys.exit(EXIT_MISSING_TOKEN)
    return token


def format_issue(issue: Dict) -> str:
    """Format an issue for display."""
    lines = [
        f"#{issue['number']}: {issue['title']}",
        f"  State: {issue['state']}",
        f"  URL: {issue['html_url']}",
        f"  Created: {issue['created_at']}",
        f"  Author: {issue['user']['login']}",
    ]
    if issue.get("labels"):
        labels = ", ".join(label["name"] for label in issue["labels"])
        lines.append(f"  Labels: {labels}")
    if issue.get("assignees"):
        assignees = ", ".join(a["login"] for a in issue["assignees"])
        lines.append(f"  Assignees: {assignees}")
    return "\n".join(lines)


def format_release(release: Dict) -> str:
    """Format a release for display."""
    name = release.get('name') or '(no name)'
    lines = [
        f"{release['tag_name']}: {name}",
        f"  URL: {release['html_url']}",
        f"  Created: {release['created_at']}",
        f"  Author: {release['author']['login']}",
    ]
    if release.get("draft"):
        lines.append("  Status: DRAFT")
    if release.get("prerelease"):
        lines.append("  Status: PRERELEASE")
    return "\n".join(lines)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="github_ops",
        description="""
GitHub Operations CLI Tool

A command-line interface for common GitHub operations including issues,
pull request comments, releases, and repository dispatch events.

AUTHENTICATION:
    Set the GITHUB_TOKEN environment variable with a personal access token.
    The token needs 'repo' scope for most operations.

EXIT CODES:
    0 - Success
    1 - Missing GITHUB_TOKEN
    2 - Resource not found (404)
    3 - API error (other HTTP errors)
    4 - Invalid arguments
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # List open issues
    %(prog)s issue list --repo owner/repo --state open

    # Create an issue with labels
    %(prog)s issue create --repo owner/repo --title "Bug" --body "Details" --labels bug,urgent

    # Comment on a PR
    %(prog)s pr comment --repo owner/repo --pr 123 --body "LGTM!"

    # Create a release
    %(prog)s release create --repo owner/repo --tag v1.0.0 --name "Version 1.0" --notes "Release notes"

    # Trigger a dispatch event
    %(prog)s dispatch --repo owner/repo --event-type deploy --payload '{"env": "production"}'

For more information, visit: https://docs.github.com/en/rest
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Issue command
    issue_parser = subparsers.add_parser(
        "issue",
        help="Issue operations (create, list, get, update)",
        description="Manage GitHub issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # List all open issues
    github_ops issue list --repo owner/repo

    # List closed issues with specific labels
    github_ops issue list --repo owner/repo --state closed --labels bug,high-priority

    # Create a new issue
    github_ops issue create --repo owner/repo --title "Bug report" --body "Description here"

    # Get issue details
    github_ops issue get --repo owner/repo --number 42

    # Close an issue
    github_ops issue update --repo owner/repo --number 42 --state closed
        """,
    )
    issue_subparsers = issue_parser.add_subparsers(dest="action", help="Issue action")

    # issue list
    issue_list = issue_subparsers.add_parser(
        "list",
        help="List issues in a repository",
        description="List issues in a repository with optional filters",
        epilog="Example: github_ops issue list --repo owner/repo --state open --labels bug",
    )
    issue_list.add_argument(
        "--repo", "-r", required=True, help="Repository in 'owner/repo' format"
    )
    issue_list.add_argument(
        "--state",
        "-s",
        choices=["open", "closed", "all"],
        default="open",
        help="Filter by state (default: open)",
    )
    issue_list.add_argument(
        "--labels", "-l", help="Comma-separated list of label names"
    )
    issue_list.add_argument(
        "--sort",
        choices=["created", "updated", "comments"],
        default="created",
        help="Sort by (default: created)",
    )
    issue_list.add_argument(
        "--direction",
        choices=["asc", "desc"],
        default="desc",
        help="Sort direction (default: desc)",
    )
    issue_list.add_argument(
        "--per-page",
        type=int,
        default=30,
        help="Results per page, max 100 (default: 30)",
    )
    issue_list.add_argument("--page", type=int, default=1, help="Page number (default: 1)")

    # issue get
    issue_get = issue_subparsers.add_parser(
        "get",
        help="Get a specific issue",
        description="Get details of a specific issue",
        epilog="Example: github_ops issue get --repo owner/repo --number 42",
    )
    issue_get.add_argument(
        "--repo", "-r", required=True, help="Repository in 'owner/repo' format"
    )
    issue_get.add_argument(
        "--number", "-n", type=int, required=True, help="Issue number"
    )

    # issue create
    issue_create = issue_subparsers.add_parser(
        "create",
        help="Create a new issue",
        description="Create a new issue in a repository",
        epilog='Example: github_ops issue create --repo owner/repo --title "Bug" --body "Description"',
    )
    issue_create.add_argument(
        "--repo", "-r", required=True, help="Repository in 'owner/repo' format"
    )
    issue_create.add_argument("--title", "-t", required=True, help="Issue title")
    issue_create.add_argument("--body", "-b", help="Issue body/description")
    issue_create.add_argument(
        "--labels", "-l", help="Comma-separated list of label names"
    )
    issue_create.add_argument(
        "--assignees", "-a", help="Comma-separated list of usernames to assign"
    )
    issue_create.add_argument(
        "--milestone", "-m", type=int, help="Milestone number"
    )

    # issue update
    issue_update = issue_subparsers.add_parser(
        "update",
        help="Update an existing issue",
        description="Update an existing issue in a repository",
        epilog="Example: github_ops issue update --repo owner/repo --number 42 --state closed",
    )
    issue_update.add_argument(
        "--repo", "-r", required=True, help="Repository in 'owner/repo' format"
    )
    issue_update.add_argument(
        "--number", "-n", type=int, required=True, help="Issue number"
    )
    issue_update.add_argument("--title", "-t", help="New issue title")
    issue_update.add_argument("--body", "-b", help="New issue body/description")
    issue_update.add_argument(
        "--state", "-s", choices=["open", "closed"], help="New state"
    )
    issue_update.add_argument(
        "--labels", "-l", help="Comma-separated list of label names"
    )
    issue_update.add_argument(
        "--assignees", "-a", help="Comma-separated list of usernames to assign"
    )
    issue_update.add_argument(
        "--milestone", "-m", type=int, help="Milestone number"
    )

    # PR command
    pr_parser = subparsers.add_parser(
        "pr",
        help="Pull request operations (comment)",
        description="Manage pull request comments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Add a comment to a PR
    github_ops pr comment --repo owner/repo --pr 123 --body "LGTM!"

    # Add a multi-line comment
    github_ops pr comment --repo owner/repo --pr 123 --body "Great work!

    Some suggestions:
    - Consider adding tests
    - Update the docs"
        """,
    )
    pr_subparsers = pr_parser.add_subparsers(dest="action", help="PR action")

    # pr comment
    pr_comment = pr_subparsers.add_parser(
        "comment",
        help="Add a comment to a pull request",
        description="Create a new comment on a pull request",
        epilog='Example: github_ops pr comment --repo owner/repo --pr 123 --body "LGTM!"',
    )
    pr_comment.add_argument(
        "--repo", "-r", required=True, help="Repository in 'owner/repo' format"
    )
    pr_comment.add_argument(
        "--pr", "-p", type=int, required=True, help="Pull request number"
    )
    pr_comment.add_argument("--body", "-b", required=True, help="Comment body")

    # Release command
    release_parser = subparsers.add_parser(
        "release",
        help="Release operations (create, list)",
        description="Manage GitHub releases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Create a release
    github_ops release create --repo owner/repo --tag v1.0.0 --name "Version 1.0" --notes "Release notes"

    # Create a draft release
    github_ops release create --repo owner/repo --tag v2.0.0 --draft

    # Create a prerelease with auto-generated notes
    github_ops release create --repo owner/repo --tag v1.1.0-beta --prerelease --generate-notes

    # List releases
    github_ops release list --repo owner/repo
        """,
    )
    release_subparsers = release_parser.add_subparsers(dest="action", help="Release action")

    # release create
    release_create = release_subparsers.add_parser(
        "create",
        help="Create a new release",
        description="Create a new release in a repository",
        epilog='Example: github_ops release create --repo owner/repo --tag v1.0.0 --notes "Release notes"',
    )
    release_create.add_argument(
        "--repo", "-r", required=True, help="Repository in 'owner/repo' format"
    )
    release_create.add_argument(
        "--tag", "-t", required=True, help="Tag name for the release"
    )
    release_create.add_argument("--name", "-n", help="Release name/title")
    release_create.add_argument("--notes", help="Release notes/description")
    release_create.add_argument(
        "--target",
        help="Branch or commit SHA for the tag (defaults to default branch)",
    )
    release_create.add_argument(
        "--draft", action="store_true", help="Create as a draft release"
    )
    release_create.add_argument(
        "--prerelease", action="store_true", help="Mark as a prerelease"
    )
    release_create.add_argument(
        "--generate-notes",
        action="store_true",
        help="Auto-generate release notes",
    )

    # release list
    release_list = release_subparsers.add_parser(
        "list",
        help="List releases in a repository",
        description="List releases in a repository",
        epilog="Example: github_ops release list --repo owner/repo",
    )
    release_list.add_argument(
        "--repo", "-r", required=True, help="Repository in 'owner/repo' format"
    )
    release_list.add_argument(
        "--per-page",
        type=int,
        default=30,
        help="Results per page, max 100 (default: 30)",
    )
    release_list.add_argument("--page", type=int, default=1, help="Page number (default: 1)")

    # Dispatch command
    dispatch_parser = subparsers.add_parser(
        "dispatch",
        help="Trigger a repository dispatch event",
        description="""
Trigger a repository dispatch event.

This creates a 'repository_dispatch' webhook event that can be used to trigger
GitHub Actions workflows. The workflow must be configured to listen for the
specific event_type.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Simple dispatch event
    github_ops dispatch --repo owner/repo --event-type deploy

    # Dispatch with payload
    github_ops dispatch --repo owner/repo --event-type deploy --payload '{"env": "prod", "version": "1.0.0"}'

NOTE:
    - The workflow must be on the default branch to receive dispatch events
    - The client_payload is limited to 10 top-level properties
    - Your token needs 'repo' scope and contents:write permission
        """,
    )
    dispatch_parser.add_argument(
        "--repo", "-r", required=True, help="Repository in 'owner/repo' format"
    )
    dispatch_parser.add_argument(
        "--event-type", "-e", required=True, help="Custom webhook event type name"
    )
    dispatch_parser.add_argument(
        "--payload",
        "-p",
        help="JSON payload to send with the event (max 10 top-level properties)",
    )

    return parser


def handle_issue_command(client: GitHubOps, args: argparse.Namespace) -> int:
    """Handle issue subcommands."""
    if args.action == "list":
        issues = client.list_issues(
            repo=args.repo,
            state=args.state,
            labels=args.labels,
            sort=args.sort,
            direction=args.direction,
            per_page=args.per_page,
            page=args.page,
        )
        if not issues:
            print("No issues found.")
        else:
            for issue in issues:
                print(format_issue(issue))
                print()
        return EXIT_SUCCESS

    elif args.action == "get":
        issue = client.get_issue(repo=args.repo, issue_number=args.number)
        print(format_issue(issue))
        if issue.get("body"):
            print(f"\n{issue['body']}")
        return EXIT_SUCCESS

    elif args.action == "create":
        labels = args.labels.split(",") if args.labels else None
        assignees = args.assignees.split(",") if args.assignees else None
        issue = client.create_issue(
            repo=args.repo,
            title=args.title,
            body=args.body,
            labels=labels,
            assignees=assignees,
            milestone=args.milestone,
        )
        print(f"Created issue #{issue['number']}: {issue['title']}")
        print(f"URL: {issue['html_url']}")
        return EXIT_SUCCESS

    elif args.action == "update":
        labels = args.labels.split(",") if args.labels else None
        assignees = args.assignees.split(",") if args.assignees else None
        issue = client.update_issue(
            repo=args.repo,
            issue_number=args.number,
            title=args.title,
            body=args.body,
            state=args.state,
            labels=labels,
            assignees=assignees,
            milestone=args.milestone,
        )
        print(f"Updated issue #{issue['number']}: {issue['title']}")
        print(f"State: {issue['state']}")
        print(f"URL: {issue['html_url']}")
        return EXIT_SUCCESS

    else:
        print("Error: Please specify an action (list, get, create, update)", file=sys.stderr)
        return EXIT_INVALID_ARGS


def handle_pr_command(client: GitHubOps, args: argparse.Namespace) -> int:
    """Handle PR subcommands."""
    if args.action == "comment":
        comment = client.create_pr_comment(
            repo=args.repo,
            pr_number=args.pr,
            body=args.body,
        )
        print(f"Created comment on PR #{args.pr}")
        print(f"URL: {comment['html_url']}")
        return EXIT_SUCCESS
    else:
        print("Error: Please specify an action (comment)", file=sys.stderr)
        return EXIT_INVALID_ARGS


def handle_release_command(client: GitHubOps, args: argparse.Namespace) -> int:
    """Handle release subcommands."""
    if args.action == "create":
        release = client.create_release(
            repo=args.repo,
            tag_name=args.tag,
            name=args.name,
            body=args.notes,
            target_commitish=args.target,
            draft=args.draft,
            prerelease=args.prerelease,
            generate_release_notes=args.generate_notes,
        )
        print(f"Created release: {release['tag_name']}")
        if release.get("name"):
            print(f"Name: {release['name']}")
        print(f"URL: {release['html_url']}")
        if release.get("draft"):
            print("Status: DRAFT")
        if release.get("prerelease"):
            print("Status: PRERELEASE")
        return EXIT_SUCCESS

    elif args.action == "list":
        releases = client.list_releases(
            repo=args.repo,
            per_page=args.per_page,
            page=args.page,
        )
        if not releases:
            print("No releases found.")
        else:
            for release in releases:
                print(format_release(release))
                print()
        return EXIT_SUCCESS

    else:
        print("Error: Please specify an action (create, list)", file=sys.stderr)
        return EXIT_INVALID_ARGS


def handle_dispatch_command(client: GitHubOps, args: argparse.Namespace) -> int:
    """Handle dispatch subcommand."""
    payload = None
    if args.payload:
        try:
            payload = json.loads(args.payload)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON payload: {e}", file=sys.stderr)
            return EXIT_INVALID_ARGS

    client.create_dispatch_event(
        repo=args.repo,
        event_type=args.event_type,
        client_payload=payload,
    )
    print(f"Triggered repository dispatch event: {args.event_type}")
    print(f"Repository: {args.repo}")
    if payload:
        print(f"Payload: {json.dumps(payload, indent=2)}")
    return EXIT_SUCCESS


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return EXIT_INVALID_ARGS

    # Get token (exits with EXIT_MISSING_TOKEN if not set)
    token = get_token()
    client = GitHubOps(token)

    try:
        if args.command == "issue":
            return handle_issue_command(client, args)
        elif args.command == "pr":
            return handle_pr_command(client, args)
        elif args.command == "release":
            return handle_release_command(client, args)
        elif args.command == "dispatch":
            return handle_dispatch_command(client, args)
        else:
            parser.print_help()
            return EXIT_INVALID_ARGS

    except GitHubAPIError as e:
        print(f"Error: {e}", file=sys.stderr)
        if e.status_code == 404:
            return EXIT_NOT_FOUND
        return EXIT_API_ERROR
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS
    except requests.RequestException as e:
        print(f"Error: Network error: {e}", file=sys.stderr)
        return EXIT_API_ERROR


if __name__ == "__main__":
    sys.exit(main())
