#!/usr/bin/env python3
"""
Comprehensive tests for github_ops CLI tool.

These tests use mocking to avoid making actual API calls to GitHub.
Run with: pytest test_github_ops.py -v
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
import github_ops
from github_ops import (
    EXIT_API_ERROR,
    EXIT_INVALID_ARGS,
    EXIT_MISSING_TOKEN,
    EXIT_NOT_FOUND,
    EXIT_SUCCESS,
    GitHubAPIError,
    GitHubOps,
    create_parser,
    format_issue,
    format_release,
    get_token,
    main,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_token():
    """Provide a mock GitHub token."""
    return "ghp_test_token_12345"


@pytest.fixture
def client(mock_token):
    """Create a GitHubOps client with mocked session."""
    return GitHubOps(mock_token)


@pytest.fixture
def sample_issue():
    """Return a sample issue response."""
    return {
        "number": 42,
        "title": "Test Issue",
        "state": "open",
        "html_url": "https://github.com/owner/repo/issues/42",
        "created_at": "2024-01-15T10:00:00Z",
        "user": {"login": "testuser"},
        "labels": [{"name": "bug"}, {"name": "urgent"}],
        "assignees": [{"login": "developer1"}],
        "body": "This is the issue body.",
    }


@pytest.fixture
def sample_release():
    """Return a sample release response."""
    return {
        "tag_name": "v1.0.0",
        "name": "Version 1.0.0",
        "html_url": "https://github.com/owner/repo/releases/tag/v1.0.0",
        "created_at": "2024-01-15T10:00:00Z",
        "author": {"login": "releaseuser"},
        "draft": False,
        "prerelease": False,
        "body": "Release notes here.",
    }


@pytest.fixture
def sample_comment():
    """Return a sample comment response."""
    return {
        "id": 123456,
        "html_url": "https://github.com/owner/repo/issues/42#issuecomment-123456",
        "body": "LGTM!",
        "user": {"login": "reviewer"},
        "created_at": "2024-01-15T10:00:00Z",
    }


# =============================================================================
# GitHubOps Client Tests
# =============================================================================


class TestGitHubOpsClient:
    """Tests for the GitHubOps client class."""

    def test_init_sets_headers(self, mock_token):
        """Test that client initialization sets correct headers."""
        client = GitHubOps(mock_token)
        assert client.session.headers["Authorization"] == f"Bearer {mock_token}"
        assert "application/vnd.github+json" in client.session.headers["Accept"]
        assert "X-GitHub-Api-Version" in client.session.headers

    def test_parse_repo_valid(self, client):
        """Test parsing valid repo format."""
        owner, repo = client._parse_repo("owner/repo")
        assert owner == "owner"
        assert repo == "repo"

    def test_parse_repo_invalid(self, client):
        """Test parsing invalid repo format raises error."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            client._parse_repo("invalid-format")

    def test_parse_repo_too_many_slashes(self, client):
        """Test parsing repo with too many slashes raises error."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            client._parse_repo("owner/repo/extra")

    @patch.object(GitHubOps, "_request")
    def test_list_issues(self, mock_request, client, sample_issue):
        """Test listing issues."""
        mock_request.return_value = [sample_issue]

        issues = client.list_issues("owner/repo", state="open")

        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert args[0] == "GET"
        assert "/repos/owner/repo/issues" in args[1]
        assert kwargs["params"]["state"] == "open"
        assert len(issues) == 1
        assert issues[0]["number"] == 42

    @patch.object(GitHubOps, "_request")
    def test_list_issues_with_labels(self, mock_request, client, sample_issue):
        """Test listing issues with label filter."""
        mock_request.return_value = [sample_issue]

        client.list_issues("owner/repo", labels="bug,feature")

        args, kwargs = mock_request.call_args
        assert kwargs["params"]["labels"] == "bug,feature"

    @patch.object(GitHubOps, "_request")
    def test_get_issue(self, mock_request, client, sample_issue):
        """Test getting a specific issue."""
        mock_request.return_value = sample_issue

        issue = client.get_issue("owner/repo", 42)

        mock_request.assert_called_once_with(
            "GET", "/repos/owner/repo/issues/42"
        )
        assert issue["number"] == 42

    @patch.object(GitHubOps, "_request")
    def test_create_issue_minimal(self, mock_request, client, sample_issue):
        """Test creating an issue with minimal parameters."""
        mock_request.return_value = sample_issue

        issue = client.create_issue("owner/repo", title="Test Issue")

        args, kwargs = mock_request.call_args
        assert args[0] == "POST"
        assert "/repos/owner/repo/issues" in args[1]
        assert kwargs["json_data"]["title"] == "Test Issue"
        assert "body" not in kwargs["json_data"]

    @patch.object(GitHubOps, "_request")
    def test_create_issue_full(self, mock_request, client, sample_issue):
        """Test creating an issue with all parameters."""
        mock_request.return_value = sample_issue

        issue = client.create_issue(
            "owner/repo",
            title="Test Issue",
            body="Issue body",
            labels=["bug", "urgent"],
            assignees=["user1"],
            milestone=1,
        )

        args, kwargs = mock_request.call_args
        data = kwargs["json_data"]
        assert data["title"] == "Test Issue"
        assert data["body"] == "Issue body"
        assert data["labels"] == ["bug", "urgent"]
        assert data["assignees"] == ["user1"]
        assert data["milestone"] == 1

    @patch.object(GitHubOps, "_request")
    def test_update_issue(self, mock_request, client, sample_issue):
        """Test updating an issue."""
        sample_issue["state"] = "closed"
        mock_request.return_value = sample_issue

        issue = client.update_issue(
            "owner/repo",
            issue_number=42,
            state="closed",
            title="Updated Title",
        )

        args, kwargs = mock_request.call_args
        assert args[0] == "PATCH"
        assert "/repos/owner/repo/issues/42" in args[1]
        assert kwargs["json_data"]["state"] == "closed"
        assert kwargs["json_data"]["title"] == "Updated Title"

    def test_update_issue_no_fields(self, client):
        """Test updating an issue with no fields raises error."""
        with pytest.raises(ValueError, match="At least one field"):
            client.update_issue("owner/repo", 42)

    @patch.object(GitHubOps, "_request")
    def test_create_pr_comment(self, mock_request, client, sample_comment):
        """Test creating a PR comment."""
        mock_request.return_value = sample_comment

        comment = client.create_pr_comment("owner/repo", 123, "LGTM!")

        args, kwargs = mock_request.call_args
        assert args[0] == "POST"
        assert "/repos/owner/repo/issues/123/comments" in args[1]
        assert kwargs["json_data"]["body"] == "LGTM!"

    @patch.object(GitHubOps, "_request")
    def test_create_release_minimal(self, mock_request, client, sample_release):
        """Test creating a release with minimal parameters."""
        mock_request.return_value = sample_release

        release = client.create_release("owner/repo", tag_name="v1.0.0")

        args, kwargs = mock_request.call_args
        assert args[0] == "POST"
        assert "/repos/owner/repo/releases" in args[1]
        assert kwargs["json_data"]["tag_name"] == "v1.0.0"

    @patch.object(GitHubOps, "_request")
    def test_create_release_full(self, mock_request, client, sample_release):
        """Test creating a release with all parameters."""
        sample_release["draft"] = True
        sample_release["prerelease"] = True
        mock_request.return_value = sample_release

        release = client.create_release(
            "owner/repo",
            tag_name="v1.0.0",
            name="Version 1.0.0",
            body="Release notes",
            target_commitish="main",
            draft=True,
            prerelease=True,
            generate_release_notes=True,
        )

        args, kwargs = mock_request.call_args
        data = kwargs["json_data"]
        assert data["tag_name"] == "v1.0.0"
        assert data["name"] == "Version 1.0.0"
        assert data["body"] == "Release notes"
        assert data["target_commitish"] == "main"
        assert data["draft"] is True
        assert data["prerelease"] is True
        assert data["generate_release_notes"] is True

    @patch.object(GitHubOps, "_request")
    def test_list_releases(self, mock_request, client, sample_release):
        """Test listing releases."""
        mock_request.return_value = [sample_release]

        releases = client.list_releases("owner/repo")

        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert args[0] == "GET"
        assert "/repos/owner/repo/releases" in args[1]
        assert len(releases) == 1

    @patch.object(GitHubOps, "_request")
    def test_create_dispatch_event_simple(self, mock_request, client):
        """Test creating a dispatch event without payload."""
        mock_request.return_value = {}

        client.create_dispatch_event("owner/repo", "deploy")

        args, kwargs = mock_request.call_args
        assert args[0] == "POST"
        assert "/repos/owner/repo/dispatches" in args[1]
        assert kwargs["json_data"]["event_type"] == "deploy"
        assert "client_payload" not in kwargs["json_data"]

    @patch.object(GitHubOps, "_request")
    def test_create_dispatch_event_with_payload(self, mock_request, client):
        """Test creating a dispatch event with payload."""
        mock_request.return_value = {}
        payload = {"env": "prod", "version": "1.0.0"}

        client.create_dispatch_event("owner/repo", "deploy", client_payload=payload)

        args, kwargs = mock_request.call_args
        assert kwargs["json_data"]["client_payload"] == payload

    def test_create_dispatch_event_payload_too_many_properties(self, client):
        """Test dispatch event with too many payload properties raises error."""
        payload = {f"key{i}": i for i in range(11)}  # 11 properties

        with pytest.raises(ValueError, match="more than 10"):
            client.create_dispatch_event("owner/repo", "deploy", client_payload=payload)


# =============================================================================
# HTTP Request Tests
# =============================================================================


class TestHTTPRequests:
    """Tests for HTTP request handling."""

    @patch("requests.Session.request")
    def test_request_success(self, mock_request, client):
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_request.return_value = mock_response

        result = client._request("GET", "/test")

        assert result == {"data": "test"}

    @patch("requests.Session.request")
    def test_request_204_no_content(self, mock_request, client):
        """Test 204 response returns empty dict."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_request.return_value = mock_response

        result = client._request("POST", "/test")

        assert result == {}

    @patch("requests.Session.request")
    def test_request_404_not_found(self, mock_request, client):
        """Test 404 response raises GitHubAPIError."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_request.return_value = mock_response

        with pytest.raises(GitHubAPIError) as exc_info:
            client._request("GET", "/test")

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value).lower()

    @patch("requests.Session.request")
    def test_request_401_unauthorized(self, mock_request, client):
        """Test 401 response raises GitHubAPIError."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_request.return_value = mock_response

        with pytest.raises(GitHubAPIError) as exc_info:
            client._request("GET", "/test")

        assert exc_info.value.status_code == 401
        assert "authentication" in str(exc_info.value).lower()

    @patch("requests.Session.request")
    def test_request_403_forbidden(self, mock_request, client):
        """Test 403 response raises GitHubAPIError."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_request.return_value = mock_response

        with pytest.raises(GitHubAPIError) as exc_info:
            client._request("GET", "/test")

        assert exc_info.value.status_code == 403

    @patch("requests.Session.request")
    def test_request_generic_error(self, mock_request, client):
        """Test generic error response raises GitHubAPIError."""
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.json.return_value = {"message": "Validation failed"}
        mock_request.return_value = mock_response

        with pytest.raises(GitHubAPIError) as exc_info:
            client._request("POST", "/test", json_data={"bad": "data"})

        assert "Validation failed" in str(exc_info.value)


# =============================================================================
# Token Tests
# =============================================================================


class TestToken:
    """Tests for token handling."""

    def test_get_token_success(self, mock_token):
        """Test getting token from environment."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": mock_token}):
            token = get_token()
            assert token == mock_token

    def test_get_token_missing(self):
        """Test missing token exits with correct code."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove GITHUB_TOKEN if it exists
            os.environ.pop("GITHUB_TOKEN", None)
            with pytest.raises(SystemExit) as exc_info:
                get_token()
            assert exc_info.value.code == EXIT_MISSING_TOKEN


# =============================================================================
# Formatter Tests
# =============================================================================


class TestFormatters:
    """Tests for output formatters."""

    def test_format_issue_basic(self, sample_issue):
        """Test basic issue formatting."""
        output = format_issue(sample_issue)

        assert "#42: Test Issue" in output
        assert "State: open" in output
        assert "https://github.com/owner/repo/issues/42" in output
        assert "Author: testuser" in output

    def test_format_issue_with_labels(self, sample_issue):
        """Test issue formatting with labels."""
        output = format_issue(sample_issue)

        assert "Labels: bug, urgent" in output

    def test_format_issue_with_assignees(self, sample_issue):
        """Test issue formatting with assignees."""
        output = format_issue(sample_issue)

        assert "Assignees: developer1" in output

    def test_format_issue_no_labels(self, sample_issue):
        """Test issue formatting without labels."""
        sample_issue["labels"] = []
        output = format_issue(sample_issue)

        assert "Labels:" not in output

    def test_format_release_basic(self, sample_release):
        """Test basic release formatting."""
        output = format_release(sample_release)

        assert "v1.0.0: Version 1.0.0" in output
        assert "https://github.com/owner/repo/releases/tag/v1.0.0" in output
        assert "Author: releaseuser" in output

    def test_format_release_draft(self, sample_release):
        """Test release formatting for draft."""
        sample_release["draft"] = True
        output = format_release(sample_release)

        assert "DRAFT" in output

    def test_format_release_prerelease(self, sample_release):
        """Test release formatting for prerelease."""
        sample_release["prerelease"] = True
        output = format_release(sample_release)

        assert "PRERELEASE" in output


# =============================================================================
# CLI Parser Tests
# =============================================================================


class TestCLIParser:
    """Tests for the argument parser."""

    def test_parser_no_args(self):
        """Test parser with no arguments."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_issue_list_command(self):
        """Test issue list command parsing."""
        parser = create_parser()
        args = parser.parse_args(["issue", "list", "--repo", "owner/repo"])

        assert args.command == "issue"
        assert args.action == "list"
        assert args.repo == "owner/repo"
        assert args.state == "open"  # default

    def test_issue_list_with_options(self):
        """Test issue list command with options."""
        parser = create_parser()
        args = parser.parse_args([
            "issue", "list",
            "--repo", "owner/repo",
            "--state", "closed",
            "--labels", "bug,feature",
            "--sort", "updated",
            "--per-page", "50",
        ])

        assert args.state == "closed"
        assert args.labels == "bug,feature"
        assert args.sort == "updated"
        assert args.per_page == 50

    def test_issue_create_command(self):
        """Test issue create command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "issue", "create",
            "--repo", "owner/repo",
            "--title", "Bug Report",
            "--body", "Description here",
        ])

        assert args.command == "issue"
        assert args.action == "create"
        assert args.title == "Bug Report"
        assert args.body == "Description here"

    def test_issue_get_command(self):
        """Test issue get command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "issue", "get",
            "--repo", "owner/repo",
            "--number", "42",
        ])

        assert args.action == "get"
        assert args.number == 42

    def test_issue_update_command(self):
        """Test issue update command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "issue", "update",
            "--repo", "owner/repo",
            "--number", "42",
            "--state", "closed",
        ])

        assert args.action == "update"
        assert args.number == 42
        assert args.state == "closed"

    def test_pr_comment_command(self):
        """Test PR comment command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "pr", "comment",
            "--repo", "owner/repo",
            "--pr", "123",
            "--body", "LGTM!",
        ])

        assert args.command == "pr"
        assert args.action == "comment"
        assert args.pr == 123
        assert args.body == "LGTM!"

    def test_release_create_command(self):
        """Test release create command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "release", "create",
            "--repo", "owner/repo",
            "--tag", "v1.0.0",
            "--name", "Version 1.0.0",
            "--notes", "Release notes",
        ])

        assert args.command == "release"
        assert args.action == "create"
        assert args.tag == "v1.0.0"
        assert args.name == "Version 1.0.0"
        assert args.notes == "Release notes"

    def test_release_create_flags(self):
        """Test release create command with flags."""
        parser = create_parser()
        args = parser.parse_args([
            "release", "create",
            "--repo", "owner/repo",
            "--tag", "v1.0.0",
            "--draft",
            "--prerelease",
            "--generate-notes",
        ])

        assert args.draft is True
        assert args.prerelease is True
        assert args.generate_notes is True

    def test_release_list_command(self):
        """Test release list command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "release", "list",
            "--repo", "owner/repo",
        ])

        assert args.command == "release"
        assert args.action == "list"

    def test_dispatch_command(self):
        """Test dispatch command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "dispatch",
            "--repo", "owner/repo",
            "--event-type", "deploy",
        ])

        assert args.command == "dispatch"
        assert args.event_type == "deploy"

    def test_dispatch_with_payload(self):
        """Test dispatch command with payload."""
        parser = create_parser()
        payload = '{"env": "prod"}'
        args = parser.parse_args([
            "dispatch",
            "--repo", "owner/repo",
            "--event-type", "deploy",
            "--payload", payload,
        ])

        assert args.payload == payload

    def test_short_options(self):
        """Test short option aliases."""
        parser = create_parser()
        args = parser.parse_args([
            "issue", "create",
            "-r", "owner/repo",
            "-t", "Title",
            "-b", "Body",
        ])

        assert args.repo == "owner/repo"
        assert args.title == "Title"
        assert args.body == "Body"


# =============================================================================
# Main Function Integration Tests
# =============================================================================


class TestMainFunction:
    """Integration tests for the main function."""

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_main_issue_list(self, mock_client_class, sample_issue):
        """Test main function with issue list command."""
        mock_client = MagicMock()
        mock_client.list_issues.return_value = [sample_issue]
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", ["github_ops", "issue", "list", "--repo", "owner/repo"]):
            result = main()

        assert result == EXIT_SUCCESS
        mock_client.list_issues.assert_called_once()

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_main_issue_get(self, mock_client_class, sample_issue):
        """Test main function with issue get command."""
        mock_client = MagicMock()
        mock_client.get_issue.return_value = sample_issue
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", ["github_ops", "issue", "get", "--repo", "owner/repo", "--number", "42"]):
            result = main()

        assert result == EXIT_SUCCESS
        mock_client.get_issue.assert_called_once_with(repo="owner/repo", issue_number=42)

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_main_issue_create(self, mock_client_class, sample_issue):
        """Test main function with issue create command."""
        mock_client = MagicMock()
        mock_client.create_issue.return_value = sample_issue
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "issue", "create",
            "--repo", "owner/repo",
            "--title", "Test Issue",
        ]):
            result = main()

        assert result == EXIT_SUCCESS
        mock_client.create_issue.assert_called_once()

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_main_issue_update(self, mock_client_class, sample_issue):
        """Test main function with issue update command."""
        mock_client = MagicMock()
        mock_client.update_issue.return_value = sample_issue
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "issue", "update",
            "--repo", "owner/repo",
            "--number", "42",
            "--state", "closed",
        ]):
            result = main()

        assert result == EXIT_SUCCESS

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_main_pr_comment(self, mock_client_class, sample_comment):
        """Test main function with PR comment command."""
        mock_client = MagicMock()
        mock_client.create_pr_comment.return_value = sample_comment
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "pr", "comment",
            "--repo", "owner/repo",
            "--pr", "123",
            "--body", "LGTM!",
        ]):
            result = main()

        assert result == EXIT_SUCCESS
        mock_client.create_pr_comment.assert_called_once()

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_main_release_create(self, mock_client_class, sample_release):
        """Test main function with release create command."""
        mock_client = MagicMock()
        mock_client.create_release.return_value = sample_release
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "release", "create",
            "--repo", "owner/repo",
            "--tag", "v1.0.0",
        ]):
            result = main()

        assert result == EXIT_SUCCESS

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_main_release_list(self, mock_client_class, sample_release):
        """Test main function with release list command."""
        mock_client = MagicMock()
        mock_client.list_releases.return_value = [sample_release]
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "release", "list",
            "--repo", "owner/repo",
        ]):
            result = main()

        assert result == EXIT_SUCCESS

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_main_dispatch(self, mock_client_class):
        """Test main function with dispatch command."""
        mock_client = MagicMock()
        mock_client.create_dispatch_event.return_value = {}
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "dispatch",
            "--repo", "owner/repo",
            "--event-type", "deploy",
        ]):
            result = main()

        assert result == EXIT_SUCCESS

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_main_dispatch_with_payload(self, mock_client_class):
        """Test main function with dispatch command and payload."""
        mock_client = MagicMock()
        mock_client.create_dispatch_event.return_value = {}
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "dispatch",
            "--repo", "owner/repo",
            "--event-type", "deploy",
            "--payload", '{"env": "prod"}',
        ]):
            result = main()

        assert result == EXIT_SUCCESS
        mock_client.create_dispatch_event.assert_called_once()
        call_args = mock_client.create_dispatch_event.call_args
        assert call_args[1]["client_payload"] == {"env": "prod"}

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    def test_main_dispatch_invalid_payload(self):
        """Test main function with invalid JSON payload."""
        with patch.object(sys, "argv", [
            "github_ops", "dispatch",
            "--repo", "owner/repo",
            "--event-type", "deploy",
            "--payload", "invalid json",
        ]):
            result = main()

        assert result == EXIT_INVALID_ARGS

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    def test_main_no_command(self):
        """Test main function with no command shows help."""
        with patch.object(sys, "argv", ["github_ops"]):
            result = main()

        assert result == EXIT_INVALID_ARGS


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_main_api_404_error(self, mock_client_class):
        """Test main function handles 404 errors."""
        mock_client = MagicMock()
        mock_client.get_issue.side_effect = GitHubAPIError("Not found", 404)
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "issue", "get",
            "--repo", "owner/repo",
            "--number", "999",
        ]):
            result = main()

        assert result == EXIT_NOT_FOUND

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_main_api_generic_error(self, mock_client_class):
        """Test main function handles generic API errors."""
        mock_client = MagicMock()
        mock_client.list_issues.side_effect = GitHubAPIError("Server error", 500)
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "issue", "list",
            "--repo", "owner/repo",
        ]):
            result = main()

        assert result == EXIT_API_ERROR

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_main_value_error(self, mock_client_class):
        """Test main function handles ValueError."""
        mock_client = MagicMock()
        mock_client.update_issue.side_effect = ValueError("Invalid input")
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "issue", "update",
            "--repo", "owner/repo",
            "--number", "42",
            "--title", "New Title",
        ]):
            result = main()

        assert result == EXIT_INVALID_ARGS

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_main_network_error(self, mock_client_class):
        """Test main function handles network errors."""
        import requests

        mock_client = MagicMock()
        mock_client.list_issues.side_effect = requests.RequestException("Network error")
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "issue", "list",
            "--repo", "owner/repo",
        ]):
            result = main()

        assert result == EXIT_API_ERROR


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_empty_issues_list(self, mock_client_class):
        """Test handling empty issues list."""
        mock_client = MagicMock()
        mock_client.list_issues.return_value = []
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "issue", "list",
            "--repo", "owner/repo",
        ]):
            result = main()

        assert result == EXIT_SUCCESS

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_empty_releases_list(self, mock_client_class):
        """Test handling empty releases list."""
        mock_client = MagicMock()
        mock_client.list_releases.return_value = []
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "release", "list",
            "--repo", "owner/repo",
        ]):
            result = main()

        assert result == EXIT_SUCCESS

    def test_format_issue_no_assignees(self, sample_issue):
        """Test formatting issue without assignees."""
        sample_issue["assignees"] = []
        output = format_issue(sample_issue)

        assert "Assignees:" not in output

    def test_format_release_no_name(self, sample_release):
        """Test formatting release without name."""
        sample_release["name"] = None
        output = format_release(sample_release)

        assert "(no name)" in output

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("github_ops.GitHubOps")
    def test_issue_create_with_labels_string(self, mock_client_class, sample_issue):
        """Test issue create parses comma-separated labels."""
        mock_client = MagicMock()
        mock_client.create_issue.return_value = sample_issue
        mock_client_class.return_value = mock_client

        with patch.object(sys, "argv", [
            "github_ops", "issue", "create",
            "--repo", "owner/repo",
            "--title", "Test",
            "--labels", "bug,feature,urgent",
        ]):
            main()

        call_args = mock_client.create_issue.call_args
        assert call_args[1]["labels"] == ["bug", "feature", "urgent"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
