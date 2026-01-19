#!/usr/bin/env python3
"""
Comprehensive tests for git_changelog.py

Tests cover:
- Conventional commit parsing
- Commit retrieval with various options
- Grouping functionality
- Output formatting (Markdown, JSON, plain text)
- Error handling and exit codes
- CLI argument parsing and validation
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, Mock, patch, PropertyMock

# pytest is optional - tests can run with standard unittest
try:
    import pytest
except ImportError:
    pytest = None

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from git_changelog import (
    GitChangelog,
    ParsedCommit,
    OutputFormat,
    GroupBy,
    COMMIT_TYPES,
    CONVENTIONAL_COMMIT_PATTERN,
    EXIT_SUCCESS,
    EXIT_NOT_GIT_REPO,
    EXIT_INVALID_RANGE,
    EXIT_GIT_ERROR,
    EXIT_INVALID_ARGS,
    create_parser,
    validate_args,
    main,
)


class TestConventionalCommitPattern(unittest.TestCase):
    """Tests for the conventional commit regex pattern."""

    def test_basic_commit(self):
        """Test parsing a basic conventional commit."""
        message = "feat: add new feature"
        match = CONVENTIONAL_COMMIT_PATTERN.match(message)
        self.assertIsNotNone(match)
        self.assertEqual(match.group('type'), 'feat')
        self.assertIsNone(match.group('scope'))
        self.assertEqual(match.group('description'), 'add new feature')
        self.assertIsNone(match.group('breaking'))

    def test_commit_with_scope(self):
        """Test parsing a commit with scope."""
        message = "fix(auth): resolve login issue"
        match = CONVENTIONAL_COMMIT_PATTERN.match(message)
        self.assertIsNotNone(match)
        self.assertEqual(match.group('type'), 'fix')
        self.assertEqual(match.group('scope'), 'auth')
        self.assertEqual(match.group('description'), 'resolve login issue')

    def test_breaking_change(self):
        """Test parsing a breaking change commit."""
        message = "feat(api)!: remove deprecated endpoints"
        match = CONVENTIONAL_COMMIT_PATTERN.match(message)
        self.assertIsNotNone(match)
        self.assertEqual(match.group('type'), 'feat')
        self.assertEqual(match.group('scope'), 'api')
        self.assertEqual(match.group('breaking'), '!')
        self.assertEqual(match.group('description'), 'remove deprecated endpoints')

    def test_all_commit_types(self):
        """Test all standard conventional commit types."""
        for commit_type in COMMIT_TYPES.keys():
            message = f"{commit_type}: test message"
            match = CONVENTIONAL_COMMIT_PATTERN.match(message)
            self.assertIsNotNone(match, f"Failed to match type: {commit_type}")
            self.assertEqual(match.group('type'), commit_type)

    def test_non_conventional_commit(self):
        """Test that non-conventional commits don't match."""
        messages = [
            "Update README",
            "Fixed bug",
            "WIP",
            "merge branch main",
        ]
        for message in messages:
            match = CONVENTIONAL_COMMIT_PATTERN.match(message)
            self.assertIsNone(match, f"Should not match: {message}")

    def test_complex_scope(self):
        """Test commit with complex scope containing special characters."""
        message = "feat(user-auth): add OAuth2 support"
        match = CONVENTIONAL_COMMIT_PATTERN.match(message)
        self.assertIsNotNone(match)
        self.assertEqual(match.group('scope'), 'user-auth')


class TestParsedCommit(unittest.TestCase):
    """Tests for the ParsedCommit dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.commit = ParsedCommit(
            sha='abc1234567890',
            short_sha='abc1234',
            type='feat',
            scope='api',
            description='add new endpoint',
            breaking=False,
            author='Test Author',
            author_email='test@example.com',
            date=datetime(2024, 1, 15, 10, 30, 0),
            full_message='feat(api): add new endpoint\n\nThis adds a new endpoint.',
            is_conventional=True,
        )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = self.commit.to_dict()
        self.assertEqual(result['sha'], 'abc1234567890')
        self.assertEqual(result['type'], 'feat')
        self.assertEqual(result['scope'], 'api')
        self.assertEqual(result['author'], 'Test Author')
        self.assertIn('2024-01-15', result['date'])

    def test_to_dict_with_none_scope(self):
        """Test conversion when scope is None."""
        commit = ParsedCommit(
            sha='abc1234567890',
            short_sha='abc1234',
            type='fix',
            scope=None,
            description='fix bug',
            breaking=False,
            author='Test Author',
            author_email='test@example.com',
            date=datetime(2024, 1, 15),
            full_message='fix: fix bug',
            is_conventional=True,
        )
        result = commit.to_dict()
        self.assertIsNone(result['scope'])


class TestGitChangelogInit(unittest.TestCase):
    """Tests for GitChangelog initialization."""

    @patch('git_changelog.git.Repo')
    def test_init_valid_repo(self, mock_repo):
        """Test initialization with a valid repository."""
        mock_repo.return_value = MagicMock()
        changelog = GitChangelog('/valid/path')
        self.assertIsNotNone(changelog.repo)

    @patch('git_changelog.git.Repo')
    def test_init_invalid_repo(self, mock_repo):
        """Test initialization with an invalid repository."""
        from git.exc import InvalidGitRepositoryError
        mock_repo.side_effect = InvalidGitRepositoryError('/invalid/path')

        with self.assertRaises(SystemExit) as context:
            GitChangelog('/invalid/path')
        self.assertEqual(context.exception.code, EXIT_NOT_GIT_REPO)


class TestGitChangelogParseCommit(unittest.TestCase):
    """Tests for commit parsing."""

    @patch('git_changelog.git.Repo')
    def setUp(self, mock_repo):
        """Set up test fixtures."""
        mock_repo.return_value = MagicMock()
        self.changelog = GitChangelog('.')

    def _create_mock_commit(self, message, sha='abc1234567890',
                            author_name='Test Author',
                            author_email='test@example.com',
                            committed_date=1705312200):
        """Helper to create a mock commit."""
        mock_commit = MagicMock()
        mock_commit.message = message
        mock_commit.hexsha = sha
        mock_commit.author.name = author_name
        mock_commit.author.email = author_email
        mock_commit.committed_date = committed_date
        return mock_commit

    def test_parse_conventional_commit(self):
        """Test parsing a conventional commit."""
        mock_commit = self._create_mock_commit('feat(api): add new endpoint')
        result = self.changelog.parse_commit(mock_commit)

        self.assertTrue(result.is_conventional)
        self.assertEqual(result.type, 'feat')
        self.assertEqual(result.scope, 'api')
        self.assertEqual(result.description, 'add new endpoint')

    def test_parse_breaking_commit(self):
        """Test parsing a breaking change commit."""
        mock_commit = self._create_mock_commit('feat!: breaking change')
        result = self.changelog.parse_commit(mock_commit)

        self.assertTrue(result.breaking)
        self.assertTrue(result.is_conventional)

    def test_parse_non_conventional_commit(self):
        """Test parsing a non-conventional commit."""
        mock_commit = self._create_mock_commit('Update documentation')
        result = self.changelog.parse_commit(mock_commit)

        self.assertFalse(result.is_conventional)
        self.assertEqual(result.type, 'other')
        self.assertEqual(result.description, 'Update documentation')

    def test_parse_multiline_commit(self):
        """Test parsing a commit with multiple lines."""
        message = "fix(core): resolve memory leak\n\nThis fixes a critical memory leak."
        mock_commit = self._create_mock_commit(message)
        result = self.changelog.parse_commit(mock_commit)

        self.assertTrue(result.is_conventional)
        self.assertEqual(result.description, 'resolve memory leak')
        self.assertIn('critical memory leak', result.full_message)


class TestGitChangelogGetCommits(unittest.TestCase):
    """Tests for commit retrieval."""

    @patch('git_changelog.git.Repo')
    def setUp(self, mock_repo_class):
        """Set up test fixtures."""
        self.mock_repo = MagicMock()
        mock_repo_class.return_value = self.mock_repo
        self.changelog = GitChangelog('.')

    def _create_mock_commits(self, count=3):
        """Helper to create mock commits."""
        commits = []
        for i in range(count):
            mock_commit = MagicMock()
            mock_commit.message = f'feat: feature {i}'
            mock_commit.hexsha = f'abc{i}234567890'
            mock_commit.author.name = f'Author {i}'
            mock_commit.author.email = f'author{i}@example.com'
            mock_commit.committed_date = 1705312200 + i * 3600
            commits.append(mock_commit)
        return commits

    def test_get_commits_with_range(self):
        """Test getting commits with from/to range."""
        mock_commits = self._create_mock_commits(5)
        self.mock_repo.iter_commits.return_value = mock_commits

        result = self.changelog.get_commits(from_ref='v1.0.0', to_ref='v1.1.0')

        self.assertEqual(len(result), 5)
        self.mock_repo.iter_commits.assert_called_once_with('v1.0.0..v1.1.0')

    def test_get_commits_with_since(self):
        """Test getting commits with since filter."""
        mock_commits = self._create_mock_commits(3)
        self.mock_repo.iter_commits.return_value = mock_commits

        result = self.changelog.get_commits(since='2 weeks ago')

        self.assertEqual(len(result), 3)
        self.mock_repo.iter_commits.assert_called_once_with(since='2 weeks ago')

    def test_get_all_commits(self):
        """Test getting all commits."""
        mock_commits = self._create_mock_commits(10)
        self.mock_repo.iter_commits.return_value = mock_commits

        result = self.changelog.get_commits(all_commits=True)

        self.assertEqual(len(result), 10)
        self.mock_repo.iter_commits.assert_called_once_with()

    def test_get_commits_default(self):
        """Test getting commits with no arguments (default)."""
        mock_commits = self._create_mock_commits(10)
        self.mock_repo.iter_commits.return_value = mock_commits

        result = self.changelog.get_commits()

        self.mock_repo.iter_commits.assert_called_once_with(max_count=10)

    def test_get_commits_invalid_range(self):
        """Test error handling for invalid range."""
        from git.exc import BadName
        self.mock_repo.iter_commits.side_effect = BadName('invalid-ref')

        with self.assertRaises(SystemExit) as context:
            self.changelog.get_commits(from_ref='invalid-ref')
        self.assertEqual(context.exception.code, EXIT_INVALID_RANGE)

    def test_get_commits_git_error(self):
        """Test error handling for git errors."""
        from git.exc import GitCommandError
        self.mock_repo.iter_commits.side_effect = GitCommandError('git', 'error')

        with self.assertRaises(SystemExit) as context:
            self.changelog.get_commits(from_ref='v1.0.0')
        self.assertEqual(context.exception.code, EXIT_GIT_ERROR)


class TestGitChangelogGroupCommits(unittest.TestCase):
    """Tests for commit grouping."""

    @patch('git_changelog.git.Repo')
    def setUp(self, mock_repo):
        """Set up test fixtures."""
        mock_repo.return_value = MagicMock()
        self.changelog = GitChangelog('.')

        self.commits = [
            ParsedCommit(
                sha='abc1', short_sha='abc1', type='feat', scope='api',
                description='add endpoint', breaking=False,
                author='Alice', author_email='alice@example.com',
                date=datetime(2024, 1, 15), full_message='feat(api): add endpoint',
                is_conventional=True
            ),
            ParsedCommit(
                sha='abc2', short_sha='abc2', type='fix', scope='api',
                description='fix bug', breaking=False,
                author='Bob', author_email='bob@example.com',
                date=datetime(2024, 1, 16), full_message='fix(api): fix bug',
                is_conventional=True
            ),
            ParsedCommit(
                sha='abc3', short_sha='abc3', type='feat', scope='ui',
                description='add button', breaking=False,
                author='Alice', author_email='alice@example.com',
                date=datetime(2024, 1, 17), full_message='feat(ui): add button',
                is_conventional=True
            ),
        ]

    def test_group_by_none(self):
        """Test no grouping."""
        result = self.changelog.group_commits(self.commits, GroupBy.NONE)
        self.assertEqual(len(result), 1)
        self.assertIn('All Commits', result)
        self.assertEqual(len(result['All Commits']), 3)

    def test_group_by_type(self):
        """Test grouping by commit type."""
        result = self.changelog.group_commits(self.commits, GroupBy.TYPE)
        self.assertIn('Features', result)
        self.assertIn('Bug Fixes', result)
        self.assertEqual(len(result['Features']), 2)
        self.assertEqual(len(result['Bug Fixes']), 1)

    def test_group_by_author(self):
        """Test grouping by author."""
        result = self.changelog.group_commits(self.commits, GroupBy.AUTHOR)
        self.assertIn('Alice', result)
        self.assertIn('Bob', result)
        self.assertEqual(len(result['Alice']), 2)
        self.assertEqual(len(result['Bob']), 1)

    def test_group_by_scope(self):
        """Test grouping by scope."""
        result = self.changelog.group_commits(self.commits, GroupBy.SCOPE)
        self.assertIn('api', result)
        self.assertIn('ui', result)
        self.assertEqual(len(result['api']), 2)
        self.assertEqual(len(result['ui']), 1)

    def test_group_by_scope_with_no_scope(self):
        """Test grouping with commits that have no scope."""
        commits = self.commits + [
            ParsedCommit(
                sha='abc4', short_sha='abc4', type='docs', scope=None,
                description='update readme', breaking=False,
                author='Alice', author_email='alice@example.com',
                date=datetime(2024, 1, 18), full_message='docs: update readme',
                is_conventional=True
            ),
        ]
        result = self.changelog.group_commits(commits, GroupBy.SCOPE)
        self.assertIn('No Scope', result)


class TestGitChangelogFormatting(unittest.TestCase):
    """Tests for output formatting."""

    @patch('git_changelog.git.Repo')
    def setUp(self, mock_repo):
        """Set up test fixtures."""
        mock_repo.return_value = MagicMock()
        self.changelog = GitChangelog('.')

        self.commits = [
            ParsedCommit(
                sha='abc1234567890', short_sha='abc1234', type='feat', scope='api',
                description='add new endpoint', breaking=False,
                author='Test Author', author_email='test@example.com',
                date=datetime(2024, 1, 15, 10, 30),
                full_message='feat(api): add new endpoint',
                is_conventional=True
            ),
            ParsedCommit(
                sha='def1234567890', short_sha='def1234', type='fix', scope=None,
                description='fix critical bug', breaking=True,
                author='Test Author', author_email='test@example.com',
                date=datetime(2024, 1, 16, 11, 30),
                full_message='fix!: fix critical bug',
                is_conventional=True
            ),
        ]
        self.grouped = {'All Commits': self.commits}

    def test_format_markdown(self):
        """Test Markdown output format."""
        result = self.changelog.format_markdown(self.grouped, title='Test Changelog')

        self.assertIn('# Test Changelog', result)
        self.assertIn('## All Commits', result)
        self.assertIn('**api**:', result)
        self.assertIn('add new endpoint', result)
        self.assertIn('[abc1234]', result)
        self.assertIn('**BREAKING**', result)

    def test_format_markdown_no_title(self):
        """Test Markdown output without title."""
        result = self.changelog.format_markdown(self.grouped)

        # First line should be a section header (##), not a title (#)
        first_line = result.split('\n')[0]
        self.assertTrue(first_line.startswith('## '), "Should start with section header, not title")
        self.assertIn('## All Commits', result)

    def test_format_plain(self):
        """Test plain text output format."""
        result = self.changelog.format_plain(self.grouped, title='Test Changelog')

        self.assertIn('Test Changelog', result)
        self.assertIn('All Commits', result)
        self.assertIn('[api]', result)
        self.assertIn('(abc1234)', result)
        self.assertIn('[BREAKING]', result)

    def test_format_json(self):
        """Test JSON output format."""
        result = self.changelog.format_json(self.grouped, title='Test Changelog')

        data = json.loads(result)
        self.assertEqual(data['title'], 'Test Changelog')
        self.assertEqual(data['total_commits'], 2)
        self.assertIn('groups', data)
        self.assertIn('All Commits', data['groups'])
        self.assertEqual(len(data['groups']['All Commits']), 2)

    def test_format_json_structure(self):
        """Test JSON output structure."""
        result = self.changelog.format_json(self.grouped)

        data = json.loads(result)
        commit = data['groups']['All Commits'][0]

        required_fields = ['sha', 'short_sha', 'type', 'scope', 'description',
                          'breaking', 'author', 'author_email', 'date',
                          'full_message', 'is_conventional']
        for field in required_fields:
            self.assertIn(field, commit)


class TestGitChangelogGenerate(unittest.TestCase):
    """Tests for the generate method."""

    @patch('git_changelog.git.Repo')
    def setUp(self, mock_repo_class):
        """Set up test fixtures."""
        self.mock_repo = MagicMock()
        mock_repo_class.return_value = self.mock_repo
        self.changelog = GitChangelog('.')

    def _setup_mock_commits(self):
        """Helper to set up mock commits."""
        mock_commit = MagicMock()
        mock_commit.message = 'feat(api): add feature'
        mock_commit.hexsha = 'abc1234567890'
        mock_commit.author.name = 'Test Author'
        mock_commit.author.email = 'test@example.com'
        mock_commit.committed_date = 1705312200
        self.mock_repo.iter_commits.return_value = [mock_commit]

    def test_generate_markdown(self):
        """Test generating Markdown changelog."""
        self._setup_mock_commits()

        result = self.changelog.generate(
            from_ref='v1.0.0',
            output_format=OutputFormat.MARKDOWN
        )

        self.assertIn('##', result)
        self.assertIn('add feature', result)

    def test_generate_json(self):
        """Test generating JSON changelog."""
        self._setup_mock_commits()

        result = self.changelog.generate(
            from_ref='v1.0.0',
            output_format=OutputFormat.JSON
        )

        data = json.loads(result)
        self.assertIn('groups', data)

    def test_generate_plain(self):
        """Test generating plain text changelog."""
        self._setup_mock_commits()

        result = self.changelog.generate(
            from_ref='v1.0.0',
            output_format=OutputFormat.PLAIN
        )

        self.assertIn('add feature', result)
        self.assertNotIn('##', result)

    def test_generate_no_commits(self):
        """Test generating changelog with no commits."""
        self.mock_repo.iter_commits.return_value = []

        result = self.changelog.generate(from_ref='v1.0.0')

        self.assertIn('No commits found', result)

    def test_generate_grouped_by_type(self):
        """Test generating grouped changelog."""
        self._setup_mock_commits()

        result = self.changelog.generate(
            from_ref='v1.0.0',
            group_by=GroupBy.TYPE
        )

        self.assertIn('Features', result)


class TestCLIArgumentParser(unittest.TestCase):
    """Tests for CLI argument parsing."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = create_parser()

    def test_parse_basic_args(self):
        """Test parsing basic arguments."""
        args = self.parser.parse_args(['--from', 'v1.0.0', '--to', 'v1.1.0'])
        self.assertEqual(args.from_ref, 'v1.0.0')
        self.assertEqual(args.to_ref, 'v1.1.0')

    def test_parse_format_option(self):
        """Test parsing format option."""
        args = self.parser.parse_args(['--format', 'json'])
        self.assertEqual(args.output_format, 'json')

    def test_parse_group_by_option(self):
        """Test parsing group-by option."""
        args = self.parser.parse_args(['--group-by', 'type'])
        self.assertEqual(args.group_by, 'type')

    def test_parse_since_option(self):
        """Test parsing since option."""
        args = self.parser.parse_args(['--since', '2 weeks ago'])
        self.assertEqual(args.since, '2 weeks ago')

    def test_parse_all_option(self):
        """Test parsing all option."""
        args = self.parser.parse_args(['--all'])
        self.assertTrue(args.all_commits)

    def test_parse_short_options(self):
        """Test parsing short options."""
        args = self.parser.parse_args(['-f', 'v1.0.0', '-t', 'v1.1.0', '-F', 'json'])
        self.assertEqual(args.from_ref, 'v1.0.0')
        self.assertEqual(args.to_ref, 'v1.1.0')
        self.assertEqual(args.output_format, 'json')

    def test_parse_title_option(self):
        """Test parsing title option."""
        args = self.parser.parse_args(['--title', 'My Changelog'])
        self.assertEqual(args.title, 'My Changelog')

    def test_parse_output_option(self):
        """Test parsing output option."""
        args = self.parser.parse_args(['--output', 'CHANGELOG.md'])
        self.assertEqual(args.output, 'CHANGELOG.md')

    def test_default_values(self):
        """Test default argument values."""
        args = self.parser.parse_args([])
        self.assertIsNone(args.from_ref)
        self.assertEqual(args.to_ref, 'HEAD')
        self.assertEqual(args.output_format, 'markdown')
        self.assertEqual(args.group_by, 'none')
        self.assertEqual(args.repo_path, '.')
        self.assertFalse(args.all_commits)


class TestCLIArgumentValidation(unittest.TestCase):
    """Tests for CLI argument validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = create_parser()

    def test_validate_since_with_from(self):
        """Test validation rejects --since with --from."""
        args = self.parser.parse_args(['--since', '2 weeks ago', '--from', 'v1.0.0'])

        with self.assertRaises(SystemExit) as context:
            validate_args(args)
        self.assertEqual(context.exception.code, EXIT_INVALID_ARGS)

    def test_validate_all_with_from(self):
        """Test validation rejects --all with --from."""
        args = self.parser.parse_args(['--all', '--from', 'v1.0.0'])

        with self.assertRaises(SystemExit) as context:
            validate_args(args)
        self.assertEqual(context.exception.code, EXIT_INVALID_ARGS)

    def test_validate_all_with_since(self):
        """Test validation rejects --all with --since."""
        args = self.parser.parse_args(['--all', '--since', '2 weeks ago'])

        with self.assertRaises(SystemExit) as context:
            validate_args(args)
        self.assertEqual(context.exception.code, EXIT_INVALID_ARGS)

    def test_validate_valid_args(self):
        """Test validation passes for valid arguments."""
        args = self.parser.parse_args(['--from', 'v1.0.0'])
        # Should not raise
        validate_args(args)


class TestMainFunction(unittest.TestCase):
    """Tests for the main function."""

    @patch('git_changelog.GitChangelog')
    @patch('sys.argv', ['git_changelog', '--from', 'v1.0.0'])
    def test_main_success(self, mock_changelog_class):
        """Test main function success."""
        mock_changelog = MagicMock()
        mock_changelog.generate.return_value = '# Changelog\n\n- Feature'
        mock_changelog_class.return_value = mock_changelog

        with patch('builtins.print'):
            result = main()

        self.assertEqual(result, EXIT_SUCCESS)

    @patch('git_changelog.GitChangelog')
    @patch('sys.argv', ['git_changelog', '--from', 'v1.0.0', '--output', '/tmp/test_changelog.md'])
    def test_main_with_output_file(self, mock_changelog_class):
        """Test main function with output file."""
        mock_changelog = MagicMock()
        mock_changelog.generate.return_value = '# Changelog'
        mock_changelog_class.return_value = mock_changelog

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_file = f.name

        try:
            with patch('sys.argv', ['git_changelog', '--from', 'v1.0.0', '--output', temp_file]):
                result = main()

            self.assertEqual(result, EXIT_SUCCESS)
            with open(temp_file, 'r') as f:
                content = f.read()
            self.assertEqual(content, '# Changelog')
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch('sys.argv', ['git_changelog', '--since', '2 weeks ago', '--from', 'v1.0.0'])
    def test_main_invalid_args(self):
        """Test main function with invalid arguments."""
        with self.assertRaises(SystemExit) as context:
            main()
        self.assertEqual(context.exception.code, EXIT_INVALID_ARGS)


class TestOutputFormats(unittest.TestCase):
    """Test OutputFormat enum."""

    def test_output_format_values(self):
        """Test OutputFormat enum has correct values."""
        self.assertEqual(OutputFormat.MARKDOWN.value, 'markdown')
        self.assertEqual(OutputFormat.JSON.value, 'json')
        self.assertEqual(OutputFormat.PLAIN.value, 'plain')


class TestGroupBy(unittest.TestCase):
    """Test GroupBy enum."""

    def test_group_by_values(self):
        """Test GroupBy enum has correct values."""
        self.assertEqual(GroupBy.NONE.value, 'none')
        self.assertEqual(GroupBy.TYPE.value, 'type')
        self.assertEqual(GroupBy.AUTHOR.value, 'author')
        self.assertEqual(GroupBy.SCOPE.value, 'scope')


class TestCommitTypes(unittest.TestCase):
    """Test COMMIT_TYPES dictionary."""

    def test_standard_types_present(self):
        """Test standard conventional commit types are present."""
        expected_types = ['feat', 'fix', 'docs', 'style', 'refactor',
                         'perf', 'test', 'chore', 'build', 'ci', 'revert']
        for commit_type in expected_types:
            self.assertIn(commit_type, COMMIT_TYPES)

    def test_types_have_descriptions(self):
        """Test all types have human-readable descriptions."""
        for type_key, description in COMMIT_TYPES.items():
            self.assertIsInstance(description, str)
            self.assertGreater(len(description), 0)


class TestExitCodes(unittest.TestCase):
    """Test exit codes."""

    def test_exit_codes_unique(self):
        """Test exit codes are unique."""
        codes = [EXIT_SUCCESS, EXIT_NOT_GIT_REPO, EXIT_INVALID_RANGE,
                EXIT_GIT_ERROR, EXIT_INVALID_ARGS]
        self.assertEqual(len(codes), len(set(codes)))

    def test_exit_code_values(self):
        """Test exit code values."""
        self.assertEqual(EXIT_SUCCESS, 0)
        self.assertEqual(EXIT_NOT_GIT_REPO, 1)
        self.assertEqual(EXIT_INVALID_RANGE, 2)
        self.assertEqual(EXIT_GIT_ERROR, 3)
        self.assertEqual(EXIT_INVALID_ARGS, 4)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    @patch('git_changelog.git.Repo')
    def test_empty_commit_message(self, mock_repo):
        """Test handling of empty commit message."""
        mock_repo.return_value = MagicMock()
        changelog = GitChangelog('.')

        mock_commit = MagicMock()
        mock_commit.message = ''
        mock_commit.hexsha = 'abc1234567890'
        mock_commit.author.name = 'Test'
        mock_commit.author.email = 'test@example.com'
        mock_commit.committed_date = 1705312200

        result = changelog.parse_commit(mock_commit)
        self.assertFalse(result.is_conventional)
        self.assertEqual(result.type, 'other')

    @patch('git_changelog.git.Repo')
    def test_unicode_in_commit(self, mock_repo):
        """Test handling of unicode in commit messages."""
        mock_repo.return_value = MagicMock()
        changelog = GitChangelog('.')

        mock_commit = MagicMock()
        mock_commit.message = 'feat: add unicode support'
        mock_commit.hexsha = 'abc1234567890'
        mock_commit.author.name = 'Test'
        mock_commit.author.email = 'test@example.com'
        mock_commit.committed_date = 1705312200

        result = changelog.parse_commit(mock_commit)
        self.assertTrue(result.is_conventional)
        self.assertIn('unicode', result.description)

    @patch('git_changelog.git.Repo')
    def test_very_long_commit_message(self, mock_repo):
        """Test handling of very long commit messages."""
        mock_repo.return_value = MagicMock()
        changelog = GitChangelog('.')

        long_description = 'x' * 1000
        mock_commit = MagicMock()
        mock_commit.message = f'feat: {long_description}'
        mock_commit.hexsha = 'abc1234567890'
        mock_commit.author.name = 'Test'
        mock_commit.author.email = 'test@example.com'
        mock_commit.committed_date = 1705312200

        result = changelog.parse_commit(mock_commit)
        self.assertTrue(result.is_conventional)
        self.assertEqual(len(result.description), 1000)


if __name__ == '__main__':
    unittest.main()
