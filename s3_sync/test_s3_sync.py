#!/usr/bin/env python3
"""
Comprehensive tests for s3_sync CLI tool.

Tests cover:
- All three cloud providers (S3, GCS, Azure)
- All commands (upload, download, presign, sync)
- Error handling and exit codes
- Glob pattern matching
- Argument parsing
"""

import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import s3_sync
from s3_sync import (
    EXIT_SUCCESS,
    EXIT_CREDENTIALS_ERROR,
    EXIT_NOT_FOUND,
    EXIT_TRANSFER_ERROR,
    EXIT_INVALID_ARGS,
    CloudStorageError,
    CredentialsError,
    NotFoundError,
    TransferError,
    S3Provider,
    GCSProvider,
    AzureProvider,
    get_provider,
    parse_cloud_url,
    matches_glob,
    create_parser,
    main,
    cmd_upload,
    cmd_download,
    cmd_presign,
    cmd_sync,
)


class TestExitCodes(unittest.TestCase):
    """Test that exit codes are correctly defined."""

    def test_exit_codes_values(self):
        """Verify exit code values match specification."""
        self.assertEqual(EXIT_SUCCESS, 0)
        self.assertEqual(EXIT_CREDENTIALS_ERROR, 1)
        self.assertEqual(EXIT_NOT_FOUND, 2)
        self.assertEqual(EXIT_TRANSFER_ERROR, 3)
        self.assertEqual(EXIT_INVALID_ARGS, 4)


class TestExceptions(unittest.TestCase):
    """Test custom exception classes."""

    def test_cloud_storage_error_base(self):
        """CloudStorageError is the base exception."""
        self.assertTrue(issubclass(CredentialsError, CloudStorageError))
        self.assertTrue(issubclass(NotFoundError, CloudStorageError))
        self.assertTrue(issubclass(TransferError, CloudStorageError))

    def test_exception_messages(self):
        """Exceptions preserve messages."""
        msg = "Test error message"
        self.assertEqual(str(CredentialsError(msg)), msg)
        self.assertEqual(str(NotFoundError(msg)), msg)
        self.assertEqual(str(TransferError(msg)), msg)


class TestParseCloudUrl(unittest.TestCase):
    """Test URL parsing functions."""

    def test_parse_s3_url(self):
        """Parse S3 URL correctly."""
        scheme, bucket, path = parse_cloud_url("s3://my-bucket/path/to/file.txt")
        self.assertEqual(scheme, "s3")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(path, "path/to/file.txt")

    def test_parse_gcs_url(self):
        """Parse GCS URL correctly."""
        scheme, bucket, path = parse_cloud_url("gs://my-bucket/data/file.csv")
        self.assertEqual(scheme, "gs")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(path, "data/file.csv")

    def test_parse_azure_url(self):
        """Parse Azure URL correctly."""
        scheme, bucket, path = parse_cloud_url("az://container/blob/path.json")
        self.assertEqual(scheme, "az")
        self.assertEqual(bucket, "container")
        self.assertEqual(path, "blob/path.json")

    def test_parse_url_no_path(self):
        """Parse URL with no path."""
        scheme, bucket, path = parse_cloud_url("s3://bucket/")
        self.assertEqual(scheme, "s3")
        self.assertEqual(bucket, "bucket")
        self.assertEqual(path, "")

    def test_parse_url_root_only(self):
        """Parse URL with bucket only."""
        scheme, bucket, path = parse_cloud_url("s3://bucket")
        self.assertEqual(scheme, "s3")
        self.assertEqual(bucket, "bucket")
        self.assertEqual(path, "")


class TestMatchesGlob(unittest.TestCase):
    """Test glob pattern matching function."""

    def test_no_patterns(self):
        """No patterns means match everything."""
        self.assertTrue(matches_glob("anything.txt", []))
        self.assertTrue(matches_glob("file.json", None))

    def test_include_pattern(self):
        """Include patterns filter correctly."""
        self.assertTrue(matches_glob("file.json", ["*.json"]))
        self.assertFalse(matches_glob("file.txt", ["*.json"]))

    def test_multiple_include_patterns(self):
        """Multiple include patterns work as OR."""
        patterns = ["*.json", "*.yaml"]
        self.assertTrue(matches_glob("config.json", patterns))
        self.assertTrue(matches_glob("config.yaml", patterns))
        self.assertFalse(matches_glob("config.xml", patterns))

    def test_exclude_pattern(self):
        """Exclude patterns filter correctly."""
        self.assertFalse(matches_glob("file.tmp", [], ["*.tmp"]))
        self.assertTrue(matches_glob("file.txt", [], ["*.tmp"]))

    def test_include_and_exclude(self):
        """Include and exclude patterns work together."""
        self.assertTrue(matches_glob("data.json", ["*.json"], ["test_*"]))
        self.assertFalse(matches_glob("test_data.json", ["*.json"], ["test_*"]))
        self.assertFalse(matches_glob("data.txt", ["*.json"], ["test_*"]))

    def test_question_mark_wildcard(self):
        """Question mark matches single character."""
        self.assertTrue(matches_glob("file1.txt", ["file?.txt"]))
        self.assertFalse(matches_glob("file10.txt", ["file?.txt"]))


class TestGetProvider(unittest.TestCase):
    """Test provider auto-detection."""

    @patch.dict(os.environ, {
        'AWS_ACCESS_KEY_ID': 'test',
        'AWS_SECRET_ACCESS_KEY': 'test'
    })
    @patch('s3_sync.S3Provider.__init__', return_value=None)
    def test_get_s3_provider(self, mock_init):
        """S3 provider detected from s3:// URL."""
        provider = get_provider("s3://bucket/key")
        self.assertIsInstance(provider, S3Provider)

    @patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': '/path/to/creds.json'})
    @patch('s3_sync.GCSProvider.__init__', return_value=None)
    @patch('os.path.exists', return_value=True)
    def test_get_gcs_provider(self, mock_exists, mock_init):
        """GCS provider detected from gs:// URL."""
        provider = get_provider("gs://bucket/key")
        self.assertIsInstance(provider, GCSProvider)

    @patch.dict(os.environ, {
        'AZURE_STORAGE_CONNECTION_STRING': 'DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;EndpointSuffix=core.windows.net'
    })
    @patch('s3_sync.AzureProvider.__init__', return_value=None)
    def test_get_azure_provider(self, mock_init):
        """Azure provider detected from az:// URL."""
        provider = get_provider("az://container/blob")
        self.assertIsInstance(provider, AzureProvider)

    def test_invalid_scheme(self):
        """Invalid scheme raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            get_provider("ftp://server/path")
        self.assertIn("Unknown URL scheme", str(ctx.exception))


class TestS3Provider(unittest.TestCase):
    """Test S3 provider operations."""

    def setUp(self):
        """Set up mocks for S3 provider."""
        self.env_patcher = patch.dict(os.environ, {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret'
        })
        self.env_patcher.start()

        # Mock boto3
        self.boto3_patcher = patch('s3_sync.S3Provider.__init__', return_value=None)
        self.boto3_patcher.start()

    def tearDown(self):
        """Clean up patches."""
        self.env_patcher.stop()
        self.boto3_patcher.stop()

    def test_parse_s3_path(self):
        """Test S3 path parsing."""
        bucket, key = S3Provider._parse_s3_path("s3://my-bucket/path/to/file.txt")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(key, "path/to/file.txt")

    def test_parse_s3_path_no_key(self):
        """Test S3 path parsing with no key."""
        bucket, key = S3Provider._parse_s3_path("s3://my-bucket/")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(key, "")


class TestGCSProvider(unittest.TestCase):
    """Test GCS provider operations."""

    def test_parse_gcs_path(self):
        """Test GCS path parsing."""
        bucket, key = GCSProvider._parse_gcs_path("gs://my-bucket/path/to/file.txt")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(key, "path/to/file.txt")


class TestAzureProvider(unittest.TestCase):
    """Test Azure provider operations."""

    def test_parse_azure_path(self):
        """Test Azure path parsing."""
        container, blob = AzureProvider._parse_azure_path("az://my-container/path/to/blob.txt")
        self.assertEqual(container, "my-container")
        self.assertEqual(blob, "path/to/blob.txt")


class TestArgumentParser(unittest.TestCase):
    """Test argument parser configuration."""

    def setUp(self):
        """Set up parser."""
        self.parser = create_parser()

    def test_upload_args(self):
        """Test upload command argument parsing."""
        args = self.parser.parse_args(['upload', 'local.txt', 's3://bucket/remote.txt'])
        self.assertEqual(args.command, 'upload')
        self.assertEqual(args.source, 'local.txt')
        self.assertEqual(args.destination, 's3://bucket/remote.txt')

    def test_download_args(self):
        """Test download command argument parsing."""
        args = self.parser.parse_args(['download', 's3://bucket/file.txt', './local/'])
        self.assertEqual(args.command, 'download')
        self.assertEqual(args.source, 's3://bucket/file.txt')
        self.assertEqual(args.destination, './local/')

    def test_presign_args_default_expires(self):
        """Test presign command with default expiration."""
        args = self.parser.parse_args(['presign', 's3://bucket/file.txt'])
        self.assertEqual(args.command, 'presign')
        self.assertEqual(args.path, 's3://bucket/file.txt')
        self.assertEqual(args.expires, 3600)

    def test_presign_args_custom_expires(self):
        """Test presign command with custom expiration."""
        args = self.parser.parse_args(['presign', 's3://bucket/file.txt', '--expires', '7200'])
        self.assertEqual(args.expires, 7200)

    def test_presign_args_short_expires(self):
        """Test presign command with short flag."""
        args = self.parser.parse_args(['presign', 's3://bucket/file.txt', '-e', '1800'])
        self.assertEqual(args.expires, 1800)

    def test_sync_args_basic(self):
        """Test sync command basic arguments."""
        args = self.parser.parse_args(['sync', './local/', 's3://bucket/remote/'])
        self.assertEqual(args.command, 'sync')
        self.assertEqual(args.source, './local/')
        self.assertEqual(args.destination, 's3://bucket/remote/')
        self.assertIsNone(args.include)
        self.assertIsNone(args.exclude)
        self.assertFalse(args.dry_run)
        self.assertFalse(args.delete)

    def test_sync_args_with_filters(self):
        """Test sync command with include/exclude filters."""
        args = self.parser.parse_args([
            'sync', './local/', 's3://bucket/remote/',
            '--include', '*.json',
            '--include', '*.yaml',
            '--exclude', '*.tmp'
        ])
        self.assertEqual(args.include, ['*.json', '*.yaml'])
        self.assertEqual(args.exclude, ['*.tmp'])

    def test_sync_args_dry_run(self):
        """Test sync command with dry-run flag."""
        args = self.parser.parse_args(['sync', './local/', 's3://bucket/', '--dry-run'])
        self.assertTrue(args.dry_run)

    def test_sync_args_delete(self):
        """Test sync command with delete flag."""
        args = self.parser.parse_args(['sync', './local/', 's3://bucket/', '--delete'])
        self.assertTrue(args.delete)

    def test_version_flag(self):
        """Test version flag."""
        with self.assertRaises(SystemExit) as ctx:
            self.parser.parse_args(['--version'])
        self.assertEqual(ctx.exception.code, 0)


class TestUploadCommand(unittest.TestCase):
    """Test upload command functionality."""

    @patch('s3_sync.get_provider')
    def test_upload_success(self, mock_get_provider):
        """Test successful upload."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            args = MagicMock()
            args.source = temp_path
            args.destination = 's3://bucket/file.txt'

            with patch('sys.stdout', new=StringIO()):
                result = cmd_upload(args)

            self.assertEqual(result, EXIT_SUCCESS)
            mock_provider.upload.assert_called_once()
        finally:
            os.unlink(temp_path)

    def test_upload_file_not_found(self):
        """Test upload with non-existent file."""
        args = MagicMock()
        args.source = '/nonexistent/file.txt'
        args.destination = 's3://bucket/file.txt'

        with patch('sys.stderr', new=StringIO()):
            result = cmd_upload(args)

        self.assertEqual(result, EXIT_NOT_FOUND)

    @patch('s3_sync.get_provider')
    def test_upload_credentials_error(self, mock_get_provider):
        """Test upload with credentials error."""
        mock_get_provider.side_effect = CredentialsError("No credentials")

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            args = MagicMock()
            args.source = temp_path
            args.destination = 's3://bucket/file.txt'

            with patch('sys.stderr', new=StringIO()):
                result = cmd_upload(args)

            self.assertEqual(result, EXIT_CREDENTIALS_ERROR)
        finally:
            os.unlink(temp_path)

    @patch('s3_sync.get_provider')
    def test_upload_appends_filename(self, mock_get_provider):
        """Test upload appends filename when destination ends with /."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
            f.write(b"test content")
            temp_path = f.name
            filename = os.path.basename(temp_path)

        try:
            args = MagicMock()
            args.source = temp_path
            args.destination = 's3://bucket/path/'

            with patch('sys.stdout', new=StringIO()):
                result = cmd_upload(args)

            self.assertEqual(result, EXIT_SUCCESS)
            # Check that filename was appended
            call_args = mock_provider.upload.call_args[0]
            self.assertTrue(call_args[1].endswith(filename))
        finally:
            os.unlink(temp_path)


class TestDownloadCommand(unittest.TestCase):
    """Test download command functionality."""

    @patch('s3_sync.get_provider')
    def test_download_success(self, mock_get_provider):
        """Test successful download."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.source = 's3://bucket/file.txt'
            args.destination = os.path.join(tmpdir, 'file.txt')

            with patch('sys.stdout', new=StringIO()):
                result = cmd_download(args)

            self.assertEqual(result, EXIT_SUCCESS)
            mock_provider.download.assert_called_once()

    @patch('s3_sync.get_provider')
    def test_download_with_glob(self, mock_get_provider):
        """Test download with glob pattern."""
        mock_provider = MagicMock()
        mock_provider.list_objects.return_value = iter(['data/file1.csv', 'data/file2.csv'])
        mock_get_provider.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.source = 's3://bucket/data/*.csv'
            args.destination = tmpdir

            with patch('sys.stdout', new=StringIO()):
                result = cmd_download(args)

            self.assertEqual(result, EXIT_SUCCESS)
            self.assertEqual(mock_provider.download.call_count, 2)

    @patch('s3_sync.get_provider')
    def test_download_no_matches(self, mock_get_provider):
        """Test download with no matching files."""
        mock_provider = MagicMock()
        mock_provider.list_objects.return_value = iter([])
        mock_get_provider.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.source = 's3://bucket/data/*.xyz'
            args.destination = tmpdir

            with patch('sys.stderr', new=StringIO()):
                result = cmd_download(args)

            self.assertEqual(result, EXIT_NOT_FOUND)

    @patch('s3_sync.get_provider')
    def test_download_not_found(self, mock_get_provider):
        """Test download with not found error."""
        mock_provider = MagicMock()
        mock_provider.download.side_effect = NotFoundError("Object not found")
        mock_get_provider.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.source = 's3://bucket/missing.txt'
            args.destination = os.path.join(tmpdir, 'file.txt')

            with patch('sys.stderr', new=StringIO()):
                result = cmd_download(args)

            self.assertEqual(result, EXIT_NOT_FOUND)


class TestPresignCommand(unittest.TestCase):
    """Test presign command functionality."""

    @patch('s3_sync.get_provider')
    def test_presign_success(self, mock_get_provider):
        """Test successful presign."""
        mock_provider = MagicMock()
        mock_provider.presign.return_value = "https://example.com/presigned-url"
        mock_get_provider.return_value = mock_provider

        args = MagicMock()
        args.path = 's3://bucket/file.txt'
        args.expires = 3600

        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            result = cmd_presign(args)

        self.assertEqual(result, EXIT_SUCCESS)
        mock_provider.presign.assert_called_once_with('s3://bucket/file.txt', 3600)

    def test_presign_invalid_expires(self):
        """Test presign with invalid expiration."""
        args = MagicMock()
        args.path = 's3://bucket/file.txt'
        args.expires = -1

        with patch('sys.stderr', new=StringIO()):
            result = cmd_presign(args)

        self.assertEqual(result, EXIT_INVALID_ARGS)

    @patch('s3_sync.get_provider')
    def test_presign_max_expires(self, mock_get_provider):
        """Test presign caps expiration at 7 days."""
        mock_provider = MagicMock()
        mock_provider.presign.return_value = "https://example.com/presigned-url"
        mock_get_provider.return_value = mock_provider

        args = MagicMock()
        args.path = 's3://bucket/file.txt'
        args.expires = 1000000  # More than 7 days

        with patch('sys.stdout', new=StringIO()), patch('sys.stderr', new=StringIO()):
            result = cmd_presign(args)

        self.assertEqual(result, EXIT_SUCCESS)
        # Should be capped at 604800 (7 days)
        mock_provider.presign.assert_called_once_with('s3://bucket/file.txt', 604800)


class TestSyncCommand(unittest.TestCase):
    """Test sync command functionality."""

    @patch('s3_sync.get_provider')
    def test_sync_upload(self, mock_get_provider):
        """Test sync from local to cloud."""
        mock_provider = MagicMock()
        mock_provider.list_objects.return_value = iter([])
        mock_get_provider.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_file = os.path.join(tmpdir, 'test.json')
            with open(test_file, 'w') as f:
                f.write('{"test": true}')

            args = MagicMock()
            args.source = tmpdir
            args.destination = 's3://bucket/backup/'
            args.include = None
            args.exclude = None
            args.dry_run = False
            args.delete = False

            with patch('sys.stdout', new=StringIO()):
                result = cmd_sync(args)

            self.assertEqual(result, EXIT_SUCCESS)
            mock_provider.upload.assert_called()

    @patch('s3_sync.get_provider')
    def test_sync_download(self, mock_get_provider):
        """Test sync from cloud to local."""
        mock_provider = MagicMock()
        mock_provider.list_objects.return_value = iter(['backup/test.json'])
        mock_get_provider.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.source = 's3://bucket/backup/'
            args.destination = tmpdir
            args.include = None
            args.exclude = None
            args.dry_run = False
            args.delete = False

            with patch('sys.stdout', new=StringIO()):
                result = cmd_sync(args)

            self.assertEqual(result, EXIT_SUCCESS)
            mock_provider.download.assert_called()

    @patch('s3_sync.get_provider')
    def test_sync_with_include(self, mock_get_provider):
        """Test sync with include filter."""
        mock_provider = MagicMock()
        mock_provider.list_objects.return_value = iter([])
        mock_get_provider.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            json_file = os.path.join(tmpdir, 'data.json')
            txt_file = os.path.join(tmpdir, 'data.txt')
            with open(json_file, 'w') as f:
                f.write('{}')
            with open(txt_file, 'w') as f:
                f.write('text')

            args = MagicMock()
            args.source = tmpdir
            args.destination = 's3://bucket/backup/'
            args.include = ['*.json']
            args.exclude = None
            args.dry_run = False
            args.delete = False

            with patch('sys.stdout', new=StringIO()):
                result = cmd_sync(args)

            self.assertEqual(result, EXIT_SUCCESS)
            # Should only upload JSON file
            self.assertEqual(mock_provider.upload.call_count, 1)

    @patch('s3_sync.get_provider')
    def test_sync_dry_run(self, mock_get_provider):
        """Test sync dry run."""
        mock_provider = MagicMock()
        mock_provider.list_objects.return_value = iter([])
        mock_get_provider.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')

            args = MagicMock()
            args.source = tmpdir
            args.destination = 's3://bucket/backup/'
            args.include = None
            args.exclude = None
            args.dry_run = True
            args.delete = False

            with patch('sys.stdout', new=StringIO()) as mock_stdout:
                result = cmd_sync(args)

            self.assertEqual(result, EXIT_SUCCESS)
            # Should not actually upload
            mock_provider.upload.assert_not_called()

    def test_sync_both_local(self):
        """Test sync fails when both paths are local."""
        args = MagicMock()
        args.source = './local1/'
        args.destination = './local2/'
        args.include = None
        args.exclude = None
        args.dry_run = False
        args.delete = False

        with patch('sys.stderr', new=StringIO()):
            result = cmd_sync(args)

        self.assertEqual(result, EXIT_INVALID_ARGS)

    def test_sync_both_cloud(self):
        """Test sync fails when both paths are cloud URLs."""
        args = MagicMock()
        args.source = 's3://bucket1/path/'
        args.destination = 'gs://bucket2/path/'
        args.include = None
        args.exclude = None
        args.dry_run = False
        args.delete = False

        with patch('sys.stderr', new=StringIO()):
            result = cmd_sync(args)

        self.assertEqual(result, EXIT_INVALID_ARGS)


class TestMain(unittest.TestCase):
    """Test main function."""

    @patch('sys.argv', ['s3_sync', '--help'])
    def test_main_help(self):
        """Test main with help flag."""
        with self.assertRaises(SystemExit) as ctx:
            main()
        self.assertEqual(ctx.exception.code, 0)

    @patch('s3_sync.cmd_upload')
    @patch('sys.argv', ['s3_sync', 'upload', 'file.txt', 's3://bucket/file.txt'])
    def test_main_upload(self, mock_upload):
        """Test main dispatches to upload command."""
        mock_upload.return_value = EXIT_SUCCESS
        result = main()
        self.assertEqual(result, EXIT_SUCCESS)
        mock_upload.assert_called_once()


class TestHelpOutput(unittest.TestCase):
    """Test that help output is comprehensive and well-formatted."""

    def setUp(self):
        """Set up parser."""
        self.parser = create_parser()

    def test_main_help_contains_examples(self):
        """Main help contains usage examples."""
        help_text = self.parser.format_help()
        self.assertIn('EXAMPLES:', help_text)
        self.assertIn('s3://', help_text)
        self.assertIn('gs://', help_text)
        self.assertIn('az://', help_text)

    def test_main_help_contains_env_vars(self):
        """Main help documents environment variables."""
        help_text = self.parser.format_help()
        self.assertIn('AWS_ACCESS_KEY_ID', help_text)
        self.assertIn('AWS_SECRET_ACCESS_KEY', help_text)
        self.assertIn('GOOGLE_APPLICATION_CREDENTIALS', help_text)
        self.assertIn('AZURE_STORAGE_CONNECTION_STRING', help_text)

    def test_main_help_contains_exit_codes(self):
        """Main help documents exit codes."""
        help_text = self.parser.format_help()
        self.assertIn('EXIT CODES:', help_text)
        self.assertIn('Success', help_text)
        self.assertIn('Credentials error', help_text)

    def test_upload_help(self):
        """Upload command has comprehensive help."""
        upload_parser = None
        for action in self.parser._subparsers._actions:
            if hasattr(action, '_parser_class'):
                for name, parser in action.choices.items():
                    if name == 'upload':
                        upload_parser = parser
                        break

        self.assertIsNotNone(upload_parser)
        help_text = upload_parser.format_help()
        self.assertIn('EXAMPLES:', help_text)

    def test_sync_help_documents_filters(self):
        """Sync command help documents filter options."""
        sync_parser = None
        for action in self.parser._subparsers._actions:
            if hasattr(action, '_parser_class'):
                for name, parser in action.choices.items():
                    if name == 'sync':
                        sync_parser = parser
                        break

        self.assertIsNotNone(sync_parser)
        help_text = sync_parser.format_help()
        self.assertIn('--include', help_text)
        self.assertIn('--exclude', help_text)
        self.assertIn('--dry-run', help_text)
        self.assertIn('--delete', help_text)


class TestCredentialsHandling(unittest.TestCase):
    """Test credentials error handling for each provider."""

    @patch.dict(os.environ, {}, clear=True)
    def test_s3_missing_credentials(self):
        """S3 provider raises CredentialsError without credentials."""
        # Clear AWS env vars and mock boto3 to simulate no credentials
        with patch.dict(os.environ, {}, clear=True):
            try:
                with patch('boto3.Session') as mock_session:
                    mock_session.return_value.get_credentials.return_value = None
                    with self.assertRaises(CredentialsError):
                        S3Provider()
            except ModuleNotFoundError:
                # boto3 not installed - test that it raises CredentialsError
                with self.assertRaises(CredentialsError) as ctx:
                    S3Provider()
                self.assertIn('boto3 is not installed', str(ctx.exception))

    @patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': '/nonexistent/path.json'})
    def test_gcs_missing_credentials_file(self):
        """GCS provider raises CredentialsError when credentials file doesn't exist."""
        with self.assertRaises(CredentialsError) as ctx:
            GCSProvider()
        # Either "file not found" (file missing) or "not installed" (package missing)
        error_msg = str(ctx.exception).lower()
        self.assertTrue(
            'file not found' in error_msg or 'not installed' in error_msg,
            f"Unexpected error message: {ctx.exception}"
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_azure_missing_credentials(self):
        """Azure provider raises CredentialsError without connection string."""
        with self.assertRaises(CredentialsError) as ctx:
            AzureProvider()
        # Either connection string missing or package not installed
        error_msg = str(ctx.exception)
        self.assertTrue(
            'AZURE_STORAGE_CONNECTION_STRING' in error_msg or 'not installed' in error_msg,
            f"Unexpected error message: {ctx.exception}"
        )


class TestIntegrationScenarios(unittest.TestCase):
    """Integration-style tests for common usage scenarios."""

    @patch('s3_sync.get_provider')
    def test_upload_download_cycle(self, mock_get_provider):
        """Test uploading and then downloading a file."""
        mock_provider = MagicMock()
        mock_provider.presign.return_value = "https://example.com/signed"
        mock_get_provider.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source file
            source_file = os.path.join(tmpdir, 'test.txt')
            with open(source_file, 'w') as f:
                f.write('Hello, World!')

            # Upload
            upload_args = MagicMock()
            upload_args.source = source_file
            upload_args.destination = 's3://bucket/test.txt'

            with patch('sys.stdout', new=StringIO()):
                result = cmd_upload(upload_args)
            self.assertEqual(result, EXIT_SUCCESS)

            # Download
            dest_file = os.path.join(tmpdir, 'downloaded.txt')
            download_args = MagicMock()
            download_args.source = 's3://bucket/test.txt'
            download_args.destination = dest_file

            with patch('sys.stdout', new=StringIO()):
                result = cmd_download(download_args)
            self.assertEqual(result, EXIT_SUCCESS)

            # Presign
            presign_args = MagicMock()
            presign_args.path = 's3://bucket/test.txt'
            presign_args.expires = 3600

            with patch('sys.stdout', new=StringIO()):
                result = cmd_presign(presign_args)
            self.assertEqual(result, EXIT_SUCCESS)


if __name__ == '__main__':
    unittest.main(verbosity=2)
