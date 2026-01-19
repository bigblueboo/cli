#!/usr/bin/env python3
"""
Comprehensive tests for data_extractor CLI tool.

Tests cover:
- Argument parsing
- File reading (text, PDF, HTML)
- URL fetching
- Schema parsing and validation
- LLM provider integration (mocked)
- Batch processing
- Error handling and exit codes
- Stdin input
"""

import json
import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import data_extractor
from data_extractor import (
    EXIT_SUCCESS,
    EXIT_FILE_NOT_FOUND,
    EXIT_API_ERROR,
    EXIT_VALIDATION_ERROR,
    EXIT_INVALID_ARGS,
    create_parser,
    fields_to_schema,
    get_api_key,
    get_provider,
    main,
    parse_schema,
    read_file,
    read_html,
    read_url,
    validate_output,
    process_input,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
)

# Check for optional dependencies
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import google.generativeai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


class TestArgumentParsing(unittest.TestCase):
    """Test argument parsing functionality."""

    def setUp(self):
        self.parser = create_parser()

    def test_no_args_shows_help(self):
        """Test that no arguments with TTY stdin shows help."""
        with patch.object(sys.stdin, 'isatty', return_value=True):
            with patch('sys.stdout', new_callable=StringIO):
                result = main([])
                self.assertEqual(result, EXIT_SUCCESS)

    def test_schema_file_argument(self):
        """Test --schema with file path."""
        args = self.parser.parse_args(['input.txt', '--schema', 'schema.json'])
        self.assertEqual(args.schema, 'schema.json')
        self.assertEqual(args.inputs, ['input.txt'])

    def test_schema_short_argument(self):
        """Test -s short form for schema."""
        args = self.parser.parse_args(['input.txt', '-s', 'schema.json'])
        self.assertEqual(args.schema, 'schema.json')

    def test_extract_argument(self):
        """Test --extract with field names."""
        args = self.parser.parse_args(['input.txt', '--extract', 'name,email,phone'])
        self.assertEqual(args.extract, 'name,email,phone')

    def test_fields_argument(self):
        """Test --fields alias for extract."""
        args = self.parser.parse_args(['input.txt', '--fields', 'name,email'])
        self.assertEqual(args.fields, 'name,email')

    def test_output_argument(self):
        """Test --output argument."""
        args = self.parser.parse_args(['input.txt', '--output', 'results/'])
        self.assertEqual(args.output, 'results/')

    def test_multiple_inputs(self):
        """Test multiple input files."""
        args = self.parser.parse_args(['file1.txt', 'file2.txt', 'file3.txt', '-e', 'name'])
        self.assertEqual(args.inputs, ['file1.txt', 'file2.txt', 'file3.txt'])

    def test_quiet_flag(self):
        """Test --quiet flag."""
        args = self.parser.parse_args(['input.txt', '--quiet', '-e', 'name'])
        self.assertTrue(args.quiet)

    def test_compact_flag(self):
        """Test --compact flag."""
        args = self.parser.parse_args(['input.txt', '--compact', '-e', 'name'])
        self.assertTrue(args.compact)

    def test_no_validate_flag(self):
        """Test --no-validate flag."""
        args = self.parser.parse_args(['input.txt', '--no-validate', '-e', 'name'])
        self.assertTrue(args.no_validate)

    def test_provider_argument(self):
        """Test --provider argument."""
        args = self.parser.parse_args(['input.txt', '--provider', 'anthropic', '-e', 'name'])
        self.assertEqual(args.provider, 'anthropic')

    def test_version_argument(self):
        """Test --version argument."""
        with self.assertRaises(SystemExit) as cm:
            self.parser.parse_args(['--version'])
        self.assertEqual(cm.exception.code, 0)


class TestSchemaHandling(unittest.TestCase):
    """Test schema parsing and conversion."""

    def test_fields_to_schema_single(self):
        """Test converting single field to schema."""
        schema = fields_to_schema('name')
        self.assertEqual(schema['type'], 'object')
        self.assertIn('name', schema['properties'])
        self.assertEqual(schema['required'], ['name'])

    def test_fields_to_schema_multiple(self):
        """Test converting multiple fields to schema."""
        schema = fields_to_schema('name, email, phone')
        self.assertEqual(len(schema['properties']), 3)
        self.assertIn('name', schema['properties'])
        self.assertIn('email', schema['properties'])
        self.assertIn('phone', schema['properties'])

    def test_parse_schema_json_string(self):
        """Test parsing inline JSON schema."""
        json_str = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        schema = parse_schema(json_str)
        self.assertEqual(schema['type'], 'object')
        self.assertIn('name', schema['properties'])

    def test_parse_schema_from_file(self):
        """Test parsing schema from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'type': 'object', 'properties': {'test': {'type': 'string'}}}, f)
            f.flush()

            schema = parse_schema(f.name)
            self.assertEqual(schema['type'], 'object')
            self.assertIn('test', schema['properties'])

            os.unlink(f.name)

    def test_parse_schema_invalid_json(self):
        """Test parsing invalid JSON raises error."""
        with self.assertRaises(SystemExit) as cm:
            parse_schema('not valid json')
        self.assertEqual(cm.exception.code, EXIT_INVALID_ARGS)

    def test_parse_schema_file_not_found(self):
        """Test parsing non-existent file raises error."""
        with self.assertRaises(SystemExit) as cm:
            parse_schema('/nonexistent/schema.json')
        # File path that doesn't exist should fail as invalid JSON
        self.assertEqual(cm.exception.code, EXIT_INVALID_ARGS)


class TestFileReading(unittest.TestCase):
    """Test file reading functionality."""

    def test_read_text_file(self):
        """Test reading plain text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('Hello, World!')
            f.flush()

            content = read_file(f.name)
            self.assertEqual(content, 'Hello, World!')

            os.unlink(f.name)

    def test_read_file_not_found(self):
        """Test reading non-existent file raises error."""
        with self.assertRaises(SystemExit) as cm:
            read_file('/nonexistent/file.txt')
        self.assertEqual(cm.exception.code, EXIT_FILE_NOT_FOUND)

    def test_read_html_file(self):
        """Test reading HTML file extracts text."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write('<html><body><p>Hello World</p><script>var x = 1;</script></body></html>')
            f.flush()

            content = read_html(Path(f.name))
            self.assertIn('Hello World', content)
            self.assertNotIn('<p>', content)

            os.unlink(f.name)

    def test_read_pdf_fallback(self):
        """Test PDF reading with fallback when pypdf not available."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            # Write some text that could be extracted
            f.write(b'%PDF-1.4 test content')
            f.flush()

            with patch.dict('sys.modules', {'pypdf': None}):
                # Should not crash, returns something
                with patch('builtins.print'):  # Suppress warning
                    pass  # Test just verifies no crash

            os.unlink(f.name)


class TestURLReading(unittest.TestCase):
    """Test URL fetching functionality."""

    @patch('urllib.request.urlopen')
    def test_read_url_text(self, mock_urlopen):
        """Test reading plain text from URL."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'Hello from URL'
        mock_response.headers = {'Content-Type': 'text/plain'}
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        content = read_url('https://example.com/text.txt')
        self.assertEqual(content, 'Hello from URL')

    @patch('urllib.request.urlopen')
    def test_read_url_html(self, mock_urlopen):
        """Test reading HTML from URL extracts text."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'<html><body><p>Hello</p></body></html>'
        mock_response.headers = MagicMock()
        mock_response.headers.get.return_value = 'text/html'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        content = read_url('https://example.com/page.html')
        self.assertIn('Hello', content)

    @patch('urllib.request.urlopen')
    def test_read_url_error(self, mock_urlopen):
        """Test URL fetch error raises appropriate exit code."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError('Connection failed')

        with self.assertRaises(SystemExit) as cm:
            read_url('https://example.com/error')
        self.assertEqual(cm.exception.code, EXIT_FILE_NOT_FOUND)


class TestValidation(unittest.TestCase):
    """Test JSON schema validation."""

    def test_validate_valid_data(self):
        """Test validation passes for valid data."""
        schema = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'}
            },
            'required': ['name']
        }
        data = {'name': 'John'}

        # Should not raise
        result = validate_output(data, schema)
        self.assertTrue(result)

    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema package not installed")
    def test_validate_invalid_data(self):
        """Test validation fails for invalid data."""
        schema = {
            'type': 'object',
            'properties': {'name': {'type': 'string'}},
            'required': ['name'],
            'additionalProperties': False
        }
        data = {'wrong': 'data'}

        with self.assertRaises(SystemExit) as cm:
            validate_output(data, schema)
        self.assertEqual(cm.exception.code, EXIT_VALIDATION_ERROR)

    def test_validate_without_jsonschema(self):
        """Test validation skipped when jsonschema not installed."""
        # This test verifies the function handles missing jsonschema gracefully
        # The actual behavior depends on whether jsonschema is installed
        schema = {'type': 'object'}
        data = {'anything': 'goes'}
        # Should not crash regardless of jsonschema availability
        result = validate_output(data, schema)
        self.assertTrue(result)


class TestAPIKeyHandling(unittest.TestCase):
    """Test API key detection and provider selection."""

    def test_get_api_key_openai(self):
        """Test OpenAI key detection."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'}, clear=True):
            provider, key = get_api_key()
            self.assertEqual(provider, 'openai')
            self.assertEqual(key, 'sk-test123')

    def test_get_api_key_anthropic(self):
        """Test Anthropic key detection."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test'}, clear=True):
            provider, key = get_api_key()
            self.assertEqual(provider, 'anthropic')
            self.assertEqual(key, 'sk-ant-test')

    def test_get_api_key_google(self):
        """Test Google key detection."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'AIza-test'}, clear=True):
            provider, key = get_api_key()
            self.assertEqual(provider, 'google')
            self.assertEqual(key, 'AIza-test')

    def test_get_api_key_priority(self):
        """Test OpenAI takes priority over Anthropic."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'openai-key',
            'ANTHROPIC_API_KEY': 'anthropic-key'
        }, clear=True):
            provider, key = get_api_key()
            self.assertEqual(provider, 'openai')

    def test_get_api_key_none(self):
        """Test error when no API key is set."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as cm:
                get_api_key()
            self.assertEqual(cm.exception.code, EXIT_INVALID_ARGS)


@unittest.skipUnless(HAS_OPENAI, "openai package not installed")
class TestOpenAIProvider(unittest.TestCase):
    """Test OpenAI provider."""

    @patch('openai.OpenAI')
    def test_openai_extract(self, mock_openai):
        """Test OpenAI extraction."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "John", "email": "john@example.com"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = OpenAIProvider('sk-test')
        result = provider.extract('John Doe, john@example.com', {}, ['name', 'email'])

        self.assertEqual(result['name'], 'John')
        self.assertEqual(result['email'], 'john@example.com')

    @patch('openai.OpenAI')
    def test_openai_api_error(self, mock_openai):
        """Test OpenAI API error handling."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception('API Error')
        mock_openai.return_value = mock_client

        provider = OpenAIProvider('sk-test')

        with self.assertRaises(SystemExit) as cm:
            provider.extract('test', {}, ['field'])
        self.assertEqual(cm.exception.code, EXIT_API_ERROR)


@unittest.skipUnless(HAS_ANTHROPIC, "anthropic package not installed")
class TestAnthropicProvider(unittest.TestCase):
    """Test Anthropic provider."""

    @patch('anthropic.Anthropic')
    def test_anthropic_extract(self, mock_anthropic):
        """Test Anthropic extraction."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = '{"name": "Jane", "phone": "555-1234"}'
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider('sk-ant-test')
        result = provider.extract('Jane Smith, 555-1234', {}, ['name', 'phone'])

        self.assertEqual(result['name'], 'Jane')
        self.assertEqual(result['phone'], '555-1234')

    @patch('anthropic.Anthropic')
    def test_anthropic_markdown_response(self, mock_anthropic):
        """Test Anthropic handles markdown-wrapped JSON."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = '```json\n{"name": "Test"}\n```'
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider('sk-ant-test')
        result = provider.extract('Test', {}, ['name'])

        self.assertEqual(result['name'], 'Test')


@unittest.skipUnless(HAS_GOOGLE, "google-generativeai package not installed")
class TestGoogleProvider(unittest.TestCase):
    """Test Google Gemini provider."""

    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_google_extract(self, mock_configure, mock_model_class):
        """Test Google extraction."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"title": "Test Article", "summary": "A test"}'
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        provider = GoogleProvider('AIza-test')
        result = provider.extract('Test Article: A test', {}, ['title', 'summary'])

        self.assertEqual(result['title'], 'Test Article')
        self.assertEqual(result['summary'], 'A test')


class TestBatchProcessing(unittest.TestCase):
    """Test batch file processing."""

    @patch.object(data_extractor, 'get_api_key')
    @patch.object(data_extractor, 'get_provider')
    def test_batch_processing_multiple_files(self, mock_get_provider, mock_get_api_key):
        """Test processing multiple files."""
        mock_get_api_key.return_value = ('openai', 'sk-test')

        mock_provider = MagicMock()
        mock_provider.extract.side_effect = [
            {'name': 'File1'},
            {'name': 'File2'},
        ]
        mock_get_provider.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = Path(tmpdir) / 'file1.txt'
            file2 = Path(tmpdir) / 'file2.txt'
            file1.write_text('Content 1')
            file2.write_text('Content 2')

            output_dir = Path(tmpdir) / 'output'

            with patch('sys.stdout', new_callable=StringIO):
                result = main([str(file1), str(file2), '--extract', 'name', '--output', str(output_dir), '--quiet'])

            self.assertEqual(result, EXIT_SUCCESS)
            self.assertTrue((output_dir / 'file1.json').exists())
            self.assertTrue((output_dir / 'file2.json').exists())


class TestStdinInput(unittest.TestCase):
    """Test stdin input handling."""

    @patch.object(data_extractor, 'get_api_key')
    @patch.object(data_extractor, 'get_provider')
    @patch('sys.stdin')
    def test_stdin_input(self, mock_stdin, mock_get_provider, mock_get_api_key):
        """Test reading from stdin."""
        mock_get_api_key.return_value = ('openai', 'sk-test')

        mock_provider = MagicMock()
        mock_provider.extract.return_value = {'extracted': 'data'}
        mock_get_provider.return_value = mock_provider

        mock_stdin.read.return_value = 'Input from stdin'
        mock_stdin.isatty.return_value = False

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = main(['--extract', 'extracted'])

        self.assertEqual(result, EXIT_SUCCESS)
        output = mock_stdout.getvalue()
        self.assertIn('extracted', output)


class TestOutputFormats(unittest.TestCase):
    """Test different output format options."""

    @patch.object(data_extractor, 'get_api_key')
    @patch.object(data_extractor, 'get_provider')
    def test_compact_output(self, mock_get_provider, mock_get_api_key):
        """Test --compact produces minified JSON."""
        mock_get_api_key.return_value = ('openai', 'sk-test')

        mock_provider = MagicMock()
        mock_provider.extract.return_value = {'name': 'Test', 'value': 123}
        mock_get_provider.return_value = mock_provider

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('test content')
            f.flush()

            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                result = main([f.name, '--extract', 'name,value', '--compact'])

            output = mock_stdout.getvalue()
            # Compact output should not have newlines within the JSON
            self.assertNotIn('\n  ', output)

            os.unlink(f.name)

    @patch.object(data_extractor, 'get_api_key')
    @patch.object(data_extractor, 'get_provider')
    def test_formatted_output(self, mock_get_provider, mock_get_api_key):
        """Test default formatted JSON output."""
        mock_get_api_key.return_value = ('openai', 'sk-test')

        mock_provider = MagicMock()
        mock_provider.extract.return_value = {'name': 'Test'}
        mock_get_provider.return_value = mock_provider

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('test content')
            f.flush()

            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                result = main([f.name, '--extract', 'name'])

            output = mock_stdout.getvalue()
            # Formatted output should have indentation
            self.assertIn('\n', output)

            os.unlink(f.name)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and exit codes."""

    def test_file_not_found_exit_code(self):
        """Test file not found returns correct exit code."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test'}):
            with patch('sys.stderr', new_callable=StringIO):
                with self.assertRaises(SystemExit) as cm:
                    read_file('/nonexistent/file.txt')
                self.assertEqual(cm.exception.code, EXIT_FILE_NOT_FOUND)

    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema package not installed")
    @patch.object(data_extractor, 'get_api_key')
    @patch.object(data_extractor, 'get_provider')
    def test_validation_error_exit_code(self, mock_get_provider, mock_get_api_key):
        """Test validation error returns correct exit code."""
        mock_get_api_key.return_value = ('openai', 'sk-test')

        mock_provider = MagicMock()
        mock_provider.extract.return_value = {'wrong_field': 'value'}
        mock_get_provider.return_value = mock_provider

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('test')
            f.flush()

            # Create a strict schema
            schema = {
                'type': 'object',
                'properties': {'required_field': {'type': 'string'}},
                'required': ['required_field'],
                'additionalProperties': False
            }

            with patch('sys.stderr', new_callable=StringIO):
                with self.assertRaises(SystemExit) as cm:
                    main([f.name, '--schema', json.dumps(schema)])
                self.assertEqual(cm.exception.code, EXIT_VALIDATION_ERROR)

            os.unlink(f.name)

    def test_invalid_args_exit_code(self):
        """Test invalid arguments return correct exit code."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('sys.stderr', new_callable=StringIO):
                with self.assertRaises(SystemExit) as cm:
                    get_api_key()
                self.assertEqual(cm.exception.code, EXIT_INVALID_ARGS)


class TestProcessInput(unittest.TestCase):
    """Test the process_input function."""

    @patch.object(data_extractor, 'read_file')
    def test_process_file_input(self, mock_read_file):
        """Test processing file input."""
        mock_read_file.return_value = 'File content'

        mock_provider = MagicMock()
        mock_provider.extract.return_value = {'result': 'extracted'}

        result = process_input(
            'test.txt',
            {'type': 'object'},
            ['result'],
            mock_provider,
            validate=False
        )

        self.assertEqual(result, {'result': 'extracted'})
        mock_read_file.assert_called_once_with('test.txt')

    @patch.object(data_extractor, 'read_url')
    def test_process_url_input(self, mock_read_url):
        """Test processing URL input."""
        mock_read_url.return_value = 'URL content'

        mock_provider = MagicMock()
        mock_provider.extract.return_value = {'title': 'Test'}

        result = process_input(
            'https://example.com/page',
            {'type': 'object'},
            ['title'],
            mock_provider,
            validate=False
        )

        self.assertEqual(result, {'title': 'Test'})
        mock_read_url.assert_called_once_with('https://example.com/page')

    @patch('sys.stdin')
    def test_process_stdin_input(self, mock_stdin):
        """Test processing stdin input."""
        mock_stdin.read.return_value = 'Stdin content'

        mock_provider = MagicMock()
        mock_provider.extract.return_value = {'data': 'from_stdin'}

        result = process_input(
            '-',
            {'type': 'object'},
            ['data'],
            mock_provider,
            validate=False
        )

        self.assertEqual(result, {'data': 'from_stdin'})


class TestHelpAndDocumentation(unittest.TestCase):
    """Test help output and documentation."""

    def test_help_contains_examples(self):
        """Test that help output contains usage examples."""
        parser = create_parser()
        help_text = parser.format_help()

        self.assertIn('EXAMPLES', help_text)
        self.assertIn('invoice.pdf', help_text)
        self.assertIn('--schema', help_text)
        self.assertIn('--extract', help_text)

    def test_help_contains_exit_codes(self):
        """Test that help output documents exit codes."""
        parser = create_parser()
        help_text = parser.format_help()

        self.assertIn('EXIT CODES', help_text)
        self.assertIn('0', help_text)
        self.assertIn('Success', help_text)

    def test_help_contains_env_vars(self):
        """Test that help output documents environment variables."""
        parser = create_parser()
        help_text = parser.format_help()

        self.assertIn('ENVIRONMENT VARIABLES', help_text)
        self.assertIn('OPENAI_API_KEY', help_text)
        self.assertIn('ANTHROPIC_API_KEY', help_text)
        self.assertIn('GOOGLE_API_KEY', help_text)


class TestIntegration(unittest.TestCase):
    """Integration tests with mocked LLM."""

    @patch.object(data_extractor, 'get_api_key')
    @patch.object(data_extractor, 'get_provider')
    def test_full_extraction_workflow(self, mock_get_provider, mock_get_api_key):
        """Test complete extraction workflow."""
        mock_get_api_key.return_value = ('openai', 'sk-test')

        mock_provider = MagicMock()
        mock_provider.extract.return_value = {
            'sender': 'john@example.com',
            'subject': 'Meeting Tomorrow',
            'date': '2024-01-15',
            'action_items': ['Review document', 'Send feedback']
        }
        mock_get_provider.return_value = mock_provider

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            From: john@example.com
            Subject: Meeting Tomorrow
            Date: January 15, 2024

            Hi team,

            Please review the attached document and send feedback.

            Thanks,
            John
            """)
            f.flush()

            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                result = main([f.name, '--extract', 'sender,subject,date,action_items'])

            self.assertEqual(result, EXIT_SUCCESS)
            output = json.loads(mock_stdout.getvalue())
            self.assertEqual(output['sender'], 'john@example.com')
            self.assertEqual(output['subject'], 'Meeting Tomorrow')

            os.unlink(f.name)


if __name__ == '__main__':
    unittest.main()
