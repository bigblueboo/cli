#!/usr/bin/env python3
"""
Comprehensive tests for llm_query CLI tool.

Uses mocking to simulate API responses without requiring actual API keys.
"""

import json
import os
import sys
import tempfile
from io import StringIO
from unittest import mock

import pytest

# Import the module under test
import llm_query


class TestDetectProvider:
    """Tests for provider detection based on model names."""

    def test_openai_gpt4o(self):
        assert llm_query.detect_provider('gpt-4o') == 'openai'

    def test_openai_gpt4o_mini(self):
        assert llm_query.detect_provider('gpt-4o-mini') == 'openai'

    def test_openai_gpt4_turbo(self):
        assert llm_query.detect_provider('gpt-4-turbo') == 'openai'

    def test_openai_o1_preview(self):
        assert llm_query.detect_provider('o1-preview') == 'openai'

    def test_openai_o1_mini(self):
        assert llm_query.detect_provider('o1-mini') == 'openai'

    def test_openai_o3_mini(self):
        assert llm_query.detect_provider('o3-mini') == 'openai'

    def test_anthropic_claude_35_sonnet(self):
        assert llm_query.detect_provider('claude-3-5-sonnet-20241022') == 'anthropic'

    def test_anthropic_claude_3_opus(self):
        assert llm_query.detect_provider('claude-3-opus-20240229') == 'anthropic'

    def test_anthropic_claude_3_haiku(self):
        assert llm_query.detect_provider('claude-3-haiku-20240307') == 'anthropic'

    def test_google_gemini_25_flash(self):
        assert llm_query.detect_provider('gemini-2.5-flash') == 'google'

    def test_google_gemini_25_pro(self):
        assert llm_query.detect_provider('gemini-2.5-pro') == 'google'

    def test_google_gemini_20_flash(self):
        assert llm_query.detect_provider('gemini-2.0-flash') == 'google'

    def test_case_insensitive(self):
        assert llm_query.detect_provider('GPT-4o') == 'openai'
        assert llm_query.detect_provider('Claude-3-5-sonnet') == 'anthropic'
        assert llm_query.detect_provider('GEMINI-2.5-flash') == 'google'

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            llm_query.detect_provider('unknown-model')

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            llm_query.detect_provider('')


class TestGetApiKey:
    """Tests for API key retrieval from environment."""

    def test_openai_key_present(self):
        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            assert llm_query.get_api_key('openai') == 'test-key'

    def test_anthropic_key_present(self):
        with mock.patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            assert llm_query.get_api_key('anthropic') == 'test-key'

    def test_google_key_present(self):
        with mock.patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'}):
            assert llm_query.get_api_key('google') == 'test-key'

    def test_openai_key_missing_exits(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            # Ensure key is not set
            os.environ.pop('OPENAI_API_KEY', None)
            with pytest.raises(SystemExit) as exc_info:
                llm_query.get_api_key('openai')
            assert exc_info.value.code == llm_query.EXIT_MISSING_API_KEY

    def test_anthropic_key_missing_exits(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop('ANTHROPIC_API_KEY', None)
            with pytest.raises(SystemExit) as exc_info:
                llm_query.get_api_key('anthropic')
            assert exc_info.value.code == llm_query.EXIT_MISSING_API_KEY

    def test_google_key_missing_exits(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop('GOOGLE_API_KEY', None)
            with pytest.raises(SystemExit) as exc_info:
                llm_query.get_api_key('google')
            assert exc_info.value.code == llm_query.EXIT_MISSING_API_KEY

    def test_unknown_provider_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            llm_query.get_api_key('unknown')
        assert exc_info.value.code == llm_query.EXIT_INVALID_ARGS


class TestLoadJsonSchema:
    """Tests for JSON schema file loading."""

    def test_valid_schema_loads(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            f.flush()
            loaded = llm_query.load_json_schema(f.name)
            assert loaded == schema
            os.unlink(f.name)

    def test_file_not_found_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            llm_query.load_json_schema('/nonexistent/schema.json')
        assert exc_info.value.code == llm_query.EXIT_INVALID_ARGS

    def test_invalid_json_exits(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('not valid json {')
            f.flush()
            with pytest.raises(SystemExit) as exc_info:
                llm_query.load_json_schema(f.name)
            assert exc_info.value.code == llm_query.EXIT_INVALID_ARGS
            os.unlink(f.name)


class TestValidateJsonOutput:
    """Tests for JSON output validation against schema."""

    def test_valid_object_passes(self):
        schema = {"type": "object"}
        output = '{"key": "value"}'
        assert llm_query.validate_json_output(output, schema) is True

    def test_valid_array_passes(self):
        schema = {"type": "array"}
        output = '[1, 2, 3]'
        assert llm_query.validate_json_output(output, schema) is True

    def test_valid_string_passes(self):
        schema = {"type": "string"}
        output = '"hello"'
        assert llm_query.validate_json_output(output, schema) is True

    def test_valid_number_passes(self):
        schema = {"type": "number"}
        output = '42.5'
        assert llm_query.validate_json_output(output, schema) is True

    def test_valid_integer_passes(self):
        schema = {"type": "integer"}
        output = '42'
        assert llm_query.validate_json_output(output, schema) is True

    def test_valid_boolean_passes(self):
        schema = {"type": "boolean"}
        output = 'true'
        assert llm_query.validate_json_output(output, schema) is True

    def test_invalid_json_exits(self):
        schema = {"type": "object"}
        output = 'not json'
        with pytest.raises(SystemExit) as exc_info:
            llm_query.validate_json_output(output, schema)
        assert exc_info.value.code == llm_query.EXIT_VALIDATION_ERROR

    def test_type_mismatch_exits(self):
        schema = {"type": "object"}
        output = '[1, 2, 3]'  # Array, not object
        with pytest.raises(SystemExit) as exc_info:
            llm_query.validate_json_output(output, schema)
        assert exc_info.value.code == llm_query.EXIT_VALIDATION_ERROR

    def test_missing_required_property_exits(self):
        schema = {"type": "object", "required": ["name", "age"]}
        output = '{"name": "John"}'  # Missing 'age'
        with pytest.raises(SystemExit) as exc_info:
            llm_query.validate_json_output(output, schema)
        assert exc_info.value.code == llm_query.EXIT_VALIDATION_ERROR

    def test_all_required_properties_present_passes(self):
        schema = {"type": "object", "required": ["name", "age"]}
        output = '{"name": "John", "age": 30}'
        assert llm_query.validate_json_output(output, schema) is True


class TestCreateParser:
    """Tests for argument parser configuration."""

    def test_parser_creation(self):
        parser = llm_query.create_parser()
        assert parser is not None
        assert parser.prog == 'llm_query'

    def test_required_model_and_prompt(self):
        parser = llm_query.create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # Missing required args

    def test_model_short_flag(self):
        parser = llm_query.create_parser()
        args = parser.parse_args(['-m', 'gpt-4o', '-p', 'test prompt'])
        assert args.model == 'gpt-4o'
        assert args.prompt == 'test prompt'

    def test_model_long_flag(self):
        parser = llm_query.create_parser()
        args = parser.parse_args(['--model', 'claude-3-5-sonnet', '--prompt', 'test'])
        assert args.model == 'claude-3-5-sonnet'

    def test_file_input(self):
        parser = llm_query.create_parser()
        args = parser.parse_args(['-m', 'gpt-4o', '-p', 'test', '-f', 'input.txt'])
        assert args.file == 'input.txt'

    def test_stdin_flag(self):
        parser = llm_query.create_parser()
        args = parser.parse_args(['-m', 'gpt-4o', '-p', 'test', '--stdin'])
        assert args.stdin is True

    def test_file_and_stdin_mutually_exclusive(self):
        parser = llm_query.create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['-m', 'gpt-4o', '-p', 'test', '-f', 'file.txt', '--stdin'])

    def test_system_prompt(self):
        parser = llm_query.create_parser()
        args = parser.parse_args(['-m', 'gpt-4o', '-p', 'test', '--system', 'You are helpful'])
        assert args.system_prompt == 'You are helpful'

    def test_system_prompt_short_flag(self):
        parser = llm_query.create_parser()
        args = parser.parse_args(['-m', 'gpt-4o', '-p', 'test', '-s', 'You are helpful'])
        assert args.system_prompt == 'You are helpful'

    def test_system_file(self):
        parser = llm_query.create_parser()
        args = parser.parse_args(['-m', 'gpt-4o', '-p', 'test', '--system-file', 'sys.txt'])
        assert args.system_file == 'sys.txt'

    def test_system_and_system_file_mutually_exclusive(self):
        parser = llm_query.create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['-m', 'gpt-4o', '-p', 'test', '--system', 'x', '--system-file', 'y'])

    def test_json_schema(self):
        parser = llm_query.create_parser()
        args = parser.parse_args(['-m', 'gpt-4o', '-p', 'test', '--json-schema', 'schema.json'])
        assert args.json_schema == 'schema.json'

    def test_verbose_flag(self):
        parser = llm_query.create_parser()
        args = parser.parse_args(['-m', 'gpt-4o', '-p', 'test', '-v'])
        assert args.verbose is True

    def test_temperature(self):
        parser = llm_query.create_parser()
        args = parser.parse_args(['-m', 'gpt-4o', '-p', 'test', '--temperature', '0.7'])
        assert args.temperature == 0.7

    def test_max_tokens(self):
        parser = llm_query.create_parser()
        args = parser.parse_args(['-m', 'gpt-4o', '-p', 'test', '--max-tokens', '1000'])
        assert args.max_tokens == 1000


class TestStreamOpenAI:
    """Tests for OpenAI streaming with mocked client."""

    @mock.patch('llm_query.OpenAI')
    def test_basic_streaming(self, mock_openai_class):
        """Test basic streaming response from OpenAI."""
        # Setup mock
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client

        # Create mock chunks
        mock_chunk1 = mock.Mock()
        mock_chunk1.choices = [mock.Mock()]
        mock_chunk1.choices[0].delta.content = 'Hello'

        mock_chunk2 = mock.Mock()
        mock_chunk2.choices = [mock.Mock()]
        mock_chunk2.choices[0].delta.content = ' World'

        mock_chunk3 = mock.Mock()
        mock_chunk3.choices = [mock.Mock()]
        mock_chunk3.choices[0].delta.content = None  # End of stream

        mock_client.chat.completions.create.return_value = iter([
            mock_chunk1, mock_chunk2, mock_chunk3
        ])

        # Run the function
        result = list(llm_query.stream_openai(
            model='gpt-4o',
            prompt='Say hello',
            content=None,
            system_prompt=None,
            json_schema=None,
            api_key='test-key',
        ))

        assert result == ['Hello', ' World']

    @mock.patch('llm_query.OpenAI')
    def test_with_system_prompt(self, mock_openai_class):
        """Test that system prompt is included in messages."""
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter([])

        list(llm_query.stream_openai(
            model='gpt-4o',
            prompt='test',
            content=None,
            system_prompt='You are helpful',
            json_schema=None,
            api_key='test-key',
        ))

        # Check the call arguments
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == 'You are helpful'

    @mock.patch('llm_query.OpenAI')
    def test_with_json_schema(self, mock_openai_class):
        """Test that JSON schema is passed correctly."""
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter([])

        schema = {"type": "object"}
        list(llm_query.stream_openai(
            model='gpt-4o',
            prompt='test',
            content=None,
            system_prompt=None,
            json_schema=schema,
            api_key='test-key',
        ))

        call_args = mock_client.chat.completions.create.call_args
        response_format = call_args.kwargs['response_format']
        assert response_format['type'] == 'json_schema'
        assert response_format['json_schema']['schema'] == schema


class TestStreamAnthropic:
    """Tests for Anthropic streaming with mocked client."""

    @mock.patch('llm_query.anthropic')
    def test_basic_streaming(self, mock_anthropic_module):
        """Test basic streaming response from Anthropic."""
        mock_client = mock.Mock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        # Create a mock context manager for streaming
        mock_stream = mock.Mock()
        mock_stream.__enter__ = mock.Mock(return_value=mock_stream)
        mock_stream.__exit__ = mock.Mock(return_value=False)
        mock_stream.text_stream = iter(['Hello', ' World'])

        mock_client.messages.stream.return_value = mock_stream

        result = list(llm_query.stream_anthropic(
            model='claude-3-5-sonnet-20241022',
            prompt='Say hello',
            content=None,
            system_prompt=None,
            json_schema=None,
            api_key='test-key',
        ))

        assert result == ['Hello', ' World']

    @mock.patch('llm_query.anthropic')
    def test_with_content(self, mock_anthropic_module):
        """Test that content is appended to prompt."""
        mock_client = mock.Mock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        mock_stream = mock.Mock()
        mock_stream.__enter__ = mock.Mock(return_value=mock_stream)
        mock_stream.__exit__ = mock.Mock(return_value=False)
        mock_stream.text_stream = iter([])

        mock_client.messages.stream.return_value = mock_stream

        list(llm_query.stream_anthropic(
            model='claude-3-5-sonnet',
            prompt='Analyze this:',
            content='Some content',
            system_prompt=None,
            json_schema=None,
            api_key='test-key',
        ))

        call_args = mock_client.messages.stream.call_args
        messages = call_args.kwargs['messages']
        assert 'Analyze this:\n\nSome content' in messages[0]['content']


class TestStreamGoogle:
    """Tests for Google GenAI streaming with mocked client."""

    @mock.patch('llm_query.genai')
    def test_basic_streaming(self, mock_genai_module):
        """Test basic streaming response from Google GenAI."""
        mock_client = mock.Mock()
        mock_genai_module.Client.return_value = mock_client

        # Create mock chunks
        mock_chunk1 = mock.Mock()
        mock_chunk1.text = 'Hello'

        mock_chunk2 = mock.Mock()
        mock_chunk2.text = ' World'

        mock_client.models.generate_content_stream.return_value = iter([
            mock_chunk1, mock_chunk2
        ])

        result = list(llm_query.stream_google(
            model='gemini-2.5-flash',
            prompt='Say hello',
            content=None,
            system_prompt=None,
            json_schema=None,
            api_key='test-key',
        ))

        assert result == ['Hello', ' World']

    @mock.patch('llm_query.genai')
    def test_with_json_schema(self, mock_genai_module):
        """Test that JSON schema config is passed correctly."""
        mock_client = mock.Mock()
        mock_genai_module.Client.return_value = mock_client
        mock_client.models.generate_content_stream.return_value = iter([])

        schema = {"type": "object"}
        list(llm_query.stream_google(
            model='gemini-2.5-flash',
            prompt='test',
            content=None,
            system_prompt=None,
            json_schema=schema,
            api_key='test-key',
        ))

        call_args = mock_client.models.generate_content_stream.call_args
        config = call_args.kwargs['config']
        assert config['response_mime_type'] == 'application/json'
        assert config['response_schema'] == schema


class TestMain:
    """Tests for the main CLI entry point."""

    @mock.patch('llm_query.query_llm')
    def test_basic_invocation(self, mock_query):
        """Test basic CLI invocation."""
        mock_query.return_value = 'response'

        result = llm_query.main(['-m', 'gpt-4o', '-p', 'test prompt'])
        assert result == llm_query.EXIT_SUCCESS

    @mock.patch('llm_query.query_llm')
    def test_file_input(self, mock_query):
        """Test file input reading."""
        mock_query.return_value = 'response'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('file content')
            f.flush()

            result = llm_query.main(['-m', 'gpt-4o', '-p', 'analyze', '-f', f.name])
            assert result == llm_query.EXIT_SUCCESS

            # Check that content was passed
            call_args = mock_query.call_args
            assert call_args.kwargs['content'] == 'file content'

            os.unlink(f.name)

    def test_file_not_found(self):
        """Test error when input file not found."""
        result = llm_query.main(['-m', 'gpt-4o', '-p', 'test', '-f', '/nonexistent.txt'])
        assert result == llm_query.EXIT_INVALID_ARGS

    @mock.patch('llm_query.query_llm')
    def test_system_prompt_from_file(self, mock_query):
        """Test loading system prompt from file."""
        mock_query.return_value = 'response'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('You are a helpful assistant')
            f.flush()

            result = llm_query.main([
                '-m', 'gpt-4o', '-p', 'test', '--system-file', f.name
            ])
            assert result == llm_query.EXIT_SUCCESS

            call_args = mock_query.call_args
            assert call_args.kwargs['system_prompt'] == 'You are a helpful assistant'

            os.unlink(f.name)

    def test_system_file_not_found(self):
        """Test error when system prompt file not found."""
        result = llm_query.main([
            '-m', 'gpt-4o', '-p', 'test', '--system-file', '/nonexistent.txt'
        ])
        assert result == llm_query.EXIT_INVALID_ARGS

    @mock.patch('llm_query.query_llm')
    @mock.patch('llm_query.validate_json_output')
    @mock.patch('llm_query.load_json_schema')
    def test_json_schema_validation(self, mock_load_schema, mock_validate, mock_query):
        """Test JSON schema loading and validation."""
        mock_load_schema.return_value = {"type": "object"}
        mock_query.return_value = '{"key": "value"}'
        mock_validate.return_value = True

        result = llm_query.main([
            '-m', 'gpt-4o', '-p', 'test', '--json-schema', 'schema.json'
        ])
        assert result == llm_query.EXIT_SUCCESS

        mock_load_schema.assert_called_once_with('schema.json')
        mock_validate.assert_called_once()

    @mock.patch('llm_query.query_llm')
    def test_verbose_mode(self, mock_query, capsys):
        """Test verbose output."""
        mock_query.return_value = 'response'

        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}):
            llm_query.main(['-m', 'gpt-4o', '-p', 'test', '-v'])

        captured = capsys.readouterr()
        assert 'Provider: openai' in captured.err
        assert 'Model: gpt-4o' in captured.err


class TestExitCodes:
    """Tests verifying correct exit codes for various error conditions."""

    def test_missing_api_key_exit_code(self):
        """Verify EXIT_MISSING_API_KEY is returned when key is missing."""
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop('OPENAI_API_KEY', None)
            with pytest.raises(SystemExit) as exc_info:
                llm_query.get_api_key('openai')
            assert exc_info.value.code == 1  # EXIT_MISSING_API_KEY

    def test_invalid_args_exit_code(self):
        """Verify EXIT_INVALID_ARGS is returned for bad arguments."""
        result = llm_query.main(['-m', 'gpt-4o', '-p', 'test', '-f', '/nonexistent.txt'])
        assert result == 4  # EXIT_INVALID_ARGS

    def test_validation_error_exit_code(self):
        """Verify EXIT_VALIDATION_ERROR for schema validation failures."""
        with pytest.raises(SystemExit) as exc_info:
            llm_query.validate_json_output('not json', {"type": "object"})
        assert exc_info.value.code == 3  # EXIT_VALIDATION_ERROR


class TestIntegration:
    """Integration tests that verify complete workflows."""

    @mock.patch('llm_query.OpenAI')
    @mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_full_openai_workflow(self, mock_openai_class, capsys):
        """Test complete workflow with OpenAI."""
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client

        mock_chunk = mock.Mock()
        mock_chunk.choices = [mock.Mock()]
        mock_chunk.choices[0].delta.content = 'Test response'

        mock_client.chat.completions.create.return_value = iter([mock_chunk])

        result = llm_query.main(['-m', 'gpt-4o', '-p', 'test prompt'])

        assert result == llm_query.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'Test response' in captured.out

    @mock.patch('llm_query.anthropic')
    @mock.patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'})
    def test_full_anthropic_workflow(self, mock_anthropic_module, capsys):
        """Test complete workflow with Anthropic."""
        mock_client = mock.Mock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        mock_stream = mock.Mock()
        mock_stream.__enter__ = mock.Mock(return_value=mock_stream)
        mock_stream.__exit__ = mock.Mock(return_value=False)
        mock_stream.text_stream = iter(['Anthropic response'])

        mock_client.messages.stream.return_value = mock_stream

        result = llm_query.main(['-m', 'claude-3-5-sonnet', '-p', 'test'])

        assert result == llm_query.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'Anthropic response' in captured.out

    @mock.patch('llm_query.genai')
    @mock.patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'})
    def test_full_google_workflow(self, mock_genai_module, capsys):
        """Test complete workflow with Google GenAI."""
        mock_client = mock.Mock()
        mock_genai_module.Client.return_value = mock_client

        mock_chunk = mock.Mock()
        mock_chunk.text = 'Google response'

        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])

        result = llm_query.main(['-m', 'gemini-2.5-flash', '-p', 'test'])

        assert result == llm_query.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'Google response' in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
