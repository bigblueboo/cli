#!/usr/bin/env python3
"""
Comprehensive tests for the Prompt Manager CLI tool.

This test suite covers all CRUD operations, variable substitution,
import/export functionality, and error handling.

Run with: pytest test_prompt_manager.py -v
Or:       python -m pytest test_prompt_manager.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the module under test
import prompt_manager as pm


@pytest.fixture
def temp_storage(tmp_path):
    """
    Fixture that creates a temporary storage directory for tests.

    Sets PROMPT_MANAGER_DIR to the temp directory and cleans up after.
    """
    storage_dir = tmp_path / '.prompt_manager'
    storage_dir.mkdir()

    with patch.dict(os.environ, {'PROMPT_MANAGER_DIR': str(storage_dir)}):
        yield storage_dir


@pytest.fixture
def sample_prompts(temp_storage):
    """
    Fixture that creates sample prompts in the storage.
    """
    prompts = {
        "code-review": {
            "template": "Review this {{language}} code:\n{{code}}\nFocus on: {{focus}}",
            "category": "coding",
            "tags": ["review", "quality"],
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        },
        "summarize": {
            "template": "Summarize: {{text}}",
            "category": "writing",
            "tags": ["short", "utility"],
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        },
        "explain": {
            "template": "Explain {{topic}} in simple terms.",
            "category": "writing",
            "tags": ["educational"],
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        }
    }

    prompts_file = temp_storage / 'prompts.json'
    prompts_file.write_text(json.dumps(prompts))

    return prompts


class TestStorageFunctions:
    """Tests for storage-related functions."""

    def test_get_storage_dir_default(self):
        """Test default storage directory is ~/.prompt_manager/."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove PROMPT_MANAGER_DIR if it exists
            os.environ.pop('PROMPT_MANAGER_DIR', None)
            result = pm.get_storage_dir()
            assert result == Path.home() / '.prompt_manager'

    def test_get_storage_dir_custom(self):
        """Test custom storage directory from environment variable."""
        with patch.dict(os.environ, {'PROMPT_MANAGER_DIR': '/custom/path'}):
            result = pm.get_storage_dir()
            assert result == Path('/custom/path')

    def test_get_storage_dir_with_tilde(self):
        """Test storage directory with ~ expansion."""
        with patch.dict(os.environ, {'PROMPT_MANAGER_DIR': '~/my_prompts'}):
            result = pm.get_storage_dir()
            assert result == Path.home() / 'my_prompts'

    def test_ensure_storage_exists(self, temp_storage):
        """Test that ensure_storage_exists creates directory and file."""
        # Remove the storage dir to test creation
        prompts_file = temp_storage / 'prompts.json'
        if prompts_file.exists():
            prompts_file.unlink()

        assert pm.ensure_storage_exists()
        assert prompts_file.exists()

    def test_load_prompts_empty(self, temp_storage):
        """Test loading from empty storage."""
        result = pm.load_prompts()
        assert result == {}

    def test_load_prompts_with_data(self, temp_storage, sample_prompts):
        """Test loading prompts with existing data."""
        result = pm.load_prompts()
        assert 'code-review' in result
        assert 'summarize' in result
        assert result['code-review']['category'] == 'coding'

    def test_load_prompts_invalid_json(self, temp_storage):
        """Test loading prompts with invalid JSON."""
        prompts_file = temp_storage / 'prompts.json'
        prompts_file.write_text('invalid json {{{')

        result = pm.load_prompts()
        assert result is None

    def test_save_prompts(self, temp_storage):
        """Test saving prompts."""
        prompts = {'test': {'template': 'Hello', 'category': '', 'tags': []}}

        assert pm.save_prompts(prompts)

        prompts_file = temp_storage / 'prompts.json'
        loaded = json.loads(prompts_file.read_text())
        assert 'test' in loaded


class TestAddCommand:
    """Tests for the add command."""

    def test_add_basic(self, temp_storage):
        """Test adding a basic prompt."""
        args = create_args(name='test', template='Hello {{name}}', category=None, tags=None)

        result = pm.cmd_add(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert 'test' in prompts
        assert prompts['test']['template'] == 'Hello {{name}}'

    def test_add_with_category(self, temp_storage):
        """Test adding a prompt with category."""
        args = create_args(name='test', template='Hello', category='greeting', tags=None)

        result = pm.cmd_add(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert prompts['test']['category'] == 'greeting'

    def test_add_with_tags(self, temp_storage):
        """Test adding a prompt with tags."""
        args = create_args(name='test', template='Hello', category=None, tags='tag1, tag2, tag3')

        result = pm.cmd_add(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert prompts['test']['tags'] == ['tag1', 'tag2', 'tag3']

    def test_add_duplicate(self, temp_storage, sample_prompts):
        """Test adding a duplicate prompt fails."""
        args = create_args(name='code-review', template='New template', category=None, tags=None)

        result = pm.cmd_add(args)

        assert result == pm.EXIT_INVALID_ARGS


class TestListCommand:
    """Tests for the list command."""

    def test_list_all(self, temp_storage, sample_prompts, capsys):
        """Test listing all prompts."""
        args = create_args(category=None, tag=None)

        result = pm.cmd_list(args)

        assert result == pm.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'code-review' in captured.out
        assert 'summarize' in captured.out
        assert 'explain' in captured.out
        assert 'Total: 3' in captured.out

    def test_list_empty(self, temp_storage, capsys):
        """Test listing when no prompts exist."""
        args = create_args(category=None, tag=None)

        result = pm.cmd_list(args)

        assert result == pm.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'No prompts found.' in captured.out

    def test_list_by_category(self, temp_storage, sample_prompts, capsys):
        """Test listing prompts by category."""
        args = create_args(category='writing', tag=None)

        result = pm.cmd_list(args)

        assert result == pm.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'summarize' in captured.out
        assert 'explain' in captured.out
        assert 'code-review' not in captured.out
        assert 'Total: 2' in captured.out

    def test_list_by_tag(self, temp_storage, sample_prompts, capsys):
        """Test listing prompts by tag."""
        args = create_args(category=None, tag='utility')

        result = pm.cmd_list(args)

        assert result == pm.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'summarize' in captured.out
        assert 'code-review' not in captured.out

    def test_list_nonexistent_category(self, temp_storage, sample_prompts, capsys):
        """Test listing with nonexistent category."""
        args = create_args(category='nonexistent', tag=None)

        result = pm.cmd_list(args)

        assert result == pm.EXIT_NOT_FOUND


class TestGetCommand:
    """Tests for the get command."""

    def test_get_existing(self, temp_storage, sample_prompts, capsys):
        """Test getting an existing prompt."""
        args = create_args(name='code-review')

        result = pm.cmd_get(args)

        assert result == pm.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'Name: code-review' in captured.out
        assert 'Category: coding' in captured.out
        assert 'review, quality' in captured.out
        assert '{{language}}' in captured.out

    def test_get_nonexistent(self, temp_storage, capsys):
        """Test getting a nonexistent prompt."""
        args = create_args(name='nonexistent')

        result = pm.cmd_get(args)

        assert result == pm.EXIT_NOT_FOUND
        captured = capsys.readouterr()
        assert 'not found' in captured.err


class TestUpdateCommand:
    """Tests for the update command."""

    def test_update_template(self, temp_storage, sample_prompts):
        """Test updating prompt template."""
        args = create_args(name='code-review', template='New template {{x}}', category=None, tags=None)

        result = pm.cmd_update(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert prompts['code-review']['template'] == 'New template {{x}}'

    def test_update_category(self, temp_storage, sample_prompts):
        """Test updating prompt category."""
        args = create_args(name='code-review', template=None, category='new-category', tags=None)

        result = pm.cmd_update(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert prompts['code-review']['category'] == 'new-category'

    def test_update_tags(self, temp_storage, sample_prompts):
        """Test updating prompt tags."""
        args = create_args(name='code-review', template=None, category=None, tags='new,tags')

        result = pm.cmd_update(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert prompts['code-review']['tags'] == ['new', 'tags']

    def test_update_clear_tags(self, temp_storage, sample_prompts):
        """Test clearing prompt tags."""
        args = create_args(name='code-review', template=None, category=None, tags='')

        result = pm.cmd_update(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert prompts['code-review']['tags'] == []

    def test_update_nonexistent(self, temp_storage):
        """Test updating a nonexistent prompt."""
        args = create_args(name='nonexistent', template='test', category=None, tags=None)

        result = pm.cmd_update(args)

        assert result == pm.EXIT_NOT_FOUND

    def test_update_no_fields(self, temp_storage, sample_prompts, capsys):
        """Test updating with no fields specified."""
        args = create_args(name='code-review', template=None, category=None, tags=None)

        result = pm.cmd_update(args)

        assert result == pm.EXIT_INVALID_ARGS


class TestDeleteCommand:
    """Tests for the delete command."""

    def test_delete_existing(self, temp_storage, sample_prompts):
        """Test deleting an existing prompt."""
        args = create_args(name='code-review')

        result = pm.cmd_delete(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert 'code-review' not in prompts

    def test_delete_nonexistent(self, temp_storage):
        """Test deleting a nonexistent prompt."""
        args = create_args(name='nonexistent')

        result = pm.cmd_delete(args)

        assert result == pm.EXIT_NOT_FOUND


class TestRenderCommand:
    """Tests for the render command."""

    def test_render_basic(self, temp_storage, sample_prompts, capsys):
        """Test rendering a prompt with variables."""
        args = create_args(
            name='code-review',
            vars='{"language": "Python", "code": "def foo():", "focus": "bugs"}'
        )

        result = pm.cmd_render(args)

        assert result == pm.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'Review this Python code:' in captured.out
        assert 'def foo():' in captured.out
        assert 'Focus on: bugs' in captured.out

    def test_render_no_vars(self, temp_storage, capsys):
        """Test rendering a prompt without variables."""
        # Add a prompt without variables
        prompts = {'simple': {'template': 'Hello World', 'category': '', 'tags': []}}
        pm.save_prompts(prompts)

        args = create_args(name='simple', vars=None)

        result = pm.cmd_render(args)

        assert result == pm.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'Hello World' in captured.out

    def test_render_missing_variable(self, temp_storage, sample_prompts, capsys):
        """Test rendering with missing variable."""
        args = create_args(name='code-review', vars='{"language": "Python"}')

        result = pm.cmd_render(args)

        assert result == pm.EXIT_INVALID_ARGS

    def test_render_invalid_json(self, temp_storage, sample_prompts, capsys):
        """Test rendering with invalid JSON vars."""
        args = create_args(name='code-review', vars='not valid json')

        result = pm.cmd_render(args)

        assert result == pm.EXIT_INVALID_ARGS

    def test_render_nonexistent_prompt(self, temp_storage):
        """Test rendering a nonexistent prompt."""
        args = create_args(name='nonexistent', vars='{}')

        result = pm.cmd_render(args)

        assert result == pm.EXIT_NOT_FOUND


class TestExportCommand:
    """Tests for the export command."""

    def test_export_json(self, temp_storage, sample_prompts, capsys):
        """Test exporting prompts as JSON."""
        args = create_args(format='json')

        result = pm.cmd_export(args)

        assert result == pm.EXIT_SUCCESS
        captured = capsys.readouterr()
        exported = json.loads(captured.out)
        assert 'code-review' in exported
        assert 'summarize' in exported

    def test_export_yaml(self, temp_storage, sample_prompts, capsys):
        """Test exporting prompts as YAML."""
        pytest.importorskip('yaml')
        import yaml

        args = create_args(format='yaml')

        result = pm.cmd_export(args)

        assert result == pm.EXIT_SUCCESS
        captured = capsys.readouterr()
        exported = yaml.safe_load(captured.out)
        assert 'code-review' in exported
        assert 'summarize' in exported

    def test_export_invalid_format(self, temp_storage, sample_prompts, capsys):
        """Test exporting with invalid format."""
        args = create_args(format='invalid')

        result = pm.cmd_export(args)

        assert result == pm.EXIT_INVALID_ARGS


class TestImportCommand:
    """Tests for the import command."""

    def test_import_json(self, temp_storage, tmp_path):
        """Test importing prompts from JSON file."""
        import_data = {
            "imported-prompt": {
                "template": "Imported: {{x}}",
                "category": "imported",
                "tags": ["test"]
            }
        }

        import_file = tmp_path / 'import.json'
        import_file.write_text(json.dumps(import_data))

        args = create_args(file=str(import_file), no_overwrite=False)

        result = pm.cmd_import(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert 'imported-prompt' in prompts

    def test_import_yaml(self, temp_storage, tmp_path):
        """Test importing prompts from YAML file."""
        pytest.importorskip('yaml')
        import yaml

        import_data = {
            "yaml-prompt": {
                "template": "YAML: {{y}}",
                "category": "yaml",
                "tags": ["yaml-tag"]
            }
        }

        import_file = tmp_path / 'import.yaml'
        import_file.write_text(yaml.dump(import_data))

        args = create_args(file=str(import_file), no_overwrite=False)

        result = pm.cmd_import(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert 'yaml-prompt' in prompts

    def test_import_overwrite(self, temp_storage, sample_prompts, tmp_path):
        """Test importing overwrites existing prompts by default."""
        import_data = {
            "code-review": {
                "template": "OVERWRITTEN",
                "category": "overwritten",
                "tags": []
            }
        }

        import_file = tmp_path / 'import.json'
        import_file.write_text(json.dumps(import_data))

        args = create_args(file=str(import_file), no_overwrite=False)

        result = pm.cmd_import(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert prompts['code-review']['template'] == 'OVERWRITTEN'

    def test_import_no_overwrite(self, temp_storage, sample_prompts, tmp_path, capsys):
        """Test importing with --no-overwrite flag."""
        original_template = sample_prompts['code-review']['template']

        import_data = {
            "code-review": {
                "template": "SHOULD NOT OVERWRITE",
                "category": "test",
                "tags": []
            }
        }

        import_file = tmp_path / 'import.json'
        import_file.write_text(json.dumps(import_data))

        args = create_args(file=str(import_file), no_overwrite=True)

        result = pm.cmd_import(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert prompts['code-review']['template'] == original_template
        captured = capsys.readouterr()
        assert '1 skipped' in captured.out

    def test_import_nonexistent_file(self, temp_storage):
        """Test importing from nonexistent file."""
        args = create_args(file='/nonexistent/file.json', no_overwrite=False)

        result = pm.cmd_import(args)

        assert result == pm.EXIT_FILE_ERROR

    def test_import_invalid_format(self, temp_storage, tmp_path):
        """Test importing invalid file format."""
        import_file = tmp_path / 'invalid.txt'
        import_file.write_text('this is not valid json or yaml {{{{')

        args = create_args(file=str(import_file), no_overwrite=False)

        result = pm.cmd_import(args)

        assert result == pm.EXIT_INVALID_ARGS


class TestParserAndMain:
    """Tests for argument parsing and main function."""

    def test_parser_creation(self):
        """Test that parser is created correctly."""
        parser = pm.create_parser()
        assert parser is not None
        assert parser.prog == 'prompt_manager'

    def test_main_no_args(self, capsys):
        """Test main with no arguments shows help."""
        with patch.object(sys, 'argv', ['prompt_manager']):
            result = pm.main()

        assert result == pm.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'usage:' in captured.out.lower() or 'Available commands' in captured.out

    def test_main_add_command(self, temp_storage):
        """Test main with add command."""
        with patch.object(sys, 'argv', [
            'prompt_manager', 'add', 'test-prompt',
            '--template', 'Test template {{var}}'
        ]):
            result = pm.main()

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert 'test-prompt' in prompts


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_template(self, temp_storage):
        """Test adding a prompt with empty template."""
        args = create_args(name='empty', template='', category=None, tags=None)

        result = pm.cmd_add(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert prompts['empty']['template'] == ''

    def test_special_characters_in_name(self, temp_storage):
        """Test prompt name with special characters."""
        args = create_args(name='my-prompt_v2.0', template='Test', category=None, tags=None)

        result = pm.cmd_add(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert 'my-prompt_v2.0' in prompts

    def test_unicode_in_template(self, temp_storage, capsys):
        """Test unicode characters in template."""
        args = create_args(name='unicode', template='Hello {{emoji}}', category=None, tags=None)
        pm.cmd_add(args)

        render_args = create_args(name='unicode', vars='{"emoji": "World"}')
        result = pm.cmd_render(render_args)

        assert result == pm.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'Hello World' in captured.out

    def test_multiline_template(self, temp_storage, capsys):
        """Test multiline template."""
        template = "Line 1: {{a}}\nLine 2: {{b}}\nLine 3: {{c}}"
        args = create_args(name='multiline', template=template, category=None, tags=None)
        pm.cmd_add(args)

        render_args = create_args(name='multiline', vars='{"a": "A", "b": "B", "c": "C"}')
        result = pm.cmd_render(render_args)

        assert result == pm.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'Line 1: A' in captured.out
        assert 'Line 2: B' in captured.out
        assert 'Line 3: C' in captured.out

    def test_whitespace_in_tags(self, temp_storage):
        """Test tags with extra whitespace are trimmed."""
        args = create_args(name='test', template='Test', category=None, tags='  tag1  ,  tag2  ,  tag3  ')

        result = pm.cmd_add(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert prompts['test']['tags'] == ['tag1', 'tag2', 'tag3']

    def test_empty_tags_filtered(self, temp_storage):
        """Test empty tags are filtered out."""
        args = create_args(name='test', template='Test', category=None, tags='tag1,,tag2,,,tag3,')

        result = pm.cmd_add(args)

        assert result == pm.EXIT_SUCCESS
        prompts = pm.load_prompts()
        assert prompts['test']['tags'] == ['tag1', 'tag2', 'tag3']


# Helper function to create mock args
def create_args(**kwargs):
    """
    Create a mock argparse.Namespace object with the given attributes.

    Args:
        **kwargs: Attribute names and values to set on the Namespace.

    Returns:
        argparse.Namespace: Mock args object.
    """
    import argparse
    args = argparse.Namespace()
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
