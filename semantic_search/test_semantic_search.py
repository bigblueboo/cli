#!/usr/bin/env python3
"""
Comprehensive tests for semantic_search CLI tool.

This module provides unit tests and integration tests for the semantic_search
tool, using mocking to avoid dependencies on external libraries during testing.

Run tests with:
    pytest test_semantic_search.py -v
    pytest test_semantic_search.py -v --cov=semantic_search
"""

import json
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Import the module to test
import semantic_search as ss


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def sample_files(temp_dir):
    """Create sample files for testing."""
    # Create source directory
    src_dir = temp_dir / 'src'
    src_dir.mkdir()

    # Create Python files
    (src_dir / 'main.py').write_text('''
"""Main application module."""

def main():
    """Main entry point for the application."""
    print("Hello, World!")
    initialize_database()
    start_server()

def initialize_database():
    """Initialize the database connection."""
    # Database initialization logic
    connection = create_connection()
    return connection

def start_server():
    """Start the web server."""
    # Server startup logic
    pass
''')

    (src_dir / 'auth.py').write_text('''
"""Authentication middleware and utilities."""

class AuthMiddleware:
    """Middleware for handling authentication."""

    def __init__(self, secret_key):
        self.secret_key = secret_key

    def authenticate(self, token):
        """Authenticate a user token."""
        # Token validation logic
        return self.validate_token(token)

    def validate_token(self, token):
        """Validate the JWT token."""
        # JWT validation
        return True

def create_token(user_id):
    """Create a new authentication token."""
    return f"token_{user_id}"
''')

    # Create docs directory
    docs_dir = temp_dir / 'docs'
    docs_dir.mkdir()

    (docs_dir / 'README.md').write_text('''
# Project Documentation

## Getting Started

This guide explains how to get started with the project.

### Installation

Run `pip install -r requirements.txt` to install dependencies.

### Configuration

Set the following environment variables:
- `DATABASE_URL`: Connection string for the database
- `SECRET_KEY`: Secret key for authentication

## Error Handling

The application handles errors gracefully. All exceptions are logged
and appropriate HTTP status codes are returned.

### Common Errors

- 401 Unauthorized: Invalid authentication token
- 404 Not Found: Resource not found
- 500 Internal Server Error: Unexpected error
''')

    (docs_dir / 'api.md').write_text('''
# API Reference

## Endpoints

### GET /api/users

Returns a list of all users.

### POST /api/auth/login

Authenticate a user and return a token.

Request body:
```json
{
  "username": "user",
  "password": "pass"
}
```

### DELETE /api/users/:id

Delete a user by ID. Requires admin authentication.
''')

    # Create a binary file (should be ignored)
    (temp_dir / 'image.png').write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00')

    return temp_dir


@pytest.fixture
def mock_index_dir(temp_dir):
    """Create a mock index directory."""
    index_dir = temp_dir / '.semantic_search'
    index_dir.mkdir()
    return index_dir


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for testing without actual model."""
    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        mock_model = MagicMock()
        # Mock encode to return embeddings of correct shape
        mock_model.encode.return_value = mock.MagicMock()
        mock_model.encode.return_value.shape = (1, 384)
        mock_model.encode.return_value.__len__ = lambda self: 384
        mock_st.return_value = mock_model
        yield mock_st, mock_model


@pytest.fixture
def mock_faiss():
    """Mock FAISS for testing without actual library."""
    with patch.dict('sys.modules', {'faiss': MagicMock()}):
        import faiss
        faiss.IndexFlatIP = MagicMock()
        faiss.normalize_L2 = MagicMock()
        faiss.write_index = MagicMock()
        faiss.read_index = MagicMock()
        yield faiss


# =============================================================================
# Unit Tests - IndexMetadata
# =============================================================================

class TestIndexMetadata:
    """Tests for IndexMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating IndexMetadata with all fields."""
        metadata = ss.IndexMetadata(
            name='test-index',
            source_path='/path/to/source',
            created_at='2024-01-01T00:00:00',
            updated_at='2024-01-01T00:00:00',
            model_name='all-MiniLM-L6-v2',
            include_patterns=['*.py'],
            exclude_patterns=['*_test.py'],
            file_count=10,
            chunk_count=50,
            file_hashes={'file1.py': 'abc123'}
        )

        assert metadata.name == 'test-index'
        assert metadata.source_path == '/path/to/source'
        assert metadata.file_count == 10
        assert metadata.chunk_count == 50
        assert metadata.include_patterns == ['*.py']

    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ss.IndexMetadata(
            name='test',
            source_path='/path',
            created_at='2024-01-01',
            updated_at='2024-01-01',
            model_name='model'
        )
        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert data['name'] == 'test'
        assert 'file_hashes' in data

    def test_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            'name': 'test',
            'source_path': '/path',
            'created_at': '2024-01-01',
            'updated_at': '2024-01-01',
            'model_name': 'model',
            'include_patterns': [],
            'exclude_patterns': [],
            'file_count': 5,
            'chunk_count': 20,
            'file_hashes': {}
        }
        metadata = ss.IndexMetadata.from_dict(data)

        assert metadata.name == 'test'
        assert metadata.file_count == 5


# =============================================================================
# Unit Tests - FileChunk
# =============================================================================

class TestFileChunk:
    """Tests for FileChunk dataclass."""

    def test_create_chunk(self):
        """Test creating a FileChunk."""
        chunk = ss.FileChunk(
            file_path='/path/to/file.py',
            content='def hello(): pass',
            start_line=1,
            end_line=10,
            chunk_index=0
        )

        assert chunk.file_path == '/path/to/file.py'
        assert chunk.start_line == 1
        assert chunk.end_line == 10
        assert chunk.chunk_index == 0


# =============================================================================
# Unit Tests - Utility Functions
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_index_dir_default(self):
        """Test default index directory."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove SEMANTIC_SEARCH_DIR if present
            os.environ.pop('SEMANTIC_SEARCH_DIR', None)
            index_dir = ss.get_index_dir()
            assert index_dir == Path.home() / '.semantic_search'

    def test_get_index_dir_custom(self):
        """Test custom index directory from environment."""
        with patch.dict(os.environ, {'SEMANTIC_SEARCH_DIR': '/custom/path'}):
            index_dir = ss.get_index_dir()
            assert index_dir == Path('/custom/path')

    def test_get_index_dir_expanduser(self):
        """Test index directory with ~ expansion."""
        with patch.dict(os.environ, {'SEMANTIC_SEARCH_DIR': '~/custom'}):
            index_dir = ss.get_index_dir()
            assert str(index_dir).startswith(str(Path.home()))


class TestFormatResult:
    """Tests for format_result function."""

    def test_format_with_content(self):
        """Test formatting result with content."""
        chunk = ss.FileChunk(
            file_path='/path/to/file.py',
            content='def hello(): pass',
            start_line=1,
            end_line=5,
            chunk_index=0
        )
        result = ss.format_result(chunk, 0.95, show_content=True)

        assert '/path/to/file.py' in result
        assert 'Lines: 1-5' in result
        assert '0.95' in result
        assert 'def hello(): pass' in result

    def test_format_without_content(self):
        """Test formatting result without content."""
        chunk = ss.FileChunk(
            file_path='/path/to/file.py',
            content='def hello(): pass',
            start_line=1,
            end_line=5,
            chunk_index=0
        )
        result = ss.format_result(chunk, 0.95, show_content=False)

        assert '/path/to/file.py' in result
        assert 'def hello(): pass' not in result

    def test_format_truncates_long_content(self):
        """Test that long content is truncated."""
        long_content = 'x' * 600
        chunk = ss.FileChunk(
            file_path='/path/to/file.py',
            content=long_content,
            start_line=1,
            end_line=100,
            chunk_index=0
        )
        result = ss.format_result(chunk, 0.5, show_content=True)

        assert '...' in result
        assert len(result) < len(long_content) + 200


# =============================================================================
# Unit Tests - SemanticSearchIndex
# =============================================================================

class TestSemanticSearchIndex:
    """Tests for SemanticSearchIndex class."""

    def test_init(self, temp_dir):
        """Test index initialization."""
        index = ss.SemanticSearchIndex('test-index', index_dir=temp_dir)

        assert index.name == 'test-index'
        assert index.index_dir == temp_dir / 'test-index'
        assert index.metadata is None
        assert index.chunks == []

    def test_is_text_file_by_extension(self, temp_dir):
        """Test text file detection by extension."""
        index = ss.SemanticSearchIndex('test', index_dir=temp_dir)

        # Text files
        assert index._is_text_file(Path('file.py')) is True
        assert index._is_text_file(Path('file.md')) is True
        assert index._is_text_file(Path('file.js')) is True
        assert index._is_text_file(Path('file.json')) is True

    def test_is_text_file_binary_detection(self, temp_dir):
        """Test binary file detection."""
        index = ss.SemanticSearchIndex('test', index_dir=temp_dir)

        # Create a binary file
        binary_file = temp_dir / 'binary.dat'
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')

        assert index._is_text_file(binary_file) is False

    def test_is_text_file_text_detection(self, temp_dir):
        """Test actual text file detection."""
        index = ss.SemanticSearchIndex('test', index_dir=temp_dir)

        # Create a text file with unusual extension
        text_file = temp_dir / 'textfile.xyz'
        text_file.write_text('This is plain text content')

        assert index._is_text_file(text_file) is True

    def test_matches_patterns_include(self, temp_dir):
        """Test pattern matching with include patterns."""
        index = ss.SemanticSearchIndex('test', index_dir=temp_dir)

        assert index._matches_patterns(Path('file.py'), ['*.py'], []) is True
        assert index._matches_patterns(Path('file.md'), ['*.py'], []) is False
        assert index._matches_patterns(Path('file.py'), ['*.py', '*.md'], []) is True

    def test_matches_patterns_exclude(self, temp_dir):
        """Test pattern matching with exclude patterns."""
        index = ss.SemanticSearchIndex('test', index_dir=temp_dir)

        assert index._matches_patterns(Path('file.py'), [], ['*.py']) is False
        assert index._matches_patterns(Path('file.md'), [], ['*.py']) is True
        assert index._matches_patterns(Path('test_file.py'), ['*.py'], ['test_*']) is False

    def test_matches_patterns_no_patterns(self, temp_dir):
        """Test pattern matching with no patterns (include all)."""
        index = ss.SemanticSearchIndex('test', index_dir=temp_dir)

        assert index._matches_patterns(Path('anything.xyz'), [], []) is True

    def test_get_file_hash(self, temp_dir):
        """Test file hash calculation."""
        index = ss.SemanticSearchIndex('test', index_dir=temp_dir)

        # Create a file
        test_file = temp_dir / 'test.txt'
        test_file.write_text('test content')

        hash1 = index._get_file_hash(test_file)
        assert len(hash1) == 32  # MD5 hex length

        # Same content should produce same hash
        hash2 = index._get_file_hash(test_file)
        assert hash1 == hash2

        # Different content should produce different hash
        test_file.write_text('different content')
        hash3 = index._get_file_hash(test_file)
        assert hash1 != hash3

    def test_get_file_hash_nonexistent(self, temp_dir):
        """Test file hash for nonexistent file."""
        index = ss.SemanticSearchIndex('test', index_dir=temp_dir)
        hash_result = index._get_file_hash(temp_dir / 'nonexistent.txt')
        assert hash_result == ""

    def test_chunk_file(self, sample_files):
        """Test file chunking."""
        index = ss.SemanticSearchIndex('test')
        chunks = index._chunk_file(sample_files / 'src' / 'main.py')

        assert len(chunks) >= 1
        assert all(isinstance(c, ss.FileChunk) for c in chunks)
        assert all(c.file_path.endswith('main.py') for c in chunks)
        assert all(c.start_line >= 1 for c in chunks)

    def test_chunk_file_empty(self, temp_dir):
        """Test chunking an empty file."""
        index = ss.SemanticSearchIndex('test')
        empty_file = temp_dir / 'empty.txt'
        empty_file.write_text('')

        chunks = index._chunk_file(empty_file)
        assert chunks == []

    def test_chunk_file_nonexistent(self, temp_dir):
        """Test chunking a nonexistent file."""
        index = ss.SemanticSearchIndex('test')
        chunks = index._chunk_file(temp_dir / 'nonexistent.txt')
        assert chunks == []

    def test_collect_files(self, sample_files):
        """Test file collection."""
        index = ss.SemanticSearchIndex('test')

        # Collect all files
        files = index._collect_files(sample_files / 'src', [], [])
        assert len(files) == 2
        assert any('main.py' in str(f) for f in files)
        assert any('auth.py' in str(f) for f in files)

    def test_collect_files_with_include(self, sample_files):
        """Test file collection with include pattern."""
        index = ss.SemanticSearchIndex('test')

        files = index._collect_files(sample_files / 'docs', ['*.md'], [])
        assert len(files) == 2
        assert all(str(f).endswith('.md') for f in files)

    def test_collect_files_with_exclude(self, sample_files):
        """Test file collection with exclude pattern."""
        index = ss.SemanticSearchIndex('test')

        files = index._collect_files(sample_files / 'src', [], ['auth*'])
        assert len(files) == 1
        assert 'main.py' in str(files[0])

    def test_collect_files_single_file(self, sample_files):
        """Test collecting a single file."""
        index = ss.SemanticSearchIndex('test')

        files = index._collect_files(sample_files / 'src' / 'main.py', [], [])
        assert len(files) == 1

    def test_delete_nonexistent(self, temp_dir):
        """Test deleting a nonexistent index."""
        index = ss.SemanticSearchIndex('nonexistent', index_dir=temp_dir)
        assert index.delete() is False

    def test_delete_existing(self, temp_dir):
        """Test deleting an existing index directory."""
        index = ss.SemanticSearchIndex('test', index_dir=temp_dir)
        index.index_dir.mkdir(parents=True)
        (index.index_dir / 'test.txt').write_text('test')

        assert index.index_dir.exists()
        assert index.delete() is True
        assert not index.index_dir.exists()

    def test_load_nonexistent(self, temp_dir):
        """Test loading a nonexistent index."""
        index = ss.SemanticSearchIndex('nonexistent', index_dir=temp_dir)
        assert index.load() is False


# =============================================================================
# Unit Tests - list_indexes
# =============================================================================

class TestListIndexes:
    """Tests for list_indexes function."""

    def test_list_empty(self, temp_dir):
        """Test listing when no indexes exist."""
        with patch.object(ss, 'get_index_dir', return_value=temp_dir):
            indexes = ss.list_indexes()
            assert indexes == []

    def test_list_nonexistent_dir(self, temp_dir):
        """Test listing when index directory doesn't exist."""
        nonexistent = temp_dir / 'nonexistent'
        with patch.object(ss, 'get_index_dir', return_value=nonexistent):
            indexes = ss.list_indexes()
            assert indexes == []

    def test_list_with_indexes(self, temp_dir):
        """Test listing existing indexes."""
        # Create mock index directories with metadata
        for name in ['index1', 'index2']:
            idx_dir = temp_dir / name
            idx_dir.mkdir()
            metadata = ss.IndexMetadata(
                name=name,
                source_path=f'/path/to/{name}',
                created_at='2024-01-01',
                updated_at='2024-01-01',
                model_name='model',
                file_count=10,
                chunk_count=50
            )
            (idx_dir / 'metadata.json').write_text(json.dumps(metadata.to_dict()))

        with patch.object(ss, 'get_index_dir', return_value=temp_dir):
            indexes = ss.list_indexes()
            assert len(indexes) == 2
            assert indexes[0].name == 'index1'
            assert indexes[1].name == 'index2'


# =============================================================================
# Unit Tests - CLI Parser
# =============================================================================

class TestCLIParser:
    """Tests for CLI argument parser."""

    def test_parser_creation(self):
        """Test parser is created successfully."""
        parser = ss.create_parser()
        assert parser is not None
        assert parser.prog == 'semantic_search'

    def test_parse_index_command(self):
        """Test parsing index command."""
        parser = ss.create_parser()
        args = parser.parse_args(['index', './src', '--name', 'my-index'])

        assert args.command == 'index'
        assert args.path == Path('./src')
        assert args.name == 'my-index'

    def test_parse_index_with_patterns(self):
        """Test parsing index command with patterns."""
        parser = ss.create_parser()
        args = parser.parse_args([
            'index', './src', '--name', 'my-index',
            '--include', '*.py', '*.md',
            '--exclude', '*_test.py'
        ])

        assert args.include == ['*.py', '*.md']
        assert args.exclude == ['*_test.py']

    def test_parse_list_indexes(self):
        """Test parsing list-indexes command."""
        parser = ss.create_parser()
        args = parser.parse_args(['list-indexes'])

        assert args.command == 'list-indexes'

    def test_parse_list_indexes_json(self):
        """Test parsing list-indexes with --json."""
        parser = ss.create_parser()
        args = parser.parse_args(['list-indexes', '--json'])

        assert args.command == 'list-indexes'
        assert args.json is True

    def test_parse_delete_index(self):
        """Test parsing delete-index command."""
        parser = ss.create_parser()
        args = parser.parse_args(['delete-index', 'my-index'])

        assert args.command == 'delete-index'
        assert args.name == 'my-index'

    def test_parse_delete_index_force(self):
        """Test parsing delete-index with --force."""
        parser = ss.create_parser()
        args = parser.parse_args(['delete-index', 'my-index', '--force'])

        assert args.force is True

    def test_parse_update_index(self):
        """Test parsing update-index command."""
        parser = ss.create_parser()
        args = parser.parse_args(['update-index', 'my-index'])

        assert args.command == 'update-index'
        assert args.name == 'my-index'

    def test_parse_search_with_index(self):
        """Test parsing search with --index."""
        parser = ss.create_parser()
        args = parser.parse_args(['authentication', '--index', 'my-code'])

        assert args.query == 'authentication'
        assert args.index == 'my-code'

    def test_parse_search_with_path(self):
        """Test parsing search with --path."""
        parser = ss.create_parser()
        args = parser.parse_args(['query text', '--path', './src'])

        assert args.query == 'query text'
        assert args.path == Path('./src')

    def test_parse_search_with_top_k(self):
        """Test parsing search with --top-k."""
        parser = ss.create_parser()
        args = parser.parse_args(['query', '--index', 'idx', '--top-k', '5'])

        assert args.top_k == 5

    def test_parse_search_json_output(self):
        """Test parsing search with --json."""
        parser = ss.create_parser()
        args = parser.parse_args(['query', '--index', 'idx', '--json'])

        assert args.output_json is True

    def test_parse_no_content(self):
        """Test parsing search with --no-content."""
        parser = ss.create_parser()
        args = parser.parse_args(['query', '--index', 'idx', '--no-content'])

        assert args.no_content is True


# =============================================================================
# Integration Tests with Mocking
# =============================================================================

class TestIntegrationWithMocks:
    """Integration tests using mocked dependencies."""

    @patch('semantic_search.SemanticSearchIndex._load_model')
    def test_create_index_integration(self, mock_load_model, sample_files, temp_dir):
        """Test creating an index with mocked model and FAISS."""
        import numpy as np

        # Mock the dependencies
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(10, 384).astype(np.float32)

        with patch.dict('sys.modules', {'faiss': MagicMock()}):
            import faiss
            mock_faiss_index = MagicMock()
            faiss.IndexFlatIP.return_value = mock_faiss_index
            faiss.normalize_L2 = MagicMock()
            faiss.write_index = MagicMock()

            index = ss.SemanticSearchIndex('test-index', index_dir=temp_dir)
            index.model = mock_model

            chunk_count = index.create(
                sample_files / 'src',
                include_patterns=['*.py'],
                model_name='all-MiniLM-L6-v2',
                verbose=False
            )

            assert chunk_count > 0
            assert index.metadata is not None
            assert index.metadata.name == 'test-index'
            assert index.metadata.file_count == 2

    @patch('semantic_search.SemanticSearchIndex._load_model')
    def test_search_integration(self, mock_load_model, temp_dir):
        """Test searching an index with mocked dependencies."""
        import numpy as np

        # Setup mock model
        mock_model = MagicMock()
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        mock_model.encode.return_value = query_embedding

        with patch.dict('sys.modules', {'faiss': MagicMock()}):
            import faiss
            mock_faiss_index = MagicMock()
            mock_faiss_index.search.return_value = (
                np.array([[0.9, 0.8, 0.7]]),
                np.array([[0, 1, 2]])
            )
            faiss.normalize_L2 = MagicMock()

            index = ss.SemanticSearchIndex('test-index', index_dir=temp_dir)
            index.model = mock_model
            index.faiss_index = mock_faiss_index
            index.chunks = [
                ss.FileChunk('/path/file1.py', 'content1', 1, 10, 0),
                ss.FileChunk('/path/file2.py', 'content2', 1, 10, 1),
                ss.FileChunk('/path/file3.py', 'content3', 1, 10, 2),
            ]

            results = index.search('test query', top_k=3)

            assert len(results) == 3
            assert results[0][1] == 0.9  # Score
            assert results[0][0].file_path == '/path/file1.py'


# =============================================================================
# CLI Exit Code Tests
# =============================================================================

class TestExitCodes:
    """Tests for correct exit codes."""

    def test_exit_codes_defined(self):
        """Test that exit codes are properly defined."""
        assert ss.EXIT_SUCCESS == 0
        assert ss.EXIT_INDEX_NOT_FOUND == 1
        assert ss.EXIT_NO_RESULTS == 2
        assert ss.EXIT_INVALID_ARGS == 3


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_query(self):
        """Test handling of empty query."""
        parser = ss.create_parser()
        # Empty query should show help
        args = parser.parse_args([])
        assert args.query is None

    def test_default_values(self):
        """Test default values in parser."""
        parser = ss.create_parser()
        args = parser.parse_args(['query', '--index', 'idx'])

        assert args.top_k == ss.DEFAULT_TOP_K
        assert args.model == ss.DEFAULT_MODEL
        assert args.verbose is False
        assert args.no_content is False

    def test_text_extensions_defined(self):
        """Test that text extensions are defined."""
        assert '.py' in ss.DEFAULT_TEXT_EXTENSIONS
        assert '.md' in ss.DEFAULT_TEXT_EXTENSIONS
        assert '.js' in ss.DEFAULT_TEXT_EXTENSIONS

    def test_special_filenames(self, temp_dir):
        """Test handling of special filenames like Makefile."""
        index = ss.SemanticSearchIndex('test', index_dir=temp_dir)

        makefile = temp_dir / 'Makefile'
        makefile.write_text('all:\n\techo "hello"')

        assert index._is_text_file(makefile) is True


# =============================================================================
# Constants and Configuration Tests
# =============================================================================

class TestConfiguration:
    """Tests for configuration constants."""

    def test_default_model(self):
        """Test default model is set correctly."""
        assert ss.DEFAULT_MODEL == 'all-MiniLM-L6-v2'

    def test_default_chunk_size(self):
        """Test default chunk size."""
        assert ss.DEFAULT_CHUNK_SIZE == 512

    def test_default_top_k(self):
        """Test default top-k value."""
        assert ss.DEFAULT_TOP_K == 10


# =============================================================================
# Main Function Tests
# =============================================================================

class TestMainFunction:
    """Tests for main() function behavior."""

    def test_main_no_args_shows_help(self, capsys):
        """Test that running without args shows help."""
        with patch('sys.argv', ['semantic_search']):
            with pytest.raises(SystemExit) as exc_info:
                ss.main()
            # Should exit with success when showing help
            assert exc_info.value.code == ss.EXIT_SUCCESS

    def test_main_list_indexes_empty(self, temp_dir, capsys):
        """Test list-indexes with no indexes."""
        with patch('sys.argv', ['semantic_search', 'list-indexes']):
            with patch.object(ss, 'get_index_dir', return_value=temp_dir):
                with pytest.raises(SystemExit) as exc_info:
                    ss.main()
                assert exc_info.value.code == ss.EXIT_SUCCESS

        captured = capsys.readouterr()
        assert 'No indexes found' in captured.out

    def test_main_delete_nonexistent(self, temp_dir, capsys):
        """Test deleting a nonexistent index."""
        with patch('sys.argv', ['semantic_search', 'delete-index', 'nonexistent', '-f']):
            with patch.object(ss, 'get_index_dir', return_value=temp_dir):
                with pytest.raises(SystemExit) as exc_info:
                    ss.main()
                assert exc_info.value.code == ss.EXIT_INDEX_NOT_FOUND

    def test_main_search_no_index_or_path(self, capsys):
        """Test search without --index or --path."""
        with patch('sys.argv', ['semantic_search', 'query']):
            with pytest.raises(SystemExit) as exc_info:
                ss.main()
            assert exc_info.value.code == ss.EXIT_INVALID_ARGS

        captured = capsys.readouterr()
        assert 'Must specify either --index or --path' in captured.err

    def test_main_index_path_not_found(self, temp_dir, capsys):
        """Test indexing a nonexistent path."""
        with patch('sys.argv', ['semantic_search', 'index', '/nonexistent/path', '-n', 'test']):
            with pytest.raises(SystemExit) as exc_info:
                ss.main()
            assert exc_info.value.code == ss.EXIT_INVALID_ARGS

        captured = capsys.readouterr()
        assert 'Path not found' in captured.err


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
