#!/usr/bin/env python3
"""
Comprehensive tests for vector_search CLI tool.

Tests cover all commands, backends, error handling, and edge cases
using mocking to avoid actual database connections.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestHelperFunctions(unittest.TestCase):
    """Tests for helper functions."""

    def test_get_storage_dir_default(self):
        """Test default storage directory."""
        from vector_search import get_storage_dir
        with patch.dict(os.environ, {}, clear=True):
            path = get_storage_dir()
            self.assertTrue(str(path).endswith(".vector_search"))

    def test_get_storage_dir_custom(self):
        """Test custom storage directory."""
        from vector_search import get_storage_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom_storage")
            with patch.dict(os.environ, {"VECTOR_SEARCH_DIR": custom_path}):
                path = get_storage_dir()
                self.assertEqual(str(path), custom_path)
                self.assertTrue(path.exists())

    def test_generate_doc_id(self):
        """Test document ID generation."""
        from vector_search import generate_doc_id
        doc_id1 = generate_doc_id("content1", "source1")
        doc_id2 = generate_doc_id("content1", "source1")
        doc_id3 = generate_doc_id("content2", "source1")

        # Same content and source should produce same ID
        self.assertEqual(doc_id1, doc_id2)
        # Different content should produce different ID
        self.assertNotEqual(doc_id1, doc_id3)
        # ID should be MD5 hash format
        self.assertEqual(len(doc_id1), 32)

    def test_chunk_text_small(self):
        """Test chunking small text."""
        from vector_search import chunk_text
        text = "This is a small text."
        chunks = chunk_text(text, chunk_size=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_chunk_text_large(self):
        """Test chunking large text."""
        from vector_search import chunk_text
        text = "This is a sentence. " * 100
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        # All chunks should be non-empty
        for chunk in chunks:
            self.assertTrue(len(chunk) > 0)

    def test_chunk_text_break_at_sentence(self):
        """Test that chunking prefers sentence boundaries."""
        from vector_search import chunk_text
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        # Should try to break at periods
        for chunk in chunks:
            self.assertTrue(chunk.strip())

    def test_read_file_success(self):
        """Test reading a file successfully."""
        from vector_search import read_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            f.flush()
            content, source = read_file(f.name)
            self.assertEqual(content, "Test content")
            self.assertEqual(source, f.name)
            os.unlink(f.name)

    def test_read_file_not_found(self):
        """Test reading a non-existent file."""
        from vector_search import read_file
        content, source = read_file("/nonexistent/file.txt")
        self.assertEqual(content, "")


class TestChunkText(unittest.TestCase):
    """Additional tests for chunk_text function."""

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        from vector_search import chunk_text
        chunks = chunk_text("")
        self.assertEqual(chunks, [""])

    def test_chunk_text_single_char(self):
        """Test chunking single character."""
        from vector_search import chunk_text
        chunks = chunk_text("a", chunk_size=10)
        self.assertEqual(chunks, ["a"])


class TestGenerateDocId(unittest.TestCase):
    """Additional tests for generate_doc_id function."""

    def test_generate_doc_id_unicode(self):
        """Test document ID generation with unicode."""
        from vector_search import generate_doc_id
        doc_id = generate_doc_id("Unicode content: \u4e2d\u6587", "source.txt")
        self.assertEqual(len(doc_id), 32)

    def test_generate_doc_id_long_content(self):
        """Test document ID generation with long content."""
        from vector_search import generate_doc_id
        long_content = "x" * 10000
        doc_id = generate_doc_id(long_content, "source.txt")
        self.assertEqual(len(doc_id), 32)


class TestParser(unittest.TestCase):
    """Tests for argument parser."""

    def test_parser_index_command(self):
        """Test parsing index command."""
        from vector_search import create_parser
        parser = create_parser()
        args = parser.parse_args(["index", "file.txt", "--collection", "test"])
        self.assertEqual(args.command, "index")
        self.assertEqual(args.files, ["file.txt"])
        self.assertEqual(args.collection, "test")

    def test_parser_query_command(self):
        """Test parsing query command."""
        from vector_search import create_parser
        parser = create_parser()
        args = parser.parse_args(["query", "search text", "--collection", "test", "--top-k", "10"])
        self.assertEqual(args.command, "query")
        self.assertEqual(args.query_text, "search text")
        self.assertEqual(args.collection, "test")
        self.assertEqual(args.top_k, 10)

    def test_parser_list_collections_command(self):
        """Test parsing list-collections command."""
        from vector_search import create_parser
        parser = create_parser()
        args = parser.parse_args(["list-collections"])
        self.assertEqual(args.command, "list-collections")

    def test_parser_stats_command(self):
        """Test parsing stats command."""
        from vector_search import create_parser
        parser = create_parser()
        args = parser.parse_args(["stats", "--collection", "test"])
        self.assertEqual(args.command, "stats")
        self.assertEqual(args.collection, "test")

    def test_parser_delete_command(self):
        """Test parsing delete command."""
        from vector_search import create_parser
        parser = create_parser()
        args = parser.parse_args(["delete", "--collection", "test", "--force"])
        self.assertEqual(args.command, "delete")
        self.assertEqual(args.collection, "test")
        self.assertTrue(args.force)

    def test_parser_global_backend_option(self):
        """Test backend option with subcommand."""
        from vector_search import create_parser
        parser = create_parser()
        # Backend option comes before subcommand
        args = parser.parse_args(["--backend", "qdrant", "list-collections"])
        self.assertEqual(args.backend, "qdrant")

    def test_parser_global_json_option(self):
        """Test JSON output option with subcommand."""
        from vector_search import create_parser
        parser = create_parser()
        # JSON option comes before subcommand
        args = parser.parse_args(["--json", "list-collections"])
        self.assertTrue(args.json)

    def test_parser_global_model_option(self):
        """Test model option."""
        from vector_search import create_parser
        parser = create_parser()
        args = parser.parse_args(["--model", "custom-model", "list-collections"])
        self.assertEqual(args.model, "custom-model")

    def test_parser_chunk_options(self):
        """Test chunk size and overlap options."""
        from vector_search import create_parser
        parser = create_parser()
        args = parser.parse_args(["index", "file.txt", "-c", "test", "--chunk-size", "500", "--overlap", "100"])
        self.assertEqual(args.chunk_size, 500)
        self.assertEqual(args.overlap, 100)


class TestExitCodes(unittest.TestCase):
    """Tests for exit codes."""

    def test_exit_codes_defined(self):
        """Test that exit codes are properly defined."""
        from vector_search import (
            EXIT_SUCCESS,
            EXIT_COLLECTION_NOT_FOUND,
            EXIT_BACKEND_ERROR,
            EXIT_INVALID_ARGS,
        )
        self.assertEqual(EXIT_SUCCESS, 0)
        self.assertEqual(EXIT_COLLECTION_NOT_FOUND, 1)
        self.assertEqual(EXIT_BACKEND_ERROR, 2)
        self.assertEqual(EXIT_INVALID_ARGS, 3)


class TestCommandsWithMocking(unittest.TestCase):
    """Tests for CLI commands with mocked backends."""

    def test_cmd_list_collections_success(self):
        """Test list-collections command."""
        from vector_search import cmd_list_collections, create_parser

        mock_backend = MagicMock()
        mock_backend.list_collections.return_value = ["collection1", "collection2"]

        parser = create_parser()
        args = parser.parse_args(["list-collections"])

        with patch("vector_search.get_backend", return_value=mock_backend):
            output = StringIO()
            with patch("sys.stdout", output):
                result = cmd_list_collections(args)

        self.assertEqual(result, 0)
        self.assertIn("collection1", output.getvalue())
        self.assertIn("collection2", output.getvalue())

    def test_cmd_list_collections_empty(self):
        """Test list-collections with no collections."""
        from vector_search import cmd_list_collections, create_parser

        mock_backend = MagicMock()
        mock_backend.list_collections.return_value = []

        parser = create_parser()
        args = parser.parse_args(["list-collections"])

        with patch("vector_search.get_backend", return_value=mock_backend):
            output = StringIO()
            with patch("sys.stdout", output):
                result = cmd_list_collections(args)

        self.assertEqual(result, 0)
        self.assertIn("No collections found", output.getvalue())

    def test_cmd_list_collections_json(self):
        """Test list-collections with JSON output."""
        from vector_search import cmd_list_collections, create_parser

        mock_backend = MagicMock()
        mock_backend.list_collections.return_value = ["collection1", "collection2"]

        parser = create_parser()
        args = parser.parse_args(["--json", "list-collections"])

        with patch("vector_search.get_backend", return_value=mock_backend):
            output = StringIO()
            with patch("sys.stdout", output):
                result = cmd_list_collections(args)

        self.assertEqual(result, 0)
        output_data = json.loads(output.getvalue())
        self.assertEqual(output_data, ["collection1", "collection2"])

    def test_cmd_stats_success(self):
        """Test stats command."""
        from vector_search import cmd_stats, create_parser

        mock_backend = MagicMock()
        mock_backend.get_stats.return_value = {
            "name": "test",
            "document_count": 100,
            "dimension": 384,
            "backend": "chromadb",
        }

        parser = create_parser()
        args = parser.parse_args(["stats", "--collection", "test"])

        with patch("vector_search.get_backend", return_value=mock_backend):
            output = StringIO()
            with patch("sys.stdout", output):
                result = cmd_stats(args)

        self.assertEqual(result, 0)
        self.assertIn("100", output.getvalue())

    def test_cmd_stats_no_collection(self):
        """Test stats command without collection argument."""
        from vector_search import cmd_stats, create_parser, EXIT_INVALID_ARGS

        parser = create_parser()
        # Create args manually to simulate missing collection
        args = MagicMock()
        args.collection = None
        args.backend = "chromadb"
        args.json = False

        result = cmd_stats(args)
        self.assertEqual(result, EXIT_INVALID_ARGS)

    def test_cmd_delete_with_force(self):
        """Test delete command with force flag."""
        from vector_search import cmd_delete, create_parser

        mock_backend = MagicMock()

        parser = create_parser()
        args = parser.parse_args(["delete", "--collection", "test", "--force"])

        with patch("vector_search.get_backend", return_value=mock_backend):
            output = StringIO()
            with patch("sys.stdout", output):
                result = cmd_delete(args)

        self.assertEqual(result, 0)
        mock_backend.delete_collection.assert_called_once_with("test")

    def test_cmd_delete_with_confirmation(self):
        """Test delete command with user confirmation."""
        from vector_search import cmd_delete, create_parser

        mock_backend = MagicMock()

        parser = create_parser()
        args = parser.parse_args(["delete", "--collection", "test"])

        with patch("vector_search.get_backend", return_value=mock_backend):
            with patch("builtins.input", return_value="y"):
                output = StringIO()
                with patch("sys.stdout", output):
                    result = cmd_delete(args)

        self.assertEqual(result, 0)
        mock_backend.delete_collection.assert_called_once()

    def test_cmd_delete_cancelled(self):
        """Test delete command cancelled by user."""
        from vector_search import cmd_delete, create_parser

        mock_backend = MagicMock()

        parser = create_parser()
        args = parser.parse_args(["delete", "--collection", "test"])

        with patch("vector_search.get_backend", return_value=mock_backend):
            with patch("builtins.input", return_value="n"):
                output = StringIO()
                with patch("sys.stdout", output):
                    result = cmd_delete(args)

        self.assertEqual(result, 0)
        mock_backend.delete_collection.assert_not_called()
        self.assertIn("Cancelled", output.getvalue())

    def test_cmd_delete_no_collection(self):
        """Test delete command without collection."""
        from vector_search import cmd_delete, EXIT_INVALID_ARGS

        args = MagicMock()
        args.collection = None

        result = cmd_delete(args)
        self.assertEqual(result, EXIT_INVALID_ARGS)

    def test_cmd_query_collection_not_found(self):
        """Test query command with non-existent collection."""
        from vector_search import cmd_query, create_parser, EXIT_COLLECTION_NOT_FOUND

        mock_backend = MagicMock()
        mock_backend.collection_exists.return_value = False

        mock_model = MagicMock()

        parser = create_parser()
        args = parser.parse_args(["query", "test query", "--collection", "nonexistent"])

        with patch("vector_search.get_backend", return_value=mock_backend):
            with patch("vector_search.EmbeddingModel", return_value=mock_model):
                result = cmd_query(args)

        self.assertEqual(result, EXIT_COLLECTION_NOT_FOUND)

    def test_cmd_query_no_collection(self):
        """Test query command without collection."""
        from vector_search import cmd_query, EXIT_INVALID_ARGS

        args = MagicMock()
        args.collection = None
        args.query_text = "test"

        result = cmd_query(args)
        self.assertEqual(result, EXIT_INVALID_ARGS)

    def test_cmd_query_no_query_text(self):
        """Test query command without query text."""
        from vector_search import cmd_query, EXIT_INVALID_ARGS

        args = MagicMock()
        args.collection = "test"
        args.query_text = None

        result = cmd_query(args)
        self.assertEqual(result, EXIT_INVALID_ARGS)

    def test_cmd_query_success(self):
        """Test successful query command."""
        from vector_search import cmd_query, create_parser

        mock_backend = MagicMock()
        mock_backend.collection_exists.return_value = True
        mock_backend.query.return_value = [
            {"content": "Test result", "source": "file.txt", "score": 0.95}
        ]

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]

        parser = create_parser()
        args = parser.parse_args(["query", "test query", "--collection", "test"])

        with patch("vector_search.get_backend", return_value=mock_backend):
            with patch("vector_search.EmbeddingModel", return_value=mock_model):
                output = StringIO()
                with patch("sys.stdout", output):
                    result = cmd_query(args)

        self.assertEqual(result, 0)
        self.assertIn("Test result", output.getvalue())

    def test_cmd_query_json_output(self):
        """Test query command with JSON output."""
        from vector_search import cmd_query, create_parser

        mock_backend = MagicMock()
        mock_backend.collection_exists.return_value = True
        mock_backend.query.return_value = [
            {"content": "Test result", "source": "file.txt", "score": 0.95}
        ]

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]

        parser = create_parser()
        args = parser.parse_args(["--json", "query", "test query", "--collection", "test"])

        with patch("vector_search.get_backend", return_value=mock_backend):
            with patch("vector_search.EmbeddingModel", return_value=mock_model):
                output = StringIO()
                with patch("sys.stdout", output):
                    result = cmd_query(args)

        self.assertEqual(result, 0)
        output_data = json.loads(output.getvalue())
        self.assertEqual(len(output_data), 1)
        self.assertEqual(output_data[0]["content"], "Test result")

    def test_cmd_query_no_results(self):
        """Test query command with no results."""
        from vector_search import cmd_query, create_parser

        mock_backend = MagicMock()
        mock_backend.collection_exists.return_value = True
        mock_backend.query.return_value = []

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]

        parser = create_parser()
        args = parser.parse_args(["query", "test query", "--collection", "test"])

        with patch("vector_search.get_backend", return_value=mock_backend):
            with patch("vector_search.EmbeddingModel", return_value=mock_model):
                output = StringIO()
                with patch("sys.stdout", output):
                    result = cmd_query(args)

        self.assertEqual(result, 0)
        self.assertIn("No results found", output.getvalue())

    def test_cmd_index_no_collection(self):
        """Test index command without collection."""
        from vector_search import cmd_index, EXIT_INVALID_ARGS

        args = MagicMock()
        args.collection = None

        result = cmd_index(args)
        self.assertEqual(result, EXIT_INVALID_ARGS)

    def test_cmd_index_no_files(self):
        """Test index command with no matching files."""
        from vector_search import cmd_index, create_parser, EXIT_INVALID_ARGS

        parser = create_parser()
        args = parser.parse_args(["index", "/nonexistent/*.xyz", "--collection", "test"])

        with patch("sys.stderr", StringIO()):
            result = cmd_index(args)

        self.assertEqual(result, EXIT_INVALID_ARGS)

    def test_cmd_index_success(self):
        """Test successful index command."""
        from vector_search import cmd_index, create_parser

        mock_backend = MagicMock()
        mock_model = MagicMock()
        mock_model.dimension = 384
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test document content")
            f.flush()
            filepath = f.name

        try:
            parser = create_parser()
            args = parser.parse_args(["index", filepath, "--collection", "test"])

            with patch("vector_search.get_backend", return_value=mock_backend):
                with patch("vector_search.EmbeddingModel", return_value=mock_model):
                    output = StringIO()
                    with patch("sys.stdout", output):
                        result = cmd_index(args)

            self.assertEqual(result, 0)
            mock_backend.create_collection.assert_called_once()
            mock_backend.add_documents.assert_called_once()
        finally:
            os.unlink(filepath)

    def test_cmd_index_glob_pattern(self):
        """Test index command with glob pattern."""
        from vector_search import cmd_index, create_parser

        mock_backend = MagicMock()
        mock_model = MagicMock()
        mock_model.dimension = 384
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                filepath = os.path.join(tmpdir, f"test{i}.txt")
                with open(filepath, "w") as f:
                    f.write(f"Content {i}")

            parser = create_parser()
            pattern = os.path.join(tmpdir, "*.txt")
            args = parser.parse_args(["index", pattern, "--collection", "test"])

            with patch("vector_search.get_backend", return_value=mock_backend):
                with patch("vector_search.EmbeddingModel", return_value=mock_model):
                    output = StringIO()
                    with patch("sys.stdout", output):
                        result = cmd_index(args)

            self.assertEqual(result, 0)
            mock_backend.add_documents.assert_called_once()


class TestGetBackend(unittest.TestCase):
    """Tests for backend factory function."""

    def test_get_backend_invalid(self):
        """Test error on invalid backend."""
        from vector_search import get_backend, EXIT_INVALID_ARGS

        with self.assertRaises(SystemExit) as cm:
            with patch("sys.stderr", StringIO()):
                get_backend("invalid_backend")
        self.assertEqual(cm.exception.code, EXIT_INVALID_ARGS)


class TestMainFunction(unittest.TestCase):
    """Tests for main function."""

    def test_main_no_command(self):
        """Test main with no command shows help."""
        from vector_search import main

        with patch("sys.argv", ["vector_search"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                result = main()
                self.assertEqual(result, 0)

    def test_main_list_collections(self):
        """Test main with list-collections command."""
        from vector_search import main

        mock_backend = MagicMock()
        mock_backend.list_collections.return_value = ["test"]

        with patch("sys.argv", ["vector_search", "list-collections"]):
            with patch("vector_search.get_backend", return_value=mock_backend):
                with patch("sys.stdout", new_callable=StringIO):
                    result = main()
                    self.assertEqual(result, 0)


class TestBackendMocking(unittest.TestCase):
    """Tests for backend classes with proper mocking."""

    def test_pinecone_missing_api_key(self):
        """Test error when PINECONE_API_KEY is not set."""
        from vector_search import PineconeBackend, EXIT_BACKEND_ERROR

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as cm:
                with patch("sys.stderr", StringIO()):
                    PineconeBackend()
            self.assertEqual(cm.exception.code, EXIT_BACKEND_ERROR)

    def test_chromadb_backend_with_mock(self):
        """Test ChromaDB backend initialization with mock."""
        # Create a mock module
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            from vector_search import ChromaDBBackend
            # Need to reload to pick up the mock
            backend = ChromaDBBackend.__new__(ChromaDBBackend)
            backend.client = mock_client

            # Test list_collections
            mock_coll1 = MagicMock()
            mock_coll1.name = "coll1"
            mock_coll2 = MagicMock()
            mock_coll2.name = "coll2"
            mock_client.list_collections.return_value = [mock_coll1, mock_coll2]

            collections = backend.list_collections()
            self.assertEqual(collections, ["coll1", "coll2"])


class TestVectorBackendInterface(unittest.TestCase):
    """Tests for VectorBackend abstract interface."""

    def test_vector_backend_is_abstract(self):
        """Test that VectorBackend cannot be instantiated directly."""
        from vector_search import VectorBackend

        with self.assertRaises(TypeError):
            VectorBackend()


class TestStoragePath(unittest.TestCase):
    """Tests for storage path handling."""

    def test_storage_dir_creation(self):
        """Test that storage directory is created if it doesn't exist."""
        from vector_search import get_storage_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = os.path.join(tmpdir, "new", "nested", "path")
            with patch.dict(os.environ, {"VECTOR_SEARCH_DIR": new_path}):
                path = get_storage_dir()
                self.assertTrue(path.exists())
                self.assertEqual(str(path), new_path)


class TestDocumentProcessing(unittest.TestCase):
    """Tests for document processing functions."""

    def test_chunk_with_newlines(self):
        """Test chunking with newline characters."""
        from vector_search import chunk_text

        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        self.assertGreater(len(chunks), 1)

    def test_doc_id_deterministic(self):
        """Test that document IDs are deterministic."""
        from vector_search import generate_doc_id

        id1 = generate_doc_id("same content", "same/source.txt")
        id2 = generate_doc_id("same content", "same/source.txt")
        self.assertEqual(id1, id2)

    def test_doc_id_different_sources(self):
        """Test that different sources produce different IDs."""
        from vector_search import generate_doc_id

        id1 = generate_doc_id("content", "source1.txt")
        id2 = generate_doc_id("content", "source2.txt")
        self.assertNotEqual(id1, id2)


class TestReadFile(unittest.TestCase):
    """Tests for read_file function."""

    def test_read_file_with_encoding(self):
        """Test reading file with specific encoding."""
        from vector_search import read_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Test with unicode: \u00e9\u00e0\u00fc")
            f.flush()
            content, source = read_file(f.name)
            self.assertIn("unicode", content)
            os.unlink(f.name)

    def test_read_empty_file(self):
        """Test reading an empty file."""
        from vector_search import read_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.flush()  # Write nothing
            content, source = read_file(f.name)
            self.assertEqual(content, "")
            os.unlink(f.name)


class TestMultipleFiles(unittest.TestCase):
    """Tests for handling multiple files."""

    def test_index_multiple_files(self):
        """Test indexing multiple files at once."""
        from vector_search import cmd_index, create_parser

        mock_backend = MagicMock()
        mock_model = MagicMock()
        mock_model.dimension = 384
        mock_model.encode.return_value = [[0.1] * 384, [0.2] * 384]

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.txt")
            file2 = os.path.join(tmpdir, "file2.txt")

            with open(file1, "w") as f:
                f.write("Content of file 1")
            with open(file2, "w") as f:
                f.write("Content of file 2")

            parser = create_parser()
            args = parser.parse_args(["index", file1, file2, "--collection", "test"])

            with patch("vector_search.get_backend", return_value=mock_backend):
                with patch("vector_search.EmbeddingModel", return_value=mock_model):
                    output = StringIO()
                    with patch("sys.stdout", output):
                        result = cmd_index(args)

            self.assertEqual(result, 0)
            mock_backend.add_documents.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
