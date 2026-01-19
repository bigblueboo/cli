#!/usr/bin/env python3
"""
Comprehensive tests for db_query CLI tool.

These tests use mocking to avoid requiring actual database connections.
Run with: pytest test_db_query.py -v
"""

import io
import json
import os
import sys
import tempfile
from unittest import mock

import pytest

# Import the module under test
import db_query


class TestIsReadOnlyQuery:
    """Tests for the is_read_only_query function."""

    def test_select_query_is_readonly(self):
        """Basic SELECT queries should be allowed."""
        assert db_query.is_read_only_query("SELECT * FROM users")[0] is True
        assert db_query.is_read_only_query("SELECT id, name FROM users WHERE id = 1")[0] is True
        assert db_query.is_read_only_query("select * from users")[0] is True  # lowercase

    def test_select_with_join_is_readonly(self):
        """SELECT with JOINs should be allowed."""
        query = """
        SELECT u.name, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.total > 100
        """
        assert db_query.is_read_only_query(query)[0] is True

    def test_select_with_subquery_is_readonly(self):
        """SELECT with subqueries should be allowed."""
        query = "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)"
        assert db_query.is_read_only_query(query)[0] is True

    def test_show_and_describe_are_readonly(self):
        """SHOW and DESCRIBE commands should be allowed."""
        assert db_query.is_read_only_query("SHOW TABLES")[0] is True
        assert db_query.is_read_only_query("DESCRIBE users")[0] is True
        assert db_query.is_read_only_query("EXPLAIN SELECT * FROM users")[0] is True

    def test_insert_is_rejected(self):
        """INSERT queries should be rejected."""
        is_readonly, keyword = db_query.is_read_only_query(
            "INSERT INTO users (name) VALUES ('test')"
        )
        assert is_readonly is False
        assert keyword == "INSERT"

    def test_update_is_rejected(self):
        """UPDATE queries should be rejected."""
        is_readonly, keyword = db_query.is_read_only_query(
            "UPDATE users SET name = 'test' WHERE id = 1"
        )
        assert is_readonly is False
        assert keyword == "UPDATE"

    def test_delete_is_rejected(self):
        """DELETE queries should be rejected."""
        is_readonly, keyword = db_query.is_read_only_query(
            "DELETE FROM users WHERE id = 1"
        )
        assert is_readonly is False
        assert keyword == "DELETE"

    def test_drop_is_rejected(self):
        """DROP queries should be rejected."""
        is_readonly, keyword = db_query.is_read_only_query("DROP TABLE users")
        assert is_readonly is False
        assert keyword == "DROP"

    def test_create_is_rejected(self):
        """CREATE queries should be rejected."""
        is_readonly, keyword = db_query.is_read_only_query(
            "CREATE TABLE test (id INT)"
        )
        assert is_readonly is False
        assert keyword == "CREATE"

    def test_alter_is_rejected(self):
        """ALTER queries should be rejected."""
        is_readonly, keyword = db_query.is_read_only_query(
            "ALTER TABLE users ADD COLUMN email VARCHAR(255)"
        )
        assert is_readonly is False
        assert keyword == "ALTER"

    def test_truncate_is_rejected(self):
        """TRUNCATE queries should be rejected."""
        is_readonly, keyword = db_query.is_read_only_query("TRUNCATE TABLE users")
        assert is_readonly is False
        assert keyword == "TRUNCATE"

    def test_grant_revoke_rejected(self):
        """GRANT and REVOKE should be rejected."""
        is_readonly, keyword = db_query.is_read_only_query(
            "GRANT SELECT ON users TO public"
        )
        assert is_readonly is False
        assert keyword == "GRANT"

        is_readonly, keyword = db_query.is_read_only_query(
            "REVOKE SELECT ON users FROM public"
        )
        assert is_readonly is False
        assert keyword == "REVOKE"

    def test_comments_are_ignored(self):
        """SQL comments should be stripped before checking."""
        # Comment containing INSERT should be ignored
        query = """
        -- This is a comment about INSERT operations
        SELECT * FROM users
        """
        assert db_query.is_read_only_query(query)[0] is True

        # Block comment with DELETE
        query = """
        /* DELETE operations are not allowed */
        SELECT * FROM users
        """
        assert db_query.is_read_only_query(query)[0] is True

    def test_keyword_in_string_literal_false_positive(self):
        """Keywords in column names that look like SQL should be caught.

        Note: This test documents current behavior - the tool errs on the
        side of caution and may have false positives for unusual queries.
        """
        # This is a known limitation - the simple regex approach may flag
        # innocent queries. In practice, this is rare and safer than allowing
        # potential injections.
        query = "SELECT * FROM users WHERE description LIKE '%INSERT%'"
        # Current implementation will flag this as it contains INSERT
        is_readonly, _ = db_query.is_read_only_query(query)
        # This documents the current (conservative) behavior
        assert is_readonly is False


class TestParseConnectionString:
    """Tests for the parse_connection_string function."""

    def test_postgresql_full_url(self):
        """Parse a complete PostgreSQL connection string."""
        result = db_query.parse_connection_string(
            "postgresql://myuser:mypass@localhost:5432/mydb"
        )
        assert result['type'] == 'postgresql'
        assert result['host'] == 'localhost'
        assert result['port'] == 5432
        assert result['user'] == 'myuser'
        assert result['password'] == 'mypass'
        assert result['database'] == 'mydb'

    def test_postgresql_postgres_scheme(self):
        """Both 'postgres' and 'postgresql' schemes should work."""
        result = db_query.parse_connection_string(
            "postgres://user:pass@host/db"
        )
        assert result['type'] == 'postgresql'

    def test_mysql_full_url(self):
        """Parse a complete MySQL connection string."""
        result = db_query.parse_connection_string(
            "mysql://myuser:mypass@localhost:3306/mydb"
        )
        assert result['type'] == 'mysql'
        assert result['host'] == 'localhost'
        assert result['port'] == 3306
        assert result['user'] == 'myuser'
        assert result['password'] == 'mypass'
        assert result['database'] == 'mydb'

    def test_sqlite_file_path(self):
        """Parse SQLite file path."""
        result = db_query.parse_connection_string("sqlite:///app.db")
        assert result['type'] == 'sqlite'
        assert result['database'] == 'app.db'

    def test_sqlite_absolute_path(self):
        """Parse SQLite with absolute path."""
        result = db_query.parse_connection_string("sqlite:////var/data/app.db")
        assert result['type'] == 'sqlite'
        assert result['database'] == '/var/data/app.db'

    def test_sqlite_memory(self):
        """Parse SQLite in-memory database."""
        result = db_query.parse_connection_string("sqlite:///:memory:")
        assert result['type'] == 'sqlite'
        assert result['database'] == ':memory:'

    def test_missing_database_raises_error(self):
        """Connection string without database name should raise ValueError."""
        with pytest.raises(ValueError, match="Database name is required"):
            db_query.parse_connection_string("postgresql://user:pass@localhost/")

    def test_unsupported_database_type(self):
        """Unsupported database types should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported database type"):
            db_query.parse_connection_string("mongodb://localhost/test")

    def test_default_host(self):
        """Missing host should default to localhost."""
        result = db_query.parse_connection_string("postgresql:///mydb")
        assert result['host'] == 'localhost'


class TestFormatOutput:
    """Tests for output formatting functions."""

    def test_format_table_basic(self):
        """Test basic table formatting."""
        columns = ['id', 'name']
        rows = [(1, 'Alice'), (2, 'Bob')]
        output = db_query.format_table(columns, rows)

        assert 'id' in output
        assert 'name' in output
        assert 'Alice' in output
        assert 'Bob' in output
        assert '(2 rows)' in output

    def test_format_table_single_row(self):
        """Test table formatting with single row (singular 'row')."""
        columns = ['id']
        rows = [(1,)]
        output = db_query.format_table(columns, rows)
        assert '(1 row)' in output

    def test_format_table_empty_results(self):
        """Test table formatting with no rows."""
        columns = ['id', 'name']
        rows = []
        output = db_query.format_table(columns, rows)
        assert '(0 rows)' in output

    def test_format_table_no_columns(self):
        """Test table formatting with no columns."""
        output = db_query.format_table([], [])
        assert 'no columns' in output.lower()

    def test_format_table_null_values(self):
        """Test that NULL values are displayed correctly."""
        columns = ['id', 'name']
        rows = [(1, None), (2, 'Bob')]
        output = db_query.format_table(columns, rows)
        assert 'NULL' in output

    def test_format_csv_basic(self):
        """Test basic CSV formatting."""
        columns = ['id', 'name']
        rows = [(1, 'Alice'), (2, 'Bob')]
        output = db_query.format_csv(columns, rows)

        # Split and strip to handle any line ending style
        lines = [line.strip() for line in output.strip().split('\n')]
        assert lines[0] == 'id,name'
        assert lines[1] == '1,Alice'
        assert lines[2] == '2,Bob'

    def test_format_csv_null_values(self):
        """Test CSV formatting with NULL values."""
        columns = ['id', 'value']
        rows = [(1, None)]
        output = db_query.format_csv(columns, rows)

        lines = [line.strip() for line in output.strip().split('\n')]
        assert lines[1] == '1,'  # None becomes empty string

    def test_format_csv_special_characters(self):
        """Test CSV properly escapes special characters."""
        columns = ['id', 'text']
        rows = [(1, 'Hello, "World"')]
        output = db_query.format_csv(columns, rows)

        # CSV should quote and escape the value
        assert '"Hello, ""World"""' in output or "'Hello, \"World\"'" in output

    def test_format_json_basic(self):
        """Test basic JSON formatting."""
        columns = ['id', 'name']
        rows = [(1, 'Alice'), (2, 'Bob')]
        output = db_query.format_json(columns, rows)

        data = json.loads(output)
        assert data['columns'] == ['id', 'name']
        assert data['count'] == 2
        assert len(data['rows']) == 2
        assert data['rows'][0] == {'id': 1, 'name': 'Alice'}

    def test_format_json_null_values(self):
        """Test JSON formatting with NULL values."""
        columns = ['id', 'value']
        rows = [(1, None)]
        output = db_query.format_json(columns, rows)

        data = json.loads(output)
        assert data['rows'][0]['value'] is None

    def test_format_json_empty(self):
        """Test JSON formatting with empty results."""
        output = db_query.format_json([], [])
        data = json.loads(output)
        assert data['columns'] == []
        assert data['rows'] == []
        assert data['count'] == 0


class TestConnectFunctions:
    """Tests for database connection functions using mocks."""

    @mock.patch.dict('sys.modules', {'psycopg': mock.MagicMock()})
    def test_connect_postgresql_with_psycopg(self):
        """Test PostgreSQL connection with psycopg (v3)."""
        import sys
        mock_psycopg = sys.modules['psycopg']
        mock_conn = mock.MagicMock()
        mock_psycopg.connect.return_value = mock_conn

        params = {
            'type': 'postgresql',
            'host': 'localhost',
            'port': 5432,
            'user': 'testuser',
            'password': 'testpass',
            'database': 'testdb'
        }

        # We need to reload the module to pick up the mock
        # For this test, we'll directly test the connection logic
        result = db_query.connect_postgresql(params)

        mock_psycopg.connect.assert_called_once()
        call_kwargs = mock_psycopg.connect.call_args[1]
        assert call_kwargs['host'] == 'localhost'
        assert call_kwargs['dbname'] == 'testdb'

    def test_connect_mysql_import_error(self):
        """Test MySQL connection raises ImportError when driver not installed."""
        # Remove mysql.connector from sys.modules if present
        with mock.patch.dict('sys.modules', {'mysql': None, 'mysql.connector': None}):
            params = {
                'type': 'mysql',
                'host': 'localhost',
                'port': 3306,
                'user': 'testuser',
                'password': 'testpass',
                'database': 'testdb'
            }

            with pytest.raises(ImportError, match="mysql-connector-python"):
                db_query.connect_mysql(params)

    def test_connect_sqlite_file(self):
        """Test SQLite file connection."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name

        try:
            params = {'type': 'sqlite', 'database': temp_db}
            conn = db_query.connect_sqlite(params)

            assert conn is not None
            conn.close()
        finally:
            os.unlink(temp_db)

    def test_connect_sqlite_memory(self):
        """Test SQLite in-memory connection."""
        params = {'type': 'sqlite', 'database': ':memory:'}
        conn = db_query.connect_sqlite(params)

        assert conn is not None
        # Verify we can execute queries
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        assert cursor.fetchone() == (1,)
        conn.close()


class TestExecuteQuery:
    """Tests for query execution."""

    def test_execute_select_query(self):
        """Test executing a SELECT query."""
        # Use SQLite in-memory for real query testing
        import sqlite3
        conn = sqlite3.connect(':memory:')
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")

        columns, rows = db_query.execute_query(conn, "SELECT * FROM test")

        assert columns == ['id', 'name']
        assert len(rows) == 2
        assert (1, 'Alice') in rows
        conn.close()

    def test_execute_query_no_results(self):
        """Test query that returns no rows."""
        import sqlite3
        conn = sqlite3.connect(':memory:')
        conn.execute("CREATE TABLE test (id INTEGER)")

        columns, rows = db_query.execute_query(conn, "SELECT * FROM test")

        assert columns == ['id']
        assert rows == []
        conn.close()


class TestGetQuery:
    """Tests for query input handling."""

    def test_get_query_from_args(self):
        """Test getting query from -q argument."""
        args = mock.MagicMock()
        args.query = "SELECT * FROM users"
        args.file = None

        result = db_query.get_query(args)
        assert result == "SELECT * FROM users"

    def test_get_query_from_file(self):
        """Test getting query from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            f.write("SELECT * FROM orders")
            temp_file = f.name

        try:
            args = mock.MagicMock()
            args.query = None
            args.file = temp_file

            result = db_query.get_query(args)
            assert result == "SELECT * FROM orders"
        finally:
            os.unlink(temp_file)

    def test_get_query_file_not_found(self):
        """Test error when query file doesn't exist."""
        args = mock.MagicMock()
        args.query = None
        args.file = '/nonexistent/file.sql'

        with pytest.raises(FileNotFoundError):
            db_query.get_query(args)

    @mock.patch('sys.stdin')
    def test_get_query_from_stdin(self, mock_stdin):
        """Test getting query from stdin."""
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "SELECT COUNT(*) FROM products\n"

        args = mock.MagicMock()
        args.query = None
        args.file = None

        result = db_query.get_query(args)
        assert result == "SELECT COUNT(*) FROM products"

    @mock.patch('sys.stdin')
    def test_get_query_no_input(self, mock_stdin):
        """Test when no query input is provided."""
        mock_stdin.isatty.return_value = True  # Interactive terminal

        args = mock.MagicMock()
        args.query = None
        args.file = None

        result = db_query.get_query(args)
        assert result is None


class TestMain:
    """Integration tests for the main function."""

    def test_main_no_database(self, capsys):
        """Test error when no database is specified."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove DATABASE_URL if it exists
            os.environ.pop('DATABASE_URL', None)

            exit_code = db_query.main(['-q', 'SELECT 1'])

        assert exit_code == db_query.EXIT_INVALID_ARGS
        captured = capsys.readouterr()
        assert 'No database specified' in captured.err

    @mock.patch('sys.stdin')
    def test_main_no_query(self, mock_stdin, capsys):
        """Test error when no query is provided."""
        mock_stdin.isatty.return_value = True  # Simulate interactive terminal

        exit_code = db_query.main(['--db', 'sqlite:///:memory:'])

        assert exit_code == db_query.EXIT_INVALID_ARGS
        captured = capsys.readouterr()
        assert 'No query provided' in captured.err

    def test_main_write_query_rejected(self, capsys):
        """Test that write queries are rejected."""
        exit_code = db_query.main([
            '--db', 'sqlite:///:memory:',
            '-q', 'INSERT INTO users VALUES (1, "test")'
        ])

        assert exit_code == db_query.EXIT_QUERY_ERROR
        captured = capsys.readouterr()
        assert 'rejected' in captured.err.lower()
        assert 'INSERT' in captured.err

    def test_main_dry_run(self, capsys):
        """Test dry-run mode."""
        exit_code = db_query.main([
            '--db', 'sqlite:///:memory:',
            '-q', 'SELECT * FROM users',
            '--dry-run'
        ])

        assert exit_code == db_query.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'validation passed' in captured.err.lower()

    def test_main_dry_run_rejected(self, capsys):
        """Test dry-run mode with invalid query."""
        exit_code = db_query.main([
            '--db', 'sqlite:///:memory:',
            '-q', 'DROP TABLE users',
            '--dry-run'
        ])

        assert exit_code == db_query.EXIT_QUERY_ERROR
        captured = capsys.readouterr()
        assert 'rejected' in captured.err.lower()

    def test_main_sqlite_query_success(self, capsys, tmp_path):
        """Test successful SQLite query execution."""
        # Create a temp database with test data
        temp_db = tmp_path / "test.db"

        import sqlite3
        conn = sqlite3.connect(str(temp_db))
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
        conn.commit()
        conn.close()

        exit_code = db_query.main([
            '--db', f'sqlite:///{temp_db}',
            '-q', 'SELECT * FROM test'
        ])

        assert exit_code == db_query.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'Alice' in captured.out
        assert 'Bob' in captured.out

    def test_main_json_output(self, capsys, tmp_path):
        """Test JSON output format."""
        temp_db = tmp_path / "test.db"

        import sqlite3
        conn = sqlite3.connect(str(temp_db))
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'Test')")
        conn.commit()
        conn.close()

        exit_code = db_query.main([
            '--db', f'sqlite:///{temp_db}',
            '-q', 'SELECT * FROM test',
            '--format', 'json'
        ])

        assert exit_code == db_query.EXIT_SUCCESS
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data['count'] == 1
        assert data['rows'][0]['name'] == 'Test'

    def test_main_csv_output(self, capsys, tmp_path):
        """Test CSV output format."""
        temp_db = tmp_path / "test.db"

        import sqlite3
        conn = sqlite3.connect(str(temp_db))
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'Test')")
        conn.commit()
        conn.close()

        exit_code = db_query.main([
            '--db', f'sqlite:///{temp_db}',
            '-q', 'SELECT * FROM test',
            '--format', 'csv'
        ])

        assert exit_code == db_query.EXIT_SUCCESS
        captured = capsys.readouterr()
        lines = [line.strip() for line in captured.out.strip().split('\n')]
        assert lines[0] == 'id,name'
        assert lines[1] == '1,Test'

    def test_main_output_to_file(self, capsys, tmp_path):
        """Test writing output to file."""
        temp_db = tmp_path / "test.db"
        output_file = tmp_path / "output.csv"

        import sqlite3
        conn = sqlite3.connect(str(temp_db))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.execute("INSERT INTO test VALUES (1)")
        conn.commit()
        conn.close()

        exit_code = db_query.main([
            '--db', f'sqlite:///{temp_db}',
            '-q', 'SELECT * FROM test',
            '--format', 'csv',
            '-o', str(output_file)
        ])

        assert exit_code == db_query.EXIT_SUCCESS

        with open(output_file, 'r') as f:
            content = f.read()
        assert 'id' in content
        assert '1' in content

    def test_main_verbose_mode(self, capsys, tmp_path):
        """Test verbose output mode."""
        temp_db = tmp_path / "test.db"

        import sqlite3
        conn = sqlite3.connect(str(temp_db))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

        exit_code = db_query.main([
            '--db', f'sqlite:///{temp_db}',
            '-q', 'SELECT * FROM test',
            '-v'
        ])

        assert exit_code == db_query.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert 'Database:' in captured.err
        assert 'Connected' in captured.err

    def test_main_query_from_file(self, capsys, tmp_path):
        """Test query input from file."""
        temp_db = tmp_path / "test.db"
        query_file = tmp_path / "query.sql"

        import sqlite3
        conn = sqlite3.connect(str(temp_db))
        conn.close()

        with open(query_file, 'w') as f:
            f.write("SELECT 42 as answer")

        exit_code = db_query.main([
            '--db', f'sqlite:///{temp_db}',
            '-f', str(query_file)
        ])

        assert exit_code == db_query.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert '42' in captured.out

    def test_main_database_url_env_var(self, capsys, tmp_path):
        """Test using DATABASE_URL environment variable."""
        temp_db = tmp_path / "test.db"

        import sqlite3
        conn = sqlite3.connect(str(temp_db))
        conn.execute("CREATE TABLE test (val INTEGER)")
        conn.execute("INSERT INTO test VALUES (999)")
        conn.commit()
        conn.close()

        with mock.patch.dict(os.environ, {'DATABASE_URL': f'sqlite:///{temp_db}'}):
            exit_code = db_query.main(['-q', 'SELECT val FROM test'])

        assert exit_code == db_query.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert '999' in captured.out

    def test_main_invalid_connection_string(self, capsys):
        """Test error handling for invalid connection string."""
        exit_code = db_query.main([
            '--db', 'invalid://something',
            '-q', 'SELECT 1'
        ])

        assert exit_code == db_query.EXIT_INVALID_ARGS
        captured = capsys.readouterr()
        assert 'Unsupported database type' in captured.err

    def test_main_connection_failure(self, capsys):
        """Test handling of connection failure."""
        # Try to connect to a non-existent directory path
        # This should fail with connection error
        exit_code = db_query.main([
            '--db', 'sqlite:///nonexistent/path/to/db.sqlite',
            '-q', 'SELECT 1'
        ])

        # SQLite cannot create the file because the directory doesn't exist
        # So this will fail with a connection error
        assert exit_code == db_query.EXIT_CONNECTION_ERROR

    def test_main_query_execution_error(self, capsys, tmp_path):
        """Test handling of query execution error."""
        temp_db = tmp_path / "test.db"

        import sqlite3
        conn = sqlite3.connect(str(temp_db))
        conn.close()

        # Query a non-existent table
        exit_code = db_query.main([
            '--db', f'sqlite:///{temp_db}',
            '-q', 'SELECT * FROM nonexistent_table'
        ])

        assert exit_code == db_query.EXIT_QUERY_ERROR
        captured = capsys.readouterr()
        assert 'Query execution failed' in captured.err


class TestArgumentParser:
    """Tests for argument parsing."""

    def test_parser_help(self, capsys):
        """Test that --help works and includes examples."""
        with pytest.raises(SystemExit) as exc_info:
            db_query.main(['--help'])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert 'postgresql://' in captured.out
        assert 'sqlite:///' in captured.out
        assert 'EXAMPLES' in captured.out
        assert 'EXIT CODES' in captured.out

    def test_parser_version(self, capsys):
        """Test that --version works."""
        with pytest.raises(SystemExit) as exc_info:
            db_query.main(['--version'])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert '1.0.0' in captured.out

    def test_mutually_exclusive_query_inputs(self, capsys):
        """Test that -q and -f are mutually exclusive."""
        with pytest.raises(SystemExit) as exc_info:
            db_query.main([
                '--db', 'sqlite:///:memory:',
                '-q', 'SELECT 1',
                '-f', 'query.sql'
            ])

        assert exc_info.value.code == 2  # argparse error exit code


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @mock.patch('sys.stdin')
    def test_empty_query(self, mock_stdin, capsys):
        """Test handling of empty query."""
        mock_stdin.isatty.return_value = True

        exit_code = db_query.main([
            '--db', 'sqlite:///:memory:',
            '-q', '   '
        ])

        assert exit_code == db_query.EXIT_INVALID_ARGS

    def test_multiline_query(self, capsys, tmp_path):
        """Test handling of multiline queries."""
        temp_db = tmp_path / "test.db"

        import sqlite3
        conn = sqlite3.connect(str(temp_db))
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'Test')")
        conn.commit()
        conn.close()

        query = """
        SELECT
            id,
            name
        FROM
            test
        WHERE
            id = 1
        """

        exit_code = db_query.main([
            '--db', f'sqlite:///{temp_db}',
            '-q', query
        ])

        assert exit_code == db_query.EXIT_SUCCESS

    def test_unicode_data(self, capsys, tmp_path):
        """Test handling of unicode data in results."""
        temp_db = tmp_path / "test.db"

        import sqlite3
        conn = sqlite3.connect(str(temp_db))
        conn.execute("CREATE TABLE test (name TEXT)")
        conn.execute("INSERT INTO test VALUES ('Hello 世界')")
        conn.commit()
        conn.close()

        exit_code = db_query.main([
            '--db', f'sqlite:///{temp_db}',
            '-q', 'SELECT * FROM test'
        ])

        assert exit_code == db_query.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert '世界' in captured.out

    def test_large_result_set(self, capsys, tmp_path):
        """Test handling of larger result sets."""
        temp_db = tmp_path / "test.db"

        import sqlite3
        conn = sqlite3.connect(str(temp_db))
        conn.execute("CREATE TABLE test (id INTEGER)")
        # Insert 1000 rows
        conn.executemany("INSERT INTO test VALUES (?)",
                       [(i,) for i in range(1000)])
        conn.commit()
        conn.close()

        exit_code = db_query.main([
            '--db', f'sqlite:///{temp_db}',
            '-q', 'SELECT COUNT(*) as cnt FROM test',
            '--format', 'json'
        ])

        assert exit_code == db_query.EXIT_SUCCESS
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data['rows'][0]['cnt'] == 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
