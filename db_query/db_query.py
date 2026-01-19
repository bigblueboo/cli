#!/usr/bin/env python3
"""
db_query - A self-documenting CLI tool for executing READ-ONLY SQL queries.

This tool supports PostgreSQL, MySQL, and SQLite databases via connection strings.
It enforces read-only access by rejecting any queries that could modify data.

Exit Codes:
    0 - Success
    1 - Connection error
    2 - Query error
    3 - Invalid arguments

Environment Variables:
    DATABASE_URL - Default database connection string (optional)

Examples:
    # PostgreSQL query with inline SQL
    db_query --db "postgresql://user:pass@host/db" -q "SELECT * FROM users LIMIT 10"

    # SQLite query from file, output as CSV
    db_query --db "sqlite:///app.db" -f report.sql --format csv

    # Query from stdin with JSON output
    echo "SELECT count(*) FROM orders" | db_query --format json

    # Using DATABASE_URL environment variable
    export DATABASE_URL="mysql://user:pass@localhost/mydb"
    db_query -q "SELECT * FROM products WHERE price > 100"

Author: Generated CLI Tool
License: MIT
"""

import argparse
import csv
import io
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Exit codes
EXIT_SUCCESS = 0
EXIT_CONNECTION_ERROR = 1
EXIT_QUERY_ERROR = 2
EXIT_INVALID_ARGS = 3

# SQL keywords that indicate write operations (case-insensitive)
WRITE_KEYWORDS = [
    r'\bINSERT\b',
    r'\bUPDATE\b',
    r'\bDELETE\b',
    r'\bDROP\b',
    r'\bCREATE\b',
    r'\bALTER\b',
    r'\bTRUNCATE\b',
    r'\bREPLACE\b',
    r'\bMERGE\b',
    r'\bGRANT\b',
    r'\bREVOKE\b',
    r'\bEXEC\b',
    r'\bEXECUTE\b',
    r'\bCALL\b',
    r'\bRENAME\b',
    r'\bLOAD\b',
    r'\bIMPORT\b',
    r'\bVACUUM\b',
    r'\bREINDEX\b',
    r'\bATTACH\b',
    r'\bDETACH\b',
]


def is_read_only_query(query: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a SQL query is read-only.

    Args:
        query: The SQL query string to validate.

    Returns:
        A tuple of (is_read_only, violation_keyword).
        If read-only, returns (True, None).
        If not read-only, returns (False, keyword_that_violated).
    """
    # Remove comments (both -- style and /* */ style)
    query_clean = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
    query_clean = re.sub(r'/\*.*?\*/', '', query_clean, flags=re.DOTALL)

    # Check for write keywords
    for pattern in WRITE_KEYWORDS:
        match = re.search(pattern, query_clean, re.IGNORECASE)
        if match:
            return False, match.group(0).strip().upper()

    return True, None


def parse_connection_string(conn_str: str) -> Dict[str, Any]:
    """
    Parse a database connection string into components.

    Supported formats:
        - postgresql://user:pass@host:port/dbname
        - mysql://user:pass@host:port/dbname
        - sqlite:///path/to/database.db
        - sqlite:///:memory:

    Args:
        conn_str: The connection string to parse.

    Returns:
        A dictionary with connection parameters and 'type' key.

    Raises:
        ValueError: If the connection string format is invalid.
    """
    parsed = urlparse(conn_str)

    scheme = parsed.scheme.lower()

    # Normalize scheme names
    if scheme in ('postgres', 'postgresql'):
        db_type = 'postgresql'
    elif scheme in ('mysql', 'mysql+mysqlconnector'):
        db_type = 'mysql'
    elif scheme == 'sqlite':
        db_type = 'sqlite'
    else:
        raise ValueError(f"Unsupported database type: {scheme}. "
                        f"Supported types: postgresql, mysql, sqlite")

    result = {'type': db_type}

    if db_type == 'sqlite':
        # SQLite path handling
        # sqlite:///path/to/db.sqlite -> path/to/db.sqlite (relative)
        # sqlite:////var/data/app.db -> /var/data/app.db (absolute - note 4 slashes)
        # sqlite:///:memory: -> :memory:
        path = parsed.path
        if parsed.netloc:
            # Handle cases like sqlite://localhost/path
            path = parsed.netloc + path

        # Handle :memory: special case
        if path == '/:memory:' or path == ':memory:':
            result['database'] = ':memory:'
        elif path.startswith('//'):
            # Absolute path: sqlite:////var/data/app.db -> //var/data/app.db -> /var/data/app.db
            result['database'] = path[1:]
        elif path.startswith('/'):
            # Relative path: sqlite:///app.db -> /app.db -> app.db
            result['database'] = path[1:]
        else:
            result['database'] = path
    else:
        result['host'] = parsed.hostname or 'localhost'
        result['port'] = parsed.port
        result['user'] = parsed.username
        result['password'] = parsed.password
        result['database'] = parsed.path.lstrip('/') if parsed.path else None

        if not result['database']:
            raise ValueError("Database name is required in connection string")

    return result


def connect_postgresql(params: Dict[str, Any]):
    """
    Create a PostgreSQL database connection.

    Args:
        params: Connection parameters from parse_connection_string.

    Returns:
        A database connection object.

    Raises:
        ImportError: If psycopg is not installed.
        Exception: If connection fails.
    """
    try:
        import psycopg
    except ImportError:
        try:
            import psycopg2 as psycopg
        except ImportError:
            raise ImportError(
                "PostgreSQL support requires 'psycopg' or 'psycopg2'. "
                "Install with: pip install 'psycopg[binary]' or pip install psycopg2-binary"
            )

    connect_params = {
        'host': params.get('host', 'localhost'),
        'dbname': params['database'],
    }

    if params.get('port'):
        connect_params['port'] = params['port']
    if params.get('user'):
        connect_params['user'] = params['user']
    if params.get('password'):
        connect_params['password'] = params['password']

    return psycopg.connect(**connect_params)


def connect_mysql(params: Dict[str, Any]):
    """
    Create a MySQL database connection.

    Args:
        params: Connection parameters from parse_connection_string.

    Returns:
        A database connection object.

    Raises:
        ImportError: If mysql-connector-python is not installed.
        Exception: If connection fails.
    """
    try:
        import mysql.connector
    except ImportError:
        raise ImportError(
            "MySQL support requires 'mysql-connector-python'. "
            "Install with: pip install mysql-connector-python"
        )

    connect_params = {
        'host': params.get('host', 'localhost'),
        'database': params['database'],
    }

    if params.get('port'):
        connect_params['port'] = params['port']
    if params.get('user'):
        connect_params['user'] = params['user']
    if params.get('password'):
        connect_params['password'] = params['password']

    return mysql.connector.connect(**connect_params)


def connect_sqlite(params: Dict[str, Any]):
    """
    Create a SQLite database connection.

    Args:
        params: Connection parameters from parse_connection_string.

    Returns:
        A database connection object.
    """
    import sqlite3

    database = params['database']
    if database == ':memory:':
        return sqlite3.connect(':memory:')
    return sqlite3.connect(database)


def get_connection(conn_str: str):
    """
    Create a database connection from a connection string.

    Args:
        conn_str: Database connection string.

    Returns:
        A tuple of (connection, db_type).

    Raises:
        ValueError: If connection string is invalid.
        ImportError: If required driver is not installed.
        Exception: If connection fails.
    """
    params = parse_connection_string(conn_str)
    db_type = params['type']

    if db_type == 'postgresql':
        conn = connect_postgresql(params)
    elif db_type == 'mysql':
        conn = connect_mysql(params)
    elif db_type == 'sqlite':
        conn = connect_sqlite(params)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

    return conn, db_type


def execute_query(conn, query: str) -> Tuple[List[str], List[Tuple]]:
    """
    Execute a SQL query and return results.

    Args:
        conn: Database connection object.
        query: SQL query to execute.

    Returns:
        A tuple of (column_names, rows).

    Raises:
        Exception: If query execution fails.
    """
    cursor = conn.cursor()
    cursor.execute(query)

    # Get column names
    if cursor.description:
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    else:
        columns = []
        rows = []

    cursor.close()
    return columns, rows


def format_table(columns: List[str], rows: List[Tuple]) -> str:
    """
    Format query results as an ASCII table.

    Args:
        columns: List of column names.
        rows: List of row tuples.

    Returns:
        Formatted table string.
    """
    if not columns:
        return "Query returned no columns."

    if not rows:
        return f"Columns: {', '.join(columns)}\n(0 rows)"

    # Calculate column widths
    widths = [len(str(col)) for col in columns]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val) if val is not None else 'NULL'))

    # Build separator
    separator = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    # Build header
    header = '|' + '|'.join(f" {str(col).ljust(widths[i])} " for i, col in enumerate(columns)) + '|'

    # Build rows
    lines = [separator, header, separator]
    for row in rows:
        row_str = '|' + '|'.join(
            f" {(str(val) if val is not None else 'NULL').ljust(widths[i])} "
            for i, val in enumerate(row)
        ) + '|'
        lines.append(row_str)
    lines.append(separator)
    lines.append(f"({len(rows)} row{'s' if len(rows) != 1 else ''})")

    return '\n'.join(lines)


def format_csv(columns: List[str], rows: List[Tuple]) -> str:
    """
    Format query results as CSV.

    Args:
        columns: List of column names.
        rows: List of row tuples.

    Returns:
        CSV formatted string.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    if columns:
        writer.writerow(columns)

    for row in rows:
        writer.writerow([val if val is not None else '' for val in row])

    return output.getvalue()


def format_json(columns: List[str], rows: List[Tuple]) -> str:
    """
    Format query results as JSON.

    Args:
        columns: List of column names.
        rows: List of row tuples.

    Returns:
        JSON formatted string.
    """
    if not columns:
        return json.dumps({"columns": [], "rows": [], "count": 0}, indent=2)

    result = {
        "columns": columns,
        "rows": [
            {col: (val if val is not None else None) for col, val in zip(columns, row)}
            for row in rows
        ],
        "count": len(rows)
    }

    # Custom encoder to handle types that aren't JSON serializable
    def json_serializer(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        if hasattr(obj, '__str__'):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json.dumps(result, indent=2, default=json_serializer)


def format_output(columns: List[str], rows: List[Tuple], fmt: str) -> str:
    """
    Format query results in the specified format.

    Args:
        columns: List of column names.
        rows: List of row tuples.
        fmt: Output format ('table', 'csv', 'json').

    Returns:
        Formatted output string.
    """
    if fmt == 'table':
        return format_table(columns, rows)
    elif fmt == 'csv':
        return format_csv(columns, rows)
    elif fmt == 'json':
        return format_json(columns, rows)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog='db_query',
        description='''
Execute READ-ONLY SQL queries against PostgreSQL, MySQL, or SQLite databases.

This tool enforces read-only access by rejecting any queries containing
INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, and other data
modification statements.

SUPPORTED DATABASE TYPES:
  - PostgreSQL: postgresql://user:pass@host:port/dbname
  - MySQL:      mysql://user:pass@host:port/dbname
  - SQLite:     sqlite:///path/to/database.db
                sqlite:///:memory:

EXIT CODES:
  0  Success - query executed successfully
  1  Connection error - could not connect to database
  2  Query error - query execution failed or was rejected
  3  Invalid arguments - missing or invalid command-line arguments
''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
EXAMPLES:
  # PostgreSQL with inline query
  %(prog)s --db "postgresql://user:pass@localhost/mydb" -q "SELECT * FROM users LIMIT 5"

  # SQLite with query from file, CSV output
  %(prog)s --db "sqlite:///data.db" -f queries/report.sql --format csv > report.csv

  # MySQL using DATABASE_URL environment variable
  export DATABASE_URL="mysql://root:secret@localhost/shop"
  %(prog)s -q "SELECT name, price FROM products ORDER BY price DESC LIMIT 10"

  # Query from stdin with JSON output
  echo "SELECT COUNT(*) as total FROM orders" | %(prog)s --format json

  # Combine file query with piped output
  %(prog)s --db "postgresql://user@localhost/db" -f monthly_sales.sql --format csv | head -20

ENVIRONMENT VARIABLES:
  DATABASE_URL   Default database connection string. Can be overridden with --db.

SECURITY:
  This tool enforces READ-ONLY access. The following SQL commands are blocked:
  INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, REPLACE, MERGE,
  GRANT, REVOKE, EXEC/EXECUTE, CALL, RENAME, LOAD, IMPORT, VACUUM, REINDEX,
  ATTACH, DETACH

For more information, see: https://github.com/example/db_query
''')

    # Database connection options
    db_group = parser.add_argument_group('Database Connection')
    db_group.add_argument(
        '--db', '--database',
        dest='database',
        metavar='URL',
        help='Database connection string (overrides DATABASE_URL env var). '
             'Format: type://user:pass@host:port/dbname'
    )

    # Query input options (mutually exclusive)
    query_group = parser.add_argument_group('Query Input (mutually exclusive)')
    query_input = query_group.add_mutually_exclusive_group()
    query_input.add_argument(
        '-q', '--query',
        metavar='SQL',
        help='SQL query string to execute'
    )
    query_input.add_argument(
        '-f', '--file',
        metavar='FILE',
        help='Path to file containing SQL query'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--format', '-F',
        choices=['table', 'csv', 'json'],
        default='table',
        help='Output format (default: table)'
    )
    output_group.add_argument(
        '--no-headers',
        action='store_true',
        help='Omit column headers in CSV output'
    )
    output_group.add_argument(
        '-o', '--output',
        metavar='FILE',
        help='Write output to file instead of stdout'
    )

    # Other options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (show connection info, timing, etc.)'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate query without executing (check read-only constraint)'
    )

    return parser


def get_query(args) -> Optional[str]:
    """
    Get the SQL query from arguments, file, or stdin.

    Args:
        args: Parsed command-line arguments.

    Returns:
        SQL query string, or None if no query provided.

    Raises:
        FileNotFoundError: If query file doesn't exist.
        IOError: If file can't be read.
    """
    if args.query:
        return args.query.strip()

    if args.file:
        with open(args.file, 'r') as f:
            return f.read().strip()

    # Check if stdin has data (not a TTY)
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    return None


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI tool.

    Args:
        args: Command-line arguments (uses sys.argv if None).

    Returns:
        Exit code (0=success, 1=connection error, 2=query error, 3=invalid args).
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Get database connection string
    db_url = parsed_args.database or os.environ.get('DATABASE_URL')

    if not db_url:
        print("Error: No database specified. Use --db or set DATABASE_URL environment variable.",
              file=sys.stderr)
        parser.print_usage(sys.stderr)
        return EXIT_INVALID_ARGS

    # Get query
    try:
        query = get_query(parsed_args)
    except FileNotFoundError as e:
        print(f"Error: Query file not found: {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS
    except IOError as e:
        print(f"Error: Could not read query file: {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS

    if not query:
        print("Error: No query provided. Use -q, -f, or pipe query via stdin.",
              file=sys.stderr)
        parser.print_usage(sys.stderr)
        return EXIT_INVALID_ARGS

    # Validate read-only
    is_readonly, violation = is_read_only_query(query)
    if not is_readonly:
        print(f"Error: Query rejected - contains forbidden keyword: {violation}",
              file=sys.stderr)
        print("This tool only supports READ-ONLY queries (SELECT, SHOW, DESCRIBE, etc.).",
              file=sys.stderr)
        return EXIT_QUERY_ERROR

    if parsed_args.verbose:
        print(f"Database: {db_url.split('@')[-1] if '@' in db_url else db_url}",
              file=sys.stderr)
        print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}",
              file=sys.stderr)

    # Dry run - just validate
    if parsed_args.dry_run:
        print("Query validation passed (read-only check OK).", file=sys.stderr)
        try:
            params = parse_connection_string(db_url)
            print(f"Connection string valid. Database type: {params['type']}", file=sys.stderr)
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)
        return EXIT_SUCCESS

    # Connect to database
    try:
        conn, db_type = get_connection(db_url)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CONNECTION_ERROR
    except ValueError as e:
        print(f"Error: Invalid connection string - {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS
    except Exception as e:
        print(f"Error: Could not connect to database - {e}", file=sys.stderr)
        return EXIT_CONNECTION_ERROR

    if parsed_args.verbose:
        print(f"Connected to {db_type} database.", file=sys.stderr)

    # Execute query
    try:
        columns, rows = execute_query(conn, query)
    except Exception as e:
        print(f"Error: Query execution failed - {e}", file=sys.stderr)
        conn.close()
        return EXIT_QUERY_ERROR
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if parsed_args.verbose:
        print(f"Query returned {len(rows)} row(s).", file=sys.stderr)

    # Format output
    try:
        output = format_output(columns, rows, parsed_args.format)

        # Handle --no-headers for CSV
        if parsed_args.no_headers and parsed_args.format == 'csv' and columns:
            lines = output.split('\n')
            output = '\n'.join(lines[1:])  # Skip header line
    except Exception as e:
        print(f"Error: Output formatting failed - {e}", file=sys.stderr)
        return EXIT_QUERY_ERROR

    # Write output
    if parsed_args.output:
        try:
            with open(parsed_args.output, 'w') as f:
                f.write(output)
            if parsed_args.verbose:
                print(f"Output written to: {parsed_args.output}", file=sys.stderr)
        except IOError as e:
            print(f"Error: Could not write to output file - {e}", file=sys.stderr)
            return EXIT_QUERY_ERROR
    else:
        print(output)

    return EXIT_SUCCESS


if __name__ == '__main__':
    sys.exit(main())
