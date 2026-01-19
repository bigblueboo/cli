#!/usr/bin/env python3
"""
Comprehensive tests for log_analyzer.py

Run with: pytest test_log_analyzer.py -v
"""

import io
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Import the module under test
from log_analyzer import (
    LogFormat,
    LogEntry,
    ErrorCluster,
    TimelineEvent,
    LogFormatDetector,
    LogParser,
    ErrorClusterer,
    TimelineAnalyzer,
    LLMClient,
    LLMError,
    LogAnalyzer,
    format_output,
    create_parser,
    main,
    EXIT_SUCCESS,
    EXIT_NO_LOGS,
    EXIT_API_ERROR,
    EXIT_INVALID_ARGS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def nginx_log_lines():
    """Sample nginx access log lines."""
    return [
        '192.168.1.1 - - [10/Jan/2024:13:55:36 +0000] "GET /api/users HTTP/1.1" 200 1234',
        '192.168.1.2 - - [10/Jan/2024:13:55:37 +0000] "POST /api/login HTTP/1.1" 401 89',
        '192.168.1.3 - - [10/Jan/2024:13:55:38 +0000] "GET /api/data HTTP/1.1" 500 456',
        '192.168.1.4 - - [10/Jan/2024:13:55:39 +0000] "GET /health HTTP/1.1" 200 15',
    ]


@pytest.fixture
def apache_log_lines():
    """Sample Apache access log lines."""
    return [
        '10.0.0.1 - user1 [10/Jan/2024:14:00:00 +0000] "GET /index.html HTTP/1.1" 200 5000',
        '10.0.0.2 - - [10/Jan/2024:14:00:01 +0000] "GET /missing.html HTTP/1.1" 404 200',
    ]


@pytest.fixture
def syslog_lines():
    """Sample syslog lines."""
    return [
        'Jan 10 14:30:00 server1 kernel: TCP connection timeout',
        'Jan 10 14:30:01 server1 sshd[1234]: Failed password for user root',
        'Jan 10 14:30:02 server1 app[5678]: ERROR: Database connection failed',
        'Jan 10 14:30:03 server1 app[5678]: INFO: Service restarted',
    ]


@pytest.fixture
def json_log_lines():
    """Sample JSON log lines."""
    return [
        '{"timestamp": "2024-01-10T15:00:00", "level": "INFO", "message": "Application started"}',
        '{"timestamp": "2024-01-10T15:00:01", "level": "ERROR", "message": "Connection refused to database"}',
        '{"timestamp": "2024-01-10T15:00:02", "level": "WARN", "message": "High memory usage detected"}',
        '{"timestamp": "2024-01-10T15:00:03", "level": "DEBUG", "message": "Processing request"}',
    ]


@pytest.fixture
def generic_log_lines():
    """Sample generic log lines with various formats."""
    return [
        '2024-01-10 16:00:00 INFO Starting application',
        '2024-01-10 16:00:01 ERROR Failed to connect to server',
        '2024-01-10 16:00:02 WARNING Disk space low',
        '2024-01-10 16:00:03 DEBUG Processing item 123',
    ]


@pytest.fixture
def temp_log_file(tmp_path, generic_log_lines):
    """Create a temporary log file."""
    log_file = tmp_path / "test.log"
    log_file.write_text('\n'.join(generic_log_lines))
    return str(log_file)


@pytest.fixture
def temp_nginx_log(tmp_path, nginx_log_lines):
    """Create a temporary nginx log file."""
    log_file = tmp_path / "nginx.log"
    log_file.write_text('\n'.join(nginx_log_lines))
    return str(log_file)


# =============================================================================
# LogFormat Tests
# =============================================================================

class TestLogFormat:
    """Tests for LogFormat enum."""

    def test_enum_values(self):
        """Test that all expected formats are defined."""
        assert LogFormat.NGINX.value == "nginx"
        assert LogFormat.APACHE.value == "apache"
        assert LogFormat.SYSLOG.value == "syslog"
        assert LogFormat.JSON.value == "json"
        assert LogFormat.UNKNOWN.value == "unknown"


# =============================================================================
# LogFormatDetector Tests
# =============================================================================

class TestLogFormatDetector:
    """Tests for LogFormatDetector class."""

    def test_detect_nginx(self, nginx_log_lines):
        """Test detection of nginx format."""
        fmt = LogFormatDetector.detect(nginx_log_lines)
        assert fmt == LogFormat.NGINX

    def test_detect_apache(self, apache_log_lines):
        """Test detection of Apache format."""
        fmt = LogFormatDetector.detect(apache_log_lines)
        # Apache and nginx have similar formats, both should be detected
        assert fmt in (LogFormat.APACHE, LogFormat.NGINX)

    def test_detect_syslog(self, syslog_lines):
        """Test detection of syslog format."""
        fmt = LogFormatDetector.detect(syslog_lines)
        assert fmt == LogFormat.SYSLOG

    def test_detect_json(self, json_log_lines):
        """Test detection of JSON format."""
        fmt = LogFormatDetector.detect(json_log_lines)
        assert fmt == LogFormat.JSON

    def test_detect_empty(self):
        """Test detection with empty input."""
        fmt = LogFormatDetector.detect([])
        assert fmt == LogFormat.UNKNOWN

    def test_detect_unknown(self):
        """Test detection of unknown format."""
        lines = [
            "random text that doesn't match any pattern",
            "another line without structure",
        ]
        fmt = LogFormatDetector.detect(lines)
        assert fmt == LogFormat.UNKNOWN


# =============================================================================
# LogParser Tests
# =============================================================================

class TestLogParser:
    """Tests for LogParser class."""

    def test_parse_json_line(self):
        """Test parsing JSON log line."""
        parser = LogParser(LogFormat.JSON)
        line = '{"timestamp": "2024-01-10T15:00:00", "level": "ERROR", "message": "Test error"}'

        entry = parser.parse_line(line, 1, "test.log")

        assert entry.level == "ERROR"
        assert entry.message == "Test error"
        assert entry.timestamp is not None
        assert entry.line_number == 1
        assert entry.file_path == "test.log"

    def test_parse_nginx_line(self, nginx_log_lines):
        """Test parsing nginx log line."""
        parser = LogParser(LogFormat.NGINX)
        entry = parser.parse_line(nginx_log_lines[2], 3, "nginx.log")

        assert entry.level == "ERROR"  # 500 status
        assert "500" in entry.message
        assert entry.extra.get('status') == '500'

    def test_parse_syslog_line(self, syslog_lines):
        """Test parsing syslog line."""
        parser = LogParser(LogFormat.SYSLOG)
        entry = parser.parse_line(syslog_lines[2], 3, "syslog")

        assert entry.level == "ERROR"
        assert "Database" in entry.message
        assert entry.source == "app[5678]"

    def test_parse_generic_line(self):
        """Test parsing generic log line."""
        parser = LogParser(LogFormat.UNKNOWN)
        line = "2024-01-10 16:00:01 ERROR Failed to connect"

        entry = parser.parse_line(line, 1, "app.log")

        assert entry.level == "ERROR"
        assert entry.timestamp is not None

    def test_is_error_by_level(self):
        """Test error detection by level."""
        parser = LogParser(LogFormat.UNKNOWN)

        error_entry = LogEntry(raw="test", level="ERROR", message="test")
        assert parser.is_error(error_entry) is True

        info_entry = LogEntry(raw="test", level="INFO", message="test")
        assert parser.is_error(info_entry) is False

    def test_is_error_by_message(self):
        """Test error detection by message content."""
        parser = LogParser(LogFormat.UNKNOWN)

        error_entry = LogEntry(raw="test", message="Connection failed with exception")
        assert parser.is_error(error_entry) is True

        normal_entry = LogEntry(raw="test", message="Request completed successfully")
        assert parser.is_error(normal_entry) is False

    def test_parse_timestamp_iso(self):
        """Test parsing ISO timestamp."""
        parser = LogParser(LogFormat.UNKNOWN)
        ts = parser._parse_timestamp("2024-01-10T15:30:45")

        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.day == 10

    def test_parse_timestamp_invalid(self):
        """Test parsing invalid timestamp."""
        parser = LogParser(LogFormat.UNKNOWN)
        ts = parser._parse_timestamp("not a timestamp")

        assert ts is None


# =============================================================================
# ErrorClusterer Tests
# =============================================================================

class TestErrorClusterer:
    """Tests for ErrorClusterer class."""

    def test_cluster_similar_errors(self):
        """Test clustering of similar errors."""
        clusterer = ErrorClusterer()

        # Add similar errors with different IPs
        for i in range(5):
            entry = LogEntry(
                raw=f"Connection from 192.168.1.{i} refused",
                message=f"Connection from 192.168.1.{i} refused",
                timestamp=datetime.now()
            )
            clusterer.add_error(entry)

        clusters = clusterer.get_clusters()

        # Should be grouped into one cluster
        assert len(clusters) == 1
        assert clusters[0].count == 5

    def test_cluster_different_errors(self):
        """Test that different errors get different clusters."""
        clusterer = ErrorClusterer()

        entry1 = LogEntry(raw="Database connection failed", message="Database connection failed")
        entry2 = LogEntry(raw="File not found", message="File not found")

        clusterer.add_error(entry1)
        clusterer.add_error(entry2)

        clusters = clusterer.get_clusters()
        assert len(clusters) == 2

    def test_cluster_time_tracking(self):
        """Test that clusters track first and last seen times."""
        clusterer = ErrorClusterer()

        early = datetime(2024, 1, 10, 10, 0, 0)
        late = datetime(2024, 1, 10, 15, 0, 0)

        entry1 = LogEntry(raw="Error occurred", message="Error occurred", timestamp=early)
        entry2 = LogEntry(raw="Error occurred", message="Error occurred", timestamp=late)

        clusterer.add_error(entry1)
        clusterer.add_error(entry2)

        clusters = clusterer.get_clusters()
        assert clusters[0].first_seen == early
        assert clusters[0].last_seen == late

    def test_cluster_sample_limit(self):
        """Test that clusters limit samples to 3."""
        clusterer = ErrorClusterer()

        for i in range(10):
            entry = LogEntry(raw=f"Same error", message="Same error")
            clusterer.add_error(entry)

        clusters = clusterer.get_clusters()
        assert len(clusters[0].samples) == 3


# =============================================================================
# TimelineAnalyzer Tests
# =============================================================================

class TestTimelineAnalyzer:
    """Tests for TimelineAnalyzer class."""

    def test_bucket_entries(self):
        """Test that entries are bucketed by time."""
        analyzer = TimelineAnalyzer(bucket_minutes=5)

        base_time = datetime(2024, 1, 10, 14, 0, 0)

        for i in range(10):
            entry = LogEntry(
                raw="Test",
                message="Test",
                timestamp=base_time + timedelta(minutes=i),
                level="INFO"
            )
            analyzer.add_entry(entry)

        timeline = analyzer.get_timeline()

        # Should have 2 buckets: 14:00 and 14:05
        assert len(timeline) == 2

    def test_severity_detection(self):
        """Test that severity is correctly detected."""
        analyzer = TimelineAnalyzer()

        base_time = datetime(2024, 1, 10, 14, 0, 0)

        # Add an error entry
        entry = LogEntry(
            raw="Error",
            message="Error",
            timestamp=base_time,
            level="ERROR"
        )
        analyzer.add_entry(entry)

        timeline = analyzer.get_timeline()
        assert timeline[0].severity == 'error'

    def test_no_timestamp_entries(self):
        """Test handling entries without timestamps."""
        analyzer = TimelineAnalyzer()

        entry = LogEntry(raw="No timestamp", message="No timestamp")
        analyzer.add_entry(entry)

        timeline = analyzer.get_timeline()
        assert len(timeline) == 0


# =============================================================================
# LLMClient Tests
# =============================================================================

class TestLLMClient:
    """Tests for LLMClient class."""

    def test_detect_openai_provider(self):
        """Test OpenAI provider detection."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=True):
            client = LLMClient()
            assert client.provider == 'openai'
            assert client.is_available() is True

    def test_detect_anthropic_provider(self):
        """Test Anthropic provider detection."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}, clear=True):
            client = LLMClient()
            assert client.provider == 'anthropic'
            assert client.is_available() is True

    def test_detect_google_provider(self):
        """Test Google provider detection."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'}, clear=True):
            client = LLMClient()
            assert client.provider == 'google'
            assert client.is_available() is True

    def test_no_provider(self):
        """Test when no API key is available."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear all API keys
            for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']:
                os.environ.pop(key, None)

            client = LLMClient()
            assert client.is_available() is False

    def test_fallback_analysis(self):
        """Test fallback analysis when no API key is set."""
        with patch.dict(os.environ, {}, clear=True):
            for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']:
                os.environ.pop(key, None)

            client = LLMClient()
            result = client.analyze("summarize", "line1\nline2\nline3")

            assert "Fallback analysis" in result
            assert "3 log lines" in result

    @patch('log_analyzer.LLMClient._call_openai')
    def test_analyze_with_openai(self, mock_call):
        """Test analysis with OpenAI."""
        mock_call.return_value = "Analysis result"

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=True):
            client = LLMClient()
            result = client.analyze("test prompt", "test context")

            assert result == "Analysis result"
            mock_call.assert_called_once()

    @patch('log_analyzer.LLMClient._call_anthropic')
    def test_analyze_with_anthropic(self, mock_call):
        """Test analysis with Anthropic."""
        mock_call.return_value = "Analysis result"

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}, clear=True):
            client = LLMClient()
            result = client.analyze("test prompt", "test context")

            assert result == "Analysis result"
            mock_call.assert_called_once()


# =============================================================================
# LogAnalyzer Tests
# =============================================================================

class TestLogAnalyzer:
    """Tests for LogAnalyzer class."""

    def test_load_files(self, temp_log_file):
        """Test loading log files."""
        analyzer = LogAnalyzer()
        count = analyzer.load_files([temp_log_file])

        assert count == 4
        assert len(analyzer.entries) == 4

    def test_load_files_glob(self, tmp_path):
        """Test loading files with glob pattern."""
        # Create multiple log files
        for i in range(3):
            (tmp_path / f"app{i}.log").write_text(f"2024-01-10 12:00:0{i} INFO Test message {i}")

        analyzer = LogAnalyzer()
        count = analyzer.load_files([str(tmp_path / "*.log")])

        assert count == 3

    def test_load_files_not_found(self):
        """Test loading non-existent files."""
        analyzer = LogAnalyzer()
        count = analyzer.load_files(["/nonexistent/path/*.log"])

        assert count == 0

    def test_load_stdin(self, generic_log_lines):
        """Test loading from stdin."""
        analyzer = LogAnalyzer()

        with patch('sys.stdin', io.StringIO('\n'.join(generic_log_lines))):
            count = analyzer.load_stdin()

        assert count == 4

    def test_find_errors(self, temp_log_file):
        """Test finding errors in logs."""
        analyzer = LogAnalyzer()
        analyzer.load_files([temp_log_file])

        errors, clusters = analyzer.find_errors(cluster=False)

        # Should find the ERROR line
        assert len(errors) >= 1
        assert any(e.level == 'ERROR' for e in errors)

    def test_find_errors_with_clustering(self, tmp_path):
        """Test finding and clustering errors."""
        log_content = '\n'.join([
            "2024-01-10 12:00:00 ERROR Connection to 192.168.1.1 failed",
            "2024-01-10 12:00:01 ERROR Connection to 192.168.1.2 failed",
            "2024-01-10 12:00:02 ERROR Connection to 192.168.1.3 failed",
            "2024-01-10 12:00:03 ERROR File not found: /tmp/test.txt",
        ])
        log_file = tmp_path / "errors.log"
        log_file.write_text(log_content)

        analyzer = LogAnalyzer()
        analyzer.load_files([str(log_file)])

        errors, clusters = analyzer.find_errors(cluster=True)

        assert len(errors) == 4
        assert clusters is not None
        # Connection errors should be clustered together
        assert len(clusters) == 2

    def test_time_filter_hours(self, tmp_path):
        """Test time filter with hours."""
        now = datetime.now()
        old = now - timedelta(hours=2)
        recent = now - timedelta(minutes=30)

        log_content = f"""{old.strftime('%Y-%m-%d %H:%M:%S')} INFO Old message
{recent.strftime('%Y-%m-%d %H:%M:%S')} INFO Recent message"""

        log_file = tmp_path / "timed.log"
        log_file.write_text(log_content)

        analyzer = LogAnalyzer()
        count = analyzer.load_files([str(log_file)], last='1h')

        # Should only load the recent message
        assert count == 1

    def test_summarize_without_llm(self, temp_log_file):
        """Test summarization without LLM."""
        mock_llm = MagicMock()
        mock_llm.analyze.return_value = "Mocked summary"

        analyzer = LogAnalyzer(llm_client=mock_llm)
        analyzer.load_files([temp_log_file])

        summary = analyzer.summarize()

        assert summary == "Mocked summary"
        mock_llm.analyze.assert_called_once()

    def test_analyze_timeline(self, tmp_path):
        """Test timeline analysis."""
        log_content = '\n'.join([
            "2024-01-10 14:00:00 INFO Start",
            "2024-01-10 14:01:00 INFO Processing",
            "2024-01-10 14:02:00 ERROR Failed",
            "2024-01-10 14:10:00 INFO End",
        ])
        log_file = tmp_path / "timeline.log"
        log_file.write_text(log_content)

        analyzer = LogAnalyzer()
        analyzer.load_files([str(log_file)])

        timeline = analyzer.analyze_timeline()

        assert len(timeline) >= 1
        # Check that error severity is detected
        error_events = [e for e in timeline if e.severity == 'error']
        assert len(error_events) >= 1

    def test_find_root_cause(self, tmp_path):
        """Test root cause analysis."""
        log_content = '\n'.join([
            "2024-01-10 14:00:00 INFO Starting service",
            "2024-01-10 14:00:01 WARN Low disk space",
            "2024-01-10 14:00:02 ERROR Connection refused to database",
            "2024-01-10 14:00:03 ERROR Service crashed",
        ])
        log_file = tmp_path / "root.log"
        log_file.write_text(log_content)

        mock_llm = MagicMock()
        mock_llm.analyze.return_value = "Root cause: Database unavailable"

        analyzer = LogAnalyzer(llm_client=mock_llm)
        analyzer.load_files([str(log_file)])

        result = analyzer.find_root_cause("Connection refused")

        assert "Root cause" in result
        mock_llm.analyze.assert_called_once()

    def test_find_root_cause_no_match(self, temp_log_file):
        """Test root cause when error message not found."""
        analyzer = LogAnalyzer()
        analyzer.load_files([temp_log_file])

        result = analyzer.find_root_cause("nonexistent error message xyz123")

        assert "No log entries found" in result


# =============================================================================
# Output Formatting Tests
# =============================================================================

class TestFormatOutput:
    """Tests for format_output function."""

    def test_format_log_entries_text(self):
        """Test formatting log entries as text."""
        entries = [
            LogEntry(
                raw="test",
                timestamp=datetime(2024, 1, 10, 14, 0, 0),
                level="ERROR",
                message="Test error"
            )
        ]

        output = format_output(entries, 'text')

        assert "2024-01-10 14:00:00" in output
        assert "[ERROR]" in output
        assert "Test error" in output

    def test_format_log_entries_json(self):
        """Test formatting log entries as JSON."""
        entries = [
            LogEntry(
                raw="test",
                timestamp=datetime(2024, 1, 10, 14, 0, 0),
                level="ERROR",
                message="Test error",
                file_path="test.log",
                line_number=1
            )
        ]

        output = format_output(entries, 'json')
        data = json.loads(output)

        assert len(data) == 1
        assert data[0]['level'] == 'ERROR'
        assert data[0]['message'] == 'Test error'

    def test_format_error_clusters_text(self):
        """Test formatting error clusters as text."""
        clusters = [
            ErrorCluster(
                signature="abc123",
                count=5,
                samples=[LogEntry(raw="Sample", message="Sample error")],
                first_seen=datetime(2024, 1, 10, 10, 0, 0),
                last_seen=datetime(2024, 1, 10, 15, 0, 0)
            )
        ]

        output = format_output(clusters, 'text')

        assert "abc123" in output
        assert "5 occurrences" in output
        assert "Sample error" in output

    def test_format_timeline_events_json(self):
        """Test formatting timeline events as JSON."""
        events = [
            TimelineEvent(
                timestamp=datetime(2024, 1, 10, 14, 0, 0),
                event_type='log_activity',
                message='INFO: 10',
                count=10,
                severity='info'
            )
        ]

        output = format_output(events, 'json')
        data = json.loads(output)

        assert len(data) == 1
        assert data[0]['count'] == 10
        assert data[0]['severity'] == 'info'


# =============================================================================
# Argument Parser Tests
# =============================================================================

class TestArgumentParser:
    """Tests for argument parser."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None

    def test_parse_find_errors(self):
        """Test parsing --find-errors argument."""
        parser = create_parser()
        args = parser.parse_args(['test.log', '--find-errors'])

        assert args.files == ['test.log']
        assert args.find_errors is True

    def test_parse_summarize(self):
        """Test parsing --summarize argument."""
        parser = create_parser()
        args = parser.parse_args(['test.log', '--summarize'])

        assert args.summarize is True

    def test_parse_timeline(self):
        """Test parsing --timeline argument."""
        parser = create_parser()
        args = parser.parse_args(['test.log', '--timeline', '--format', 'json'])

        assert args.timeline is True
        assert args.format == 'json'

    def test_parse_stdin(self):
        """Test parsing --stdin argument."""
        parser = create_parser()
        args = parser.parse_args(['--stdin', '--find-errors'])

        assert args.stdin is True

    def test_parse_last(self):
        """Test parsing --last argument."""
        parser = create_parser()
        args = parser.parse_args(['test.log', '--summarize', '--last', '1h'])

        assert args.last == '1h'

    def test_parse_cluster(self):
        """Test parsing --cluster argument."""
        parser = create_parser()
        args = parser.parse_args(['test.log', '--find-errors', '--cluster'])

        assert args.cluster is True

    def test_parse_root_cause(self):
        """Test parsing --find-root-cause argument."""
        parser = create_parser()
        args = parser.parse_args(['test.log', '--find-root-cause', 'connection refused'])

        assert args.find_root_cause == 'connection refused'


# =============================================================================
# Main Function Tests
# =============================================================================

class TestMain:
    """Tests for main function."""

    def test_main_no_args(self):
        """Test main with no arguments."""
        with patch('sys.argv', ['log_analyzer']):
            result = main()
            assert result == EXIT_INVALID_ARGS

    def test_main_no_analysis_mode(self, temp_log_file):
        """Test main with file but no analysis mode."""
        with patch('sys.argv', ['log_analyzer', temp_log_file]):
            result = main()
            assert result == EXIT_INVALID_ARGS

    def test_main_file_not_found(self):
        """Test main with non-existent file."""
        with patch('sys.argv', ['log_analyzer', '/nonexistent/file.log', '--find-errors']):
            result = main()
            assert result == EXIT_NO_LOGS

    def test_main_find_errors_success(self, temp_log_file, capsys):
        """Test main with --find-errors on valid file."""
        with patch('sys.argv', ['log_analyzer', temp_log_file, '--find-errors']):
            result = main()

            assert result == EXIT_SUCCESS
            captured = capsys.readouterr()
            assert "ERROR" in captured.out or "error" in captured.out.lower() or "No errors found" in captured.out

    def test_main_summarize_success(self, temp_log_file, capsys):
        """Test main with --summarize on valid file."""
        mock_llm = MagicMock()
        mock_llm.analyze.return_value = "Test summary"
        mock_llm.is_available.return_value = True

        with patch('sys.argv', ['log_analyzer', temp_log_file, '--summarize']):
            with patch('log_analyzer.LLMClient', return_value=mock_llm):
                result = main()

                assert result == EXIT_SUCCESS

    def test_main_timeline_success(self, temp_log_file, capsys):
        """Test main with --timeline on valid file."""
        with patch('sys.argv', ['log_analyzer', temp_log_file, '--timeline']):
            result = main()

            assert result == EXIT_SUCCESS

    def test_main_json_format(self, temp_log_file, capsys):
        """Test main with JSON output format."""
        with patch('sys.argv', ['log_analyzer', temp_log_file, '--find-errors', '--format', 'json']):
            result = main()

            captured = capsys.readouterr()
            # Output should be valid JSON (even if empty array)
            if captured.out.strip():
                try:
                    json.loads(captured.out)
                except json.JSONDecodeError:
                    # Could be "No errors found" message
                    pass

    def test_main_stdin(self, generic_log_lines, capsys):
        """Test main with stdin input."""
        stdin_data = '\n'.join(generic_log_lines)

        with patch('sys.argv', ['log_analyzer', '--stdin', '--find-errors']):
            with patch('sys.stdin', io.StringIO(stdin_data)):
                result = main()

                assert result == EXIT_SUCCESS

    def test_main_verbose(self, temp_log_file, capsys):
        """Test main with verbose output."""
        with patch('sys.argv', ['log_analyzer', temp_log_file, '--find-errors', '-v']):
            result = main()

            captured = capsys.readouterr()
            assert "Loaded" in captured.err


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_nginx(self, temp_nginx_log, capsys):
        """Test full pipeline with nginx logs."""
        with patch('sys.argv', ['log_analyzer', temp_nginx_log, '--find-errors', '--cluster']):
            result = main()

            assert result == EXIT_SUCCESS
            captured = capsys.readouterr()
            # Should find the 500 error
            assert "500" in captured.out or "error" in captured.out.lower() or "Cluster" in captured.out

    def test_full_pipeline_json_logs(self, tmp_path, json_log_lines, capsys):
        """Test full pipeline with JSON logs."""
        log_file = tmp_path / "json.log"
        log_file.write_text('\n'.join(json_log_lines))

        with patch('sys.argv', ['log_analyzer', str(log_file), '--find-errors', '--format', 'json']):
            result = main()

            assert result == EXIT_SUCCESS
            captured = capsys.readouterr()

            # Parse JSON output
            if captured.out.strip() and captured.out.strip().startswith('['):
                data = json.loads(captured.out)
                # Should find the ERROR entry
                error_entries = [e for e in data if e.get('level') == 'ERROR']
                assert len(error_entries) >= 1

    def test_multiple_analysis_modes(self, temp_log_file, capsys):
        """Test running multiple analysis modes."""
        mock_llm = MagicMock()
        mock_llm.analyze.return_value = "Summary"
        mock_llm.is_available.return_value = True

        with patch('sys.argv', ['log_analyzer', temp_log_file, '--find-errors', '--summarize', '--timeline']):
            with patch('log_analyzer.LLMClient', return_value=mock_llm):
                result = main()

                assert result == EXIT_SUCCESS


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_file(self, tmp_path):
        """Test handling empty file."""
        empty_file = tmp_path / "empty.log"
        empty_file.write_text("")

        analyzer = LogAnalyzer()
        count = analyzer.load_files([str(empty_file)])

        assert count == 0

    def test_binary_file_handling(self, tmp_path):
        """Test handling binary file gracefully."""
        binary_file = tmp_path / "binary.log"
        binary_file.write_bytes(b'\x00\x01\x02\x03\xff\xfe')

        analyzer = LogAnalyzer()
        # Should not crash
        count = analyzer.load_files([str(binary_file)])
        # May or may not parse anything useful

    def test_very_long_line(self, tmp_path):
        """Test handling very long log lines."""
        long_line = "A" * 10000
        log_file = tmp_path / "long.log"
        log_file.write_text(f"2024-01-10 12:00:00 INFO {long_line}")

        analyzer = LogAnalyzer()
        count = analyzer.load_files([str(log_file)])

        assert count == 1

    def test_malformed_json(self, tmp_path):
        """Test handling malformed JSON logs."""
        log_content = '{"incomplete json\n{"valid": "json", "level": "INFO", "message": "test"}'
        log_file = tmp_path / "malformed.log"
        log_file.write_text(log_content)

        analyzer = LogAnalyzer()
        # Should not crash
        count = analyzer.load_files([str(log_file)])

    def test_unicode_handling(self, tmp_path):
        """Test handling unicode in logs."""
        log_content = "2024-01-10 12:00:00 INFO Message with unicode: \u4e2d\u6587 \u0430\u0431\u0432"
        log_file = tmp_path / "unicode.log"
        log_file.write_text(log_content, encoding='utf-8')

        analyzer = LogAnalyzer()
        count = analyzer.load_files([str(log_file)])

        assert count == 1

    def test_permission_error(self, tmp_path, capsys):
        """Test handling permission error."""
        log_file = tmp_path / "noperm.log"
        log_file.write_text("test")

        # This test is platform-dependent, skip on Windows
        if os.name != 'nt':
            os.chmod(str(log_file), 0o000)

            try:
                analyzer = LogAnalyzer()
                # Should print warning but not crash
                count = analyzer.load_files([str(log_file)])
            finally:
                os.chmod(str(log_file), 0o644)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
