#!/usr/bin/env python3
"""
log_analyzer - A self-documenting CLI tool for parsing and analyzing log files using LLMs.

This tool auto-detects common log formats (nginx, apache, syslog, JSON) and provides
various analysis capabilities including error finding, clustering, summarization,
and timeline analysis.

Environment Variables:
    OPENAI_API_KEY     - OpenAI API key (preferred)
    ANTHROPIC_API_KEY  - Anthropic API key (fallback)
    GOOGLE_API_KEY     - Google API key (fallback)

Exit Codes:
    0 - Success
    1 - No logs found
    2 - API error
    3 - Invalid arguments

Examples:
    # Find and display errors in a log file
    log_analyzer /var/log/app.log --find-errors

    # Summarize patterns in a log file
    log_analyzer app.log --summarize

    # Analyze nginx logs from the last hour
    log_analyzer nginx.log --summarize --last 1h

    # Find errors across multiple files and cluster them
    log_analyzer *.log --find-errors --cluster

    # Generate a timeline in JSON format
    log_analyzer app.log --timeline --format json

    # Find root cause from stdin
    log_analyzer --stdin < combined.log --find-root-cause "connection refused"

Author: Auto-generated CLI Tool
Version: 1.0.0
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import hashlib

# Exit codes
EXIT_SUCCESS = 0
EXIT_NO_LOGS = 1
EXIT_API_ERROR = 2
EXIT_INVALID_ARGS = 3

# Version
__version__ = "1.0.0"


class LogFormat(Enum):
    """Supported log formats."""
    NGINX = "nginx"
    APACHE = "apache"
    SYSLOG = "syslog"
    JSON = "json"
    UNKNOWN = "unknown"


@dataclass
class LogEntry:
    """Represents a parsed log entry."""
    raw: str
    timestamp: Optional[datetime] = None
    level: Optional[str] = None
    message: str = ""
    source: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    line_number: int = 0
    file_path: str = ""


@dataclass
class ErrorCluster:
    """Represents a cluster of similar errors."""
    signature: str
    count: int
    samples: List[LogEntry]
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


@dataclass
class TimelineEvent:
    """Represents an event in the timeline."""
    timestamp: datetime
    event_type: str
    message: str
    count: int = 1
    severity: str = "info"


class LogFormatDetector:
    """Detects the format of log files."""

    # Regex patterns for different log formats
    PATTERNS = {
        LogFormat.NGINX: re.compile(
            r'^(?P<ip>[\d.]+)\s+-\s+(?P<user>\S+)\s+\[(?P<time>[^\]]+)\]\s+'
            r'"(?P<request>[^"]+)"\s+(?P<status>\d+)\s+(?P<size>\d+)'
        ),
        LogFormat.APACHE: re.compile(
            r'^(?P<ip>[\d.]+)\s+\S+\s+(?P<user>\S+)\s+\[(?P<time>[^\]]+)\]\s+'
            r'"(?P<request>[^"]+)"\s+(?P<status>\d+)\s+(?P<size>\d+)'
        ),
        LogFormat.SYSLOG: re.compile(
            r'^(?P<time>\w+\s+\d+\s+[\d:]+)\s+(?P<host>\S+)\s+(?P<process>[^:]+):\s*(?P<message>.*)$'
        ),
    }

    @classmethod
    def detect(cls, lines: List[str]) -> LogFormat:
        """Detect the log format from sample lines."""
        if not lines:
            return LogFormat.UNKNOWN

        # Check for JSON format first
        json_count = 0
        for line in lines[:10]:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    json.loads(line)
                    json_count += 1
                except json.JSONDecodeError:
                    pass

        if json_count >= len(lines[:10]) * 0.5:
            return LogFormat.JSON

        # Check other formats
        format_scores = Counter()
        for line in lines[:20]:
            for fmt, pattern in cls.PATTERNS.items():
                if pattern.match(line):
                    format_scores[fmt] += 1

        if format_scores:
            return format_scores.most_common(1)[0][0]

        return LogFormat.UNKNOWN


class LogParser:
    """Parses log files into structured entries."""

    # Common timestamp patterns
    TIMESTAMP_PATTERNS = [
        (re.compile(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}'), '%Y-%m-%dT%H:%M:%S'),
        (re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'), '%Y-%m-%d %H:%M:%S'),
        (re.compile(r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}'), '%d/%b/%Y:%H:%M:%S'),
        (re.compile(r'\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2}'), '%b %d %H:%M:%S'),
    ]

    # Log level patterns
    LEVEL_PATTERN = re.compile(
        r'\b(DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL|SEVERE|TRACE)\b',
        re.IGNORECASE
    )

    # Error indicators
    ERROR_INDICATORS = [
        'error', 'exception', 'failed', 'failure', 'fatal', 'critical',
        'traceback', 'panic', 'crash', 'denied', 'refused', 'timeout',
        'abort', 'segfault', 'core dump', 'stack trace'
    ]

    def __init__(self, log_format: LogFormat = LogFormat.UNKNOWN):
        self.log_format = log_format

    def parse_line(self, line: str, line_number: int = 0, file_path: str = "") -> LogEntry:
        """Parse a single log line."""
        entry = LogEntry(
            raw=line.strip(),
            line_number=line_number,
            file_path=file_path,
            message=line.strip()
        )

        if self.log_format == LogFormat.JSON:
            return self._parse_json(entry, line)
        elif self.log_format == LogFormat.NGINX:
            return self._parse_nginx(entry, line)
        elif self.log_format == LogFormat.APACHE:
            return self._parse_apache(entry, line)
        elif self.log_format == LogFormat.SYSLOG:
            return self._parse_syslog(entry, line)
        else:
            return self._parse_generic(entry, line)

    def _parse_json(self, entry: LogEntry, line: str) -> LogEntry:
        """Parse JSON formatted log."""
        try:
            data = json.loads(line.strip())
            entry.extra = data

            # Extract common fields
            for ts_field in ['timestamp', 'time', '@timestamp', 'ts', 'datetime']:
                if ts_field in data:
                    entry.timestamp = self._parse_timestamp(str(data[ts_field]))
                    break

            for level_field in ['level', 'severity', 'log_level', 'loglevel']:
                if level_field in data:
                    entry.level = str(data[level_field]).upper()
                    break

            for msg_field in ['message', 'msg', 'text', 'log']:
                if msg_field in data:
                    entry.message = str(data[msg_field])
                    break

        except json.JSONDecodeError:
            pass

        return entry

    def _parse_nginx(self, entry: LogEntry, line: str) -> LogEntry:
        """Parse nginx access log format."""
        match = LogFormatDetector.PATTERNS[LogFormat.NGINX].match(line)
        if match:
            groups = match.groupdict()
            entry.extra = groups
            entry.timestamp = self._parse_timestamp(groups.get('time', ''))
            status = int(groups.get('status', 0))
            if status >= 400:
                entry.level = 'ERROR' if status >= 500 else 'WARN'
            else:
                entry.level = 'INFO'
            entry.message = f"{groups.get('request', '')} - {status}"
        return entry

    def _parse_apache(self, entry: LogEntry, line: str) -> LogEntry:
        """Parse Apache access log format."""
        match = LogFormatDetector.PATTERNS[LogFormat.APACHE].match(line)
        if match:
            groups = match.groupdict()
            entry.extra = groups
            entry.timestamp = self._parse_timestamp(groups.get('time', ''))
            status = int(groups.get('status', 0))
            if status >= 400:
                entry.level = 'ERROR' if status >= 500 else 'WARN'
            else:
                entry.level = 'INFO'
            entry.message = f"{groups.get('request', '')} - {status}"
        return entry

    def _parse_syslog(self, entry: LogEntry, line: str) -> LogEntry:
        """Parse syslog format."""
        match = LogFormatDetector.PATTERNS[LogFormat.SYSLOG].match(line)
        if match:
            groups = match.groupdict()
            entry.extra = groups
            entry.timestamp = self._parse_timestamp(groups.get('time', ''))
            entry.source = groups.get('process', '')
            entry.message = groups.get('message', '')

            # Try to extract level from message
            level_match = self.LEVEL_PATTERN.search(entry.message)
            if level_match:
                entry.level = level_match.group(1).upper()
        return entry

    def _parse_generic(self, entry: LogEntry, line: str) -> LogEntry:
        """Parse generic log format."""
        # Try to extract timestamp
        for pattern, fmt in self.TIMESTAMP_PATTERNS:
            match = pattern.search(line)
            if match:
                entry.timestamp = self._parse_timestamp(match.group())
                break

        # Try to extract level
        level_match = self.LEVEL_PATTERN.search(line)
        if level_match:
            entry.level = level_match.group(1).upper()
            if entry.level == 'WARNING':
                entry.level = 'WARN'

        return entry

    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        """Parse a timestamp string into a datetime object."""
        if not ts_str:
            return None

        # Try ISO format first
        try:
            # Handle various ISO formats
            ts_str = ts_str.replace('T', ' ').split('+')[0].split('.')[0]
            return datetime.fromisoformat(ts_str)
        except ValueError:
            pass

        # Try other formats
        formats = [
            '%d/%b/%Y:%H:%M:%S',
            '%b %d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(ts_str.strip(), fmt)
                # Handle syslog format without year
                if dt.year == 1900:
                    dt = dt.replace(year=datetime.now().year)
                return dt
            except ValueError:
                continue

        return None

    def is_error(self, entry: LogEntry) -> bool:
        """Check if a log entry represents an error."""
        if entry.level and entry.level in ('ERROR', 'CRITICAL', 'FATAL', 'SEVERE'):
            return True

        message_lower = entry.message.lower()
        return any(indicator in message_lower for indicator in self.ERROR_INDICATORS)


class ErrorClusterer:
    """Clusters similar errors together."""

    def __init__(self):
        self.clusters: Dict[str, ErrorCluster] = {}

    def _compute_signature(self, entry: LogEntry) -> str:
        """Compute a signature for clustering similar errors."""
        # Normalize the message by removing variable parts
        msg = entry.message

        # Remove timestamps
        msg = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>', msg)
        msg = re.sub(r'\d{2}:\d{2}:\d{2}', '<TIME>', msg)

        # Remove IP addresses
        msg = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>', msg)

        # Remove numbers
        msg = re.sub(r'\b\d+\b', '<NUM>', msg)

        # Remove UUIDs
        msg = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '<UUID>', msg, flags=re.IGNORECASE)

        # Remove hex strings
        msg = re.sub(r'0x[a-f0-9]+', '<HEX>', msg, flags=re.IGNORECASE)

        # Remove file paths
        msg = re.sub(r'/[\w/.-]+', '<PATH>', msg)

        # Create hash
        return hashlib.md5(msg.encode()).hexdigest()[:12]

    def add_error(self, entry: LogEntry) -> str:
        """Add an error to the appropriate cluster."""
        sig = self._compute_signature(entry)

        if sig not in self.clusters:
            self.clusters[sig] = ErrorCluster(
                signature=sig,
                count=0,
                samples=[],
                first_seen=entry.timestamp,
                last_seen=entry.timestamp
            )

        cluster = self.clusters[sig]
        cluster.count += 1

        if len(cluster.samples) < 3:
            cluster.samples.append(entry)

        if entry.timestamp:
            if cluster.first_seen is None or entry.timestamp < cluster.first_seen:
                cluster.first_seen = entry.timestamp
            if cluster.last_seen is None or entry.timestamp > cluster.last_seen:
                cluster.last_seen = entry.timestamp

        return sig

    def get_clusters(self) -> List[ErrorCluster]:
        """Get all clusters sorted by count (descending)."""
        return sorted(self.clusters.values(), key=lambda c: c.count, reverse=True)


class TimelineAnalyzer:
    """Analyzes logs and generates a timeline of events."""

    def __init__(self, bucket_minutes: int = 5):
        self.bucket_minutes = bucket_minutes
        self.buckets: Dict[datetime, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _bucket_time(self, ts: datetime) -> datetime:
        """Round timestamp to bucket boundary."""
        return ts.replace(
            minute=(ts.minute // self.bucket_minutes) * self.bucket_minutes,
            second=0,
            microsecond=0
        )

    def add_entry(self, entry: LogEntry):
        """Add a log entry to the timeline."""
        if entry.timestamp:
            bucket = self._bucket_time(entry.timestamp)
            level = entry.level or 'INFO'
            self.buckets[bucket][level] += 1

    def get_timeline(self) -> List[TimelineEvent]:
        """Generate timeline events."""
        events = []

        for ts in sorted(self.buckets.keys()):
            level_counts = self.buckets[ts]

            # Determine severity based on errors
            severity = 'info'
            if level_counts.get('ERROR', 0) > 0 or level_counts.get('CRITICAL', 0) > 0:
                severity = 'error'
            elif level_counts.get('WARN', 0) > 0:
                severity = 'warning'

            total = sum(level_counts.values())
            details = ', '.join(f"{k}: {v}" for k, v in sorted(level_counts.items()))

            events.append(TimelineEvent(
                timestamp=ts,
                event_type='log_activity',
                message=details,
                count=total,
                severity=severity
            ))

        return events


class LLMClient:
    """Client for interacting with LLM APIs."""

    def __init__(self):
        self.api_key = None
        self.provider = None
        self._detect_provider()

    def _detect_provider(self):
        """Detect which API key is available."""
        if os.environ.get('OPENAI_API_KEY'):
            self.api_key = os.environ['OPENAI_API_KEY']
            self.provider = 'openai'
        elif os.environ.get('ANTHROPIC_API_KEY'):
            self.api_key = os.environ['ANTHROPIC_API_KEY']
            self.provider = 'anthropic'
        elif os.environ.get('GOOGLE_API_KEY'):
            self.api_key = os.environ['GOOGLE_API_KEY']
            self.provider = 'google'

    def is_available(self) -> bool:
        """Check if an LLM API is available."""
        return self.api_key is not None

    def analyze(self, prompt: str, context: str) -> str:
        """Send a prompt to the LLM for analysis."""
        if not self.is_available():
            return self._fallback_analysis(prompt, context)

        try:
            if self.provider == 'openai':
                return self._call_openai(prompt, context)
            elif self.provider == 'anthropic':
                return self._call_anthropic(prompt, context)
            elif self.provider == 'google':
                return self._call_google(prompt, context)
        except Exception as e:
            raise LLMError(f"API error: {e}")

        return self._fallback_analysis(prompt, context)

    def _call_openai(self, prompt: str, context: str) -> str:
        """Call OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a log analysis expert. Analyze logs and provide clear, actionable insights."},
                    {"role": "user", "content": f"{prompt}\n\nLog data:\n{context}"}
                ],
                max_tokens=2000
            )
            return response.choices[0].message.content
        except ImportError:
            raise LLMError("openai package not installed. Run: pip install openai")

    def _call_anthropic(self, prompt: str, context: str) -> str:
        """Call Anthropic API."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": f"{prompt}\n\nLog data:\n{context}"}
                ]
            )
            return message.content[0].text
        except ImportError:
            raise LLMError("anthropic package not installed. Run: pip install anthropic")

    def _call_google(self, prompt: str, context: str) -> str:
        """Call Google API."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(f"{prompt}\n\nLog data:\n{context}")
            return response.text
        except ImportError:
            raise LLMError("google-generativeai package not installed. Run: pip install google-generativeai")

    def _fallback_analysis(self, prompt: str, context: str) -> str:
        """Provide basic analysis without LLM."""
        lines = context.strip().split('\n')
        return f"[Fallback analysis - no LLM API key found]\nAnalyzed {len(lines)} log lines.\nSet OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY for AI-powered analysis."


class LLMError(Exception):
    """Exception for LLM-related errors."""
    pass


class LogAnalyzer:
    """Main log analyzer class."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.parser: Optional[LogParser] = None
        self.entries: List[LogEntry] = []
        self.log_format: LogFormat = LogFormat.UNKNOWN

    def load_files(self, patterns: List[str], last: Optional[str] = None) -> int:
        """Load log files matching the given patterns."""
        files = []
        for pattern in patterns:
            matches = glob.glob(pattern)
            files.extend(matches)

        if not files:
            return 0

        # Determine time filter
        time_filter = self._parse_time_filter(last) if last else None

        all_lines = []
        file_lines: Dict[str, List[Tuple[int, str]]] = {}

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    file_lines[file_path] = [(i + 1, line) for i, line in enumerate(lines)]
                    all_lines.extend(lines)
            except (IOError, OSError) as e:
                print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)

        if not all_lines:
            return 0

        # Detect format from sample lines
        self.log_format = LogFormatDetector.detect(all_lines[:50])
        self.parser = LogParser(self.log_format)

        # Parse all entries
        for file_path, lines in file_lines.items():
            for line_num, line in lines:
                if line.strip():
                    entry = self.parser.parse_line(line, line_num, file_path)

                    # Apply time filter
                    if time_filter and entry.timestamp:
                        if entry.timestamp < time_filter:
                            continue

                    self.entries.append(entry)

        return len(self.entries)

    def load_stdin(self, last: Optional[str] = None) -> int:
        """Load logs from stdin."""
        lines = sys.stdin.readlines()

        if not lines:
            return 0

        time_filter = self._parse_time_filter(last) if last else None

        self.log_format = LogFormatDetector.detect(lines[:50])
        self.parser = LogParser(self.log_format)

        for i, line in enumerate(lines):
            if line.strip():
                entry = self.parser.parse_line(line, i + 1, '<stdin>')

                if time_filter and entry.timestamp:
                    if entry.timestamp < time_filter:
                        continue

                self.entries.append(entry)

        return len(self.entries)

    def _parse_time_filter(self, spec: str) -> Optional[datetime]:
        """Parse a time specification like '1h', '30m', '2d'."""
        match = re.match(r'^(\d+)([smhd])$', spec.lower())
        if not match:
            return None

        value, unit = int(match.group(1)), match.group(2)

        if unit == 's':
            delta = timedelta(seconds=value)
        elif unit == 'm':
            delta = timedelta(minutes=value)
        elif unit == 'h':
            delta = timedelta(hours=value)
        elif unit == 'd':
            delta = timedelta(days=value)
        else:
            return None

        return datetime.now() - delta

    def find_errors(self, cluster: bool = False) -> Tuple[List[LogEntry], Optional[List[ErrorCluster]]]:
        """Find all errors in the loaded logs."""
        errors = [e for e in self.entries if self.parser and self.parser.is_error(e)]

        if not cluster:
            return errors, None

        clusterer = ErrorClusterer()
        for error in errors:
            clusterer.add_error(error)

        return errors, clusterer.get_clusters()

    def summarize(self) -> str:
        """Generate a summary of the logs using LLM."""
        if not self.entries:
            return "No log entries to summarize."

        # Prepare context
        sample_size = min(100, len(self.entries))
        samples = self.entries[:sample_size]
        context = '\n'.join(e.raw for e in samples)

        # Statistics
        total = len(self.entries)
        levels = Counter(e.level for e in self.entries if e.level)

        stats = f"Total entries: {total}\nFormat detected: {self.log_format.value}\n"
        stats += f"Level distribution: {dict(levels)}\n"

        prompt = f"""Analyze these logs and provide a concise summary including:
1. Main activity patterns
2. Notable events or anomalies
3. Key statistics
4. Recommendations if any issues found

Statistics:
{stats}

Please be concise and focus on actionable insights."""

        try:
            return self.llm.analyze(prompt, context)
        except LLMError as e:
            return f"Error generating summary: {e}\n\n{stats}"

    def analyze_timeline(self) -> List[TimelineEvent]:
        """Generate a timeline analysis of the logs."""
        analyzer = TimelineAnalyzer()

        for entry in self.entries:
            analyzer.add_entry(entry)

        return analyzer.get_timeline()

    def find_root_cause(self, error_message: str) -> str:
        """Use LLM to find the root cause of a specific error."""
        # Find relevant entries around errors matching the message
        relevant = []

        for i, entry in enumerate(self.entries):
            if error_message.lower() in entry.message.lower():
                # Include context (entries before and after)
                start = max(0, i - 5)
                end = min(len(self.entries), i + 3)
                relevant.extend(self.entries[start:end])

        if not relevant:
            return f"No log entries found matching: {error_message}"

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for entry in relevant:
            key = (entry.file_path, entry.line_number)
            if key not in seen:
                seen.add(key)
                unique.append(entry)

        context = '\n'.join(e.raw for e in unique[:100])

        prompt = f"""Analyze these log entries to find the root cause of the error: "{error_message}"

Please provide:
1. The likely root cause
2. The sequence of events leading to the error
3. Recommended fixes or next steps

Focus on actionable insights."""

        try:
            return self.llm.analyze(prompt, context)
        except LLMError as e:
            return f"Error analyzing root cause: {e}"


def format_output(data: Any, fmt: str = 'text') -> str:
    """Format output in the specified format."""
    if fmt == 'json':
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, LogEntry):
                    result.append({
                        'timestamp': item.timestamp.isoformat() if item.timestamp else None,
                        'level': item.level,
                        'message': item.message,
                        'file': item.file_path,
                        'line': item.line_number
                    })
                elif isinstance(item, ErrorCluster):
                    result.append({
                        'signature': item.signature,
                        'count': item.count,
                        'first_seen': item.first_seen.isoformat() if item.first_seen else None,
                        'last_seen': item.last_seen.isoformat() if item.last_seen else None,
                        'samples': [s.message for s in item.samples]
                    })
                elif isinstance(item, TimelineEvent):
                    result.append({
                        'timestamp': item.timestamp.isoformat(),
                        'type': item.event_type,
                        'message': item.message,
                        'count': item.count,
                        'severity': item.severity
                    })
                else:
                    result.append(str(item))
            return json.dumps(result, indent=2)
        return json.dumps(data, indent=2, default=str)

    # Text format
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, LogEntry):
                ts = item.timestamp.strftime('%Y-%m-%d %H:%M:%S') if item.timestamp else 'unknown'
                level = f"[{item.level}]" if item.level else ""
                lines.append(f"{ts} {level} {item.message}")
            elif isinstance(item, ErrorCluster):
                lines.append(f"\n--- Cluster {item.signature} ({item.count} occurrences) ---")
                if item.first_seen and item.last_seen:
                    lines.append(f"  Time range: {item.first_seen} to {item.last_seen}")
                lines.append("  Sample messages:")
                for sample in item.samples:
                    lines.append(f"    - {sample.message[:100]}...")
            elif isinstance(item, TimelineEvent):
                severity_marker = {'error': '[!]', 'warning': '[~]', 'info': '[ ]'}
                marker = severity_marker.get(item.severity, '[ ]')
                ts = item.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                lines.append(f"{ts} {marker} {item.message} (count: {item.count})")
            else:
                lines.append(str(item))
        return '\n'.join(lines)

    return str(data)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with comprehensive help."""
    parser = argparse.ArgumentParser(
        prog='log_analyzer',
        description="""
Log Analyzer - A self-documenting CLI tool for parsing and analyzing log files using LLMs.

This tool auto-detects common log formats (nginx, apache, syslog, JSON) and provides
various analysis capabilities including error finding, clustering, summarization,
and timeline analysis.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ENVIRONMENT VARIABLES:
  OPENAI_API_KEY      OpenAI API key (preferred)
  ANTHROPIC_API_KEY   Anthropic API key (fallback)
  GOOGLE_API_KEY      Google API key (fallback)

EXIT CODES:
  0  Success
  1  No logs found
  2  API error
  3  Invalid arguments

EXAMPLES:
  Find and display errors in a log file:
    %(prog)s /var/log/app.log --find-errors

  Summarize patterns in a log file:
    %(prog)s app.log --summarize

  Analyze nginx logs from the last hour:
    %(prog)s nginx.log --summarize --last 1h

  Find errors across multiple files and cluster them:
    %(prog)s *.log --find-errors --cluster

  Generate a timeline in JSON format:
    %(prog)s app.log --timeline --format json

  Find root cause from stdin:
    %(prog)s --stdin < combined.log --find-root-cause "connection refused"

  Analyze logs from the last 30 minutes:
    %(prog)s app.log --summarize --last 30m

  Analyze logs from the last 2 days:
    %(prog)s /var/log/syslog --find-errors --last 2d

SUPPORTED LOG FORMATS:
  - nginx access logs
  - Apache access logs
  - syslog format
  - JSON logs (structured logging)
  - Generic logs with timestamps

For more information, visit: https://github.com/example/log_analyzer
        """
    )

    # Positional arguments
    parser.add_argument(
        'files',
        nargs='*',
        metavar='FILE',
        help='Log file(s) to analyze. Supports glob patterns (e.g., *.log)'
    )

    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '--stdin',
        action='store_true',
        help='Read log data from standard input instead of files'
    )
    input_group.add_argument(
        '--last',
        metavar='TIME',
        help='Only analyze logs from the last TIME period (e.g., 1h, 30m, 2d)'
    )

    # Analysis modes
    analysis_group = parser.add_argument_group('Analysis Modes')
    analysis_group.add_argument(
        '--find-errors',
        action='store_true',
        help='Find and list all error entries in the logs'
    )
    analysis_group.add_argument(
        '--summarize',
        action='store_true',
        help='Generate an AI-powered summary of the log patterns'
    )
    analysis_group.add_argument(
        '--timeline',
        action='store_true',
        help='Generate a timeline of log activity'
    )
    analysis_group.add_argument(
        '--find-root-cause',
        metavar='ERROR',
        help='Analyze logs to find the root cause of a specific error message'
    )

    # Analysis options
    options_group = parser.add_argument_group('Analysis Options')
    options_group.add_argument(
        '--cluster',
        action='store_true',
        help='Cluster similar errors together (use with --find-errors)'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    output_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    output_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress non-essential output'
    )

    # Other options
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    if not args.files and not args.stdin:
        parser.print_help()
        print("\nError: No input specified. Provide files or use --stdin.", file=sys.stderr)
        return EXIT_INVALID_ARGS

    if not any([args.find_errors, args.summarize, args.timeline, args.find_root_cause]):
        parser.print_help()
        print("\nError: No analysis mode specified. Use --find-errors, --summarize, --timeline, or --find-root-cause.", file=sys.stderr)
        return EXIT_INVALID_ARGS

    # Create analyzer
    analyzer = LogAnalyzer()

    # Load logs
    try:
        if args.stdin:
            count = analyzer.load_stdin(args.last)
        else:
            count = analyzer.load_files(args.files, args.last)

        if count == 0:
            if not args.quiet:
                print("No log entries found.", file=sys.stderr)
            return EXIT_NO_LOGS

        if args.verbose:
            print(f"Loaded {count} log entries (format: {analyzer.log_format.value})", file=sys.stderr)

    except Exception as e:
        print(f"Error loading logs: {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS

    # Perform analysis
    try:
        if args.find_errors:
            errors, clusters = analyzer.find_errors(cluster=args.cluster)

            if not errors:
                if not args.quiet:
                    print("No errors found in the logs.")
            else:
                if args.cluster and clusters:
                    print(format_output(clusters, args.format))
                else:
                    print(format_output(errors, args.format))

                if args.verbose:
                    print(f"\nTotal errors: {len(errors)}", file=sys.stderr)

        if args.summarize:
            summary = analyzer.summarize()
            if args.format == 'json':
                print(json.dumps({'summary': summary}, indent=2))
            else:
                print(summary)

        if args.timeline:
            events = analyzer.analyze_timeline()
            if not events:
                if not args.quiet:
                    print("No timeline events (entries may lack timestamps).")
            else:
                print(format_output(events, args.format))

        if args.find_root_cause:
            analysis = analyzer.find_root_cause(args.find_root_cause)
            if args.format == 'json':
                print(json.dumps({'root_cause_analysis': analysis}, indent=2))
            else:
                print(analysis)

        return EXIT_SUCCESS

    except LLMError as e:
        print(f"API Error: {e}", file=sys.stderr)
        return EXIT_API_ERROR
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return EXIT_API_ERROR


if __name__ == '__main__':
    sys.exit(main())
