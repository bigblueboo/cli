#!/usr/bin/env python3
"""
Comprehensive tests for calendar_schedule CLI tool.

These tests use mocking to avoid requiring actual Google API credentials
while thoroughly testing all functionality.

Run with: pytest test_calendar_schedule.py -v
"""

import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import the module under test
import calendar_schedule as cs


class TestParseDatetime:
    """Tests for the parse_datetime function."""

    def test_parse_iso_format(self):
        """Test parsing ISO format datetime strings."""
        result = cs.parse_datetime("2024-01-15T14:30:00")
        assert result == datetime(2024, 1, 15, 14, 30, 0)

    def test_parse_iso_format_no_seconds(self):
        """Test parsing ISO format without seconds."""
        result = cs.parse_datetime("2024-01-15T14:30")
        assert result == datetime(2024, 1, 15, 14, 30, 0)

    def test_parse_date_time_space(self):
        """Test parsing date and time separated by space."""
        result = cs.parse_datetime("2024-01-15 14:30")
        assert result == datetime(2024, 1, 15, 14, 30, 0)

    def test_parse_date_only(self):
        """Test parsing date-only string."""
        result = cs.parse_datetime("2024-01-15")
        assert result == datetime(2024, 1, 15, 0, 0, 0)

    def test_parse_today(self):
        """Test parsing 'today' relative date."""
        reference = datetime(2024, 1, 15, 10, 30, 45)
        result = cs.parse_datetime("today", reference)
        assert result == datetime(2024, 1, 15, 0, 0, 0)

    def test_parse_tomorrow(self):
        """Test parsing 'tomorrow' relative date."""
        reference = datetime(2024, 1, 15, 10, 30, 45)
        result = cs.parse_datetime("tomorrow", reference)
        assert result == datetime(2024, 1, 16, 0, 0, 0)

    def test_parse_now(self):
        """Test parsing 'now' relative date."""
        reference = datetime(2024, 1, 15, 10, 30, 45)
        result = cs.parse_datetime("now", reference)
        assert result == reference

    def test_parse_plus_days(self):
        """Test parsing relative '+N days' format."""
        reference = datetime(2024, 1, 15, 10, 30, 0)
        result = cs.parse_datetime("+7 days", reference)
        assert result == datetime(2024, 1, 22, 10, 30, 0)

    def test_parse_plus_day_singular(self):
        """Test parsing relative '+1 day' format."""
        reference = datetime(2024, 1, 15, 10, 30, 0)
        result = cs.parse_datetime("+1 day", reference)
        assert result == datetime(2024, 1, 16, 10, 30, 0)

    def test_parse_plus_weeks(self):
        """Test parsing relative '+N weeks' format."""
        reference = datetime(2024, 1, 15, 10, 30, 0)
        result = cs.parse_datetime("+2 weeks", reference)
        assert result == datetime(2024, 1, 29, 10, 30, 0)

    def test_parse_plus_hours(self):
        """Test parsing relative '+N hours' format."""
        reference = datetime(2024, 1, 15, 10, 30, 0)
        result = cs.parse_datetime("+3 hours", reference)
        assert result == datetime(2024, 1, 15, 13, 30, 0)

    def test_parse_plus_minutes(self):
        """Test parsing relative '+N minutes' format."""
        reference = datetime(2024, 1, 15, 10, 30, 0)
        result = cs.parse_datetime("+45 minutes", reference)
        assert result == datetime(2024, 1, 15, 11, 15, 0)

    def test_parse_case_insensitive(self):
        """Test that parsing is case insensitive."""
        reference = datetime(2024, 1, 15, 10, 30, 0)
        assert cs.parse_datetime("TODAY", reference) == cs.parse_datetime("today", reference)
        assert cs.parse_datetime("Tomorrow", reference) == cs.parse_datetime("tomorrow", reference)

    def test_parse_invalid_raises_valueerror(self):
        """Test that invalid date strings raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            cs.parse_datetime("invalid-date")
        assert "Unable to parse date" in str(exc_info.value)

    def test_parse_us_format(self):
        """Test parsing US date format MM/DD/YYYY."""
        result = cs.parse_datetime("01/15/2024")
        assert result == datetime(2024, 1, 15, 0, 0, 0)


class TestFormatDatetime:
    """Tests for datetime formatting functions."""

    def test_format_datetime_rfc3339(self):
        """Test RFC3339 datetime formatting."""
        dt = datetime(2024, 1, 15, 14, 30, 45)
        result = cs.format_datetime_rfc3339(dt)
        assert result == "2024-01-15T14:30:45"

    def test_format_date(self):
        """Test date-only formatting for all-day events."""
        dt = datetime(2024, 1, 15, 14, 30, 45)
        result = cs.format_date(dt)
        assert result == "2024-01-15"


class TestGetCredentials:
    """Tests for credential loading."""

    def test_missing_env_var_exits(self):
        """Test that missing GOOGLE_APPLICATION_CREDENTIALS exits with auth error."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if it exists
            os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None)
            with pytest.raises(SystemExit) as exc_info:
                cs.get_credentials()
            assert exc_info.value.code == cs.EXIT_AUTH_ERROR

    def test_file_not_found_exits(self):
        """Test that non-existent credentials file exits with auth error."""
        with patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': '/nonexistent/path.json'}):
            with pytest.raises(SystemExit) as exc_info:
                cs.get_credentials()
            assert exc_info.value.code == cs.EXIT_AUTH_ERROR

    @patch('calendar_schedule.service_account.Credentials.from_service_account_file')
    @patch('os.path.exists')
    def test_valid_credentials_loaded(self, mock_exists, mock_from_file):
        """Test that valid credentials are loaded successfully."""
        mock_exists.return_value = True
        mock_creds = MagicMock()
        mock_from_file.return_value = mock_creds

        with patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': '/valid/path.json'}):
            result = cs.get_credentials()

        assert result == mock_creds
        mock_from_file.assert_called_once_with('/valid/path.json', scopes=cs.SCOPES)


class TestGetCalendarId:
    """Tests for calendar ID retrieval."""

    def test_default_calendar_id(self):
        """Test that default calendar ID is 'primary'."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('CALENDAR_ID', None)
            assert cs.get_calendar_id() == 'primary'

    def test_custom_calendar_id(self):
        """Test that custom CALENDAR_ID env var is used."""
        with patch.dict(os.environ, {'CALENDAR_ID': 'custom@calendar.google.com'}):
            assert cs.get_calendar_id() == 'custom@calendar.google.com'


class MockHttpError(Exception):
    """Mock HttpError for testing error handling."""

    def __init__(self, status, message="Error"):
        self.resp = MagicMock()
        self.resp.status = status
        self.message = message
        super().__init__(message)


@pytest.fixture
def mock_service():
    """Create a mock Google Calendar service."""
    with patch('calendar_schedule.get_calendar_service') as mock_get_service:
        service = MagicMock()
        mock_get_service.return_value = service
        yield service


@pytest.fixture
def mock_args():
    """Create a mock args namespace for testing commands."""
    return MagicMock()


class TestCmdCreate:
    """Tests for the create command."""

    def test_create_simple_event(self, mock_service, mock_args):
        """Test creating a simple timed event."""
        mock_args.summary = "Test Meeting"
        mock_args.start = "2024-01-15 14:00"
        mock_args.end = None
        mock_args.duration = 60
        mock_args.description = None
        mock_args.location = None
        mock_args.attendees = None
        mock_args.recurrence = None
        mock_args.all_day = False
        mock_args.timezone = "UTC"
        mock_args.notify = False
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().insert().execute.return_value = {
            'id': 'test123',
            'summary': 'Test Meeting',
            'htmlLink': 'https://calendar.google.com/event?id=test123'
        }

        result = cs.cmd_create(mock_args)

        assert result == cs.EXIT_SUCCESS
        mock_service.events().insert.assert_called()

    def test_create_event_with_attendees(self, mock_service, mock_args):
        """Test creating an event with attendees."""
        mock_args.summary = "Team Meeting"
        mock_args.start = "2024-01-15 14:00"
        mock_args.end = None
        mock_args.duration = 60
        mock_args.description = "Quarterly review"
        mock_args.location = "Conference Room A"
        mock_args.attendees = "alice@example.com,bob@example.com"
        mock_args.recurrence = None
        mock_args.all_day = False
        mock_args.timezone = "America/New_York"
        mock_args.notify = True
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().insert().execute.return_value = {
            'id': 'test456',
            'summary': 'Team Meeting',
            'htmlLink': 'https://calendar.google.com/event?id=test456'
        }

        result = cs.cmd_create(mock_args)

        assert result == cs.EXIT_SUCCESS

    def test_create_recurring_event(self, mock_service, mock_args):
        """Test creating a recurring event."""
        mock_args.summary = "Weekly Sync"
        mock_args.start = "2024-01-15 10:00"
        mock_args.end = None
        mock_args.duration = 45
        mock_args.description = None
        mock_args.location = None
        mock_args.attendees = None
        mock_args.recurrence = "FREQ=WEEKLY;BYDAY=MO;COUNT=10"
        mock_args.all_day = False
        mock_args.timezone = "UTC"
        mock_args.notify = False
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().insert().execute.return_value = {
            'id': 'recurring123',
            'summary': 'Weekly Sync',
            'htmlLink': 'https://calendar.google.com/event?id=recurring123'
        }

        result = cs.cmd_create(mock_args)

        assert result == cs.EXIT_SUCCESS

    def test_create_all_day_event(self, mock_service, mock_args):
        """Test creating an all-day event."""
        mock_args.summary = "Company Holiday"
        mock_args.start = "2024-01-15"
        mock_args.end = None
        mock_args.duration = None
        mock_args.description = None
        mock_args.location = None
        mock_args.attendees = None
        mock_args.recurrence = None
        mock_args.all_day = True
        mock_args.timezone = None
        mock_args.notify = False
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().insert().execute.return_value = {
            'id': 'allday123',
            'summary': 'Company Holiday',
            'htmlLink': 'https://calendar.google.com/event?id=allday123'
        }

        result = cs.cmd_create(mock_args)

        assert result == cs.EXIT_SUCCESS

    def test_create_invalid_start_time(self, mock_service, mock_args):
        """Test that invalid start time returns error code."""
        mock_args.summary = "Test"
        mock_args.start = "invalid-time"
        mock_args.calendar_id = None

        result = cs.cmd_create(mock_args)

        assert result == cs.EXIT_INVALID_ARGS

    @patch('calendar_schedule.HttpError', MockHttpError)
    def test_create_calendar_not_found(self, mock_service, mock_args):
        """Test handling of non-existent calendar."""
        mock_args.summary = "Test"
        mock_args.start = "2024-01-15 14:00"
        mock_args.end = None
        mock_args.duration = 60
        mock_args.description = None
        mock_args.location = None
        mock_args.attendees = None
        mock_args.recurrence = None
        mock_args.all_day = False
        mock_args.timezone = "UTC"
        mock_args.notify = False
        mock_args.calendar_id = "nonexistent@calendar.google.com"
        mock_args.json = False

        mock_service.events().insert().execute.side_effect = MockHttpError(404)

        with patch('calendar_schedule.HttpError', MockHttpError):
            result = cs.cmd_create(mock_args)

        assert result == cs.EXIT_NOT_FOUND


class TestCmdUpdate:
    """Tests for the update command."""

    def test_update_summary(self, mock_service, mock_args):
        """Test updating event summary."""
        mock_args.event_id = "event123"
        mock_args.summary = "New Title"
        mock_args.start = None
        mock_args.duration = None
        mock_args.description = None
        mock_args.location = None
        mock_args.attendees = None
        mock_args.recurrence = None
        mock_args.timezone = None
        mock_args.notify = False
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().get().execute.return_value = {
            'id': 'event123',
            'summary': 'Old Title',
            'start': {'dateTime': '2024-01-15T14:00:00', 'timeZone': 'UTC'},
            'end': {'dateTime': '2024-01-15T15:00:00', 'timeZone': 'UTC'}
        }
        mock_service.events().update().execute.return_value = {
            'id': 'event123',
            'summary': 'New Title'
        }

        result = cs.cmd_update(mock_args)

        assert result == cs.EXIT_SUCCESS

    def test_update_event_not_found(self, mock_service, mock_args):
        """Test updating non-existent event."""
        mock_args.event_id = "nonexistent"
        mock_args.summary = "New Title"
        mock_args.start = None
        mock_args.duration = None
        mock_args.description = None
        mock_args.location = None
        mock_args.attendees = None
        mock_args.recurrence = None
        mock_args.timezone = None
        mock_args.notify = False
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().get().execute.side_effect = MockHttpError(404)

        with patch('calendar_schedule.HttpError', MockHttpError):
            result = cs.cmd_update(mock_args)

        assert result == cs.EXIT_NOT_FOUND


class TestCmdDelete:
    """Tests for the delete command."""

    def test_delete_success(self, mock_service, mock_args):
        """Test successful event deletion."""
        mock_args.event_id = "event123"
        mock_args.notify = False
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().delete().execute.return_value = None

        result = cs.cmd_delete(mock_args)

        assert result == cs.EXIT_SUCCESS

    def test_delete_event_not_found(self, mock_service, mock_args):
        """Test deleting non-existent event."""
        mock_args.event_id = "nonexistent"
        mock_args.notify = False
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().delete().execute.side_effect = MockHttpError(404)

        with patch('calendar_schedule.HttpError', MockHttpError):
            result = cs.cmd_delete(mock_args)

        assert result == cs.EXIT_NOT_FOUND

    def test_delete_already_deleted(self, mock_service, mock_args):
        """Test deleting already deleted event (410 Gone)."""
        mock_args.event_id = "deleted"
        mock_args.notify = False
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().delete().execute.side_effect = MockHttpError(410)

        with patch('calendar_schedule.HttpError', MockHttpError):
            result = cs.cmd_delete(mock_args)

        assert result == cs.EXIT_NOT_FOUND


class TestCmdGet:
    """Tests for the get command."""

    def test_get_event_success(self, mock_service, mock_args):
        """Test getting event details."""
        mock_args.event_id = "event123"
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().get().execute.return_value = {
            'id': 'event123',
            'summary': 'Test Event',
            'status': 'confirmed',
            'start': {'dateTime': '2024-01-15T14:00:00-05:00'},
            'end': {'dateTime': '2024-01-15T15:00:00-05:00'},
            'location': 'Conference Room',
            'description': 'Test description',
            'attendees': [
                {'email': 'alice@example.com', 'responseStatus': 'accepted'},
                {'email': 'bob@example.com', 'responseStatus': 'tentative'}
            ],
            'htmlLink': 'https://calendar.google.com/event?id=event123'
        }

        result = cs.cmd_get(mock_args)

        assert result == cs.EXIT_SUCCESS

    def test_get_event_json_output(self, mock_service, mock_args, capsys):
        """Test getting event with JSON output."""
        mock_args.event_id = "event123"
        mock_args.calendar_id = None
        mock_args.json = True

        event_data = {
            'id': 'event123',
            'summary': 'Test Event',
            'start': {'dateTime': '2024-01-15T14:00:00-05:00'},
            'end': {'dateTime': '2024-01-15T15:00:00-05:00'}
        }
        mock_service.events().get().execute.return_value = event_data

        result = cs.cmd_get(mock_args)

        assert result == cs.EXIT_SUCCESS
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output['id'] == 'event123'

    def test_get_event_not_found(self, mock_service, mock_args):
        """Test getting non-existent event."""
        mock_args.event_id = "nonexistent"
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().get().execute.side_effect = MockHttpError(404)

        with patch('calendar_schedule.HttpError', MockHttpError):
            result = cs.cmd_get(mock_args)

        assert result == cs.EXIT_NOT_FOUND


class TestCmdList:
    """Tests for the list command."""

    def test_list_events_success(self, mock_service, mock_args):
        """Test listing events."""
        mock_args.start = "2024-01-15"
        mock_args.end = "2024-01-22"
        mock_args.max_results = 25
        mock_args.show_recurring = False
        mock_args.verbose = False
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().list().execute.return_value = {
            'items': [
                {
                    'id': 'event1',
                    'summary': 'Meeting 1',
                    'start': {'dateTime': '2024-01-15T10:00:00'},
                    'end': {'dateTime': '2024-01-15T11:00:00'}
                },
                {
                    'id': 'event2',
                    'summary': 'Meeting 2',
                    'start': {'dateTime': '2024-01-16T14:00:00'},
                    'end': {'dateTime': '2024-01-16T15:00:00'}
                }
            ]
        }

        result = cs.cmd_list(mock_args)

        assert result == cs.EXIT_SUCCESS

    def test_list_no_events(self, mock_service, mock_args, capsys):
        """Test listing when no events exist."""
        mock_args.start = "2024-01-15"
        mock_args.end = "2024-01-22"
        mock_args.max_results = 25
        mock_args.show_recurring = False
        mock_args.verbose = False
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().list().execute.return_value = {'items': []}

        result = cs.cmd_list(mock_args)

        assert result == cs.EXIT_SUCCESS
        captured = capsys.readouterr()
        assert "No upcoming events found" in captured.out

    def test_list_with_relative_dates(self, mock_service, mock_args):
        """Test listing with relative date formats."""
        mock_args.start = "today"
        mock_args.end = "+7 days"
        mock_args.max_results = 25
        mock_args.show_recurring = False
        mock_args.verbose = False
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().list().execute.return_value = {'items': []}

        result = cs.cmd_list(mock_args)

        assert result == cs.EXIT_SUCCESS

    def test_list_invalid_start_time(self, mock_service, mock_args):
        """Test listing with invalid start time."""
        mock_args.start = "invalid"
        mock_args.end = None
        mock_args.calendar_id = None

        result = cs.cmd_list(mock_args)

        assert result == cs.EXIT_INVALID_ARGS


class TestCmdFreeBusy:
    """Tests for the free-busy command."""

    def test_freebusy_success(self, mock_service, mock_args):
        """Test free/busy query."""
        mock_args.start = "2024-01-15 08:00"
        mock_args.end = "2024-01-15 18:00"
        mock_args.calendars = None
        mock_args.timezone = None
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.freebusy().query().execute.return_value = {
            'kind': 'calendar#freeBusy',
            'timeMin': '2024-01-15T08:00:00Z',
            'timeMax': '2024-01-15T18:00:00Z',
            'calendars': {
                'primary': {
                    'busy': [
                        {'start': '2024-01-15T10:00:00Z', 'end': '2024-01-15T11:00:00Z'},
                        {'start': '2024-01-15T14:00:00Z', 'end': '2024-01-15T15:00:00Z'}
                    ]
                }
            }
        }

        result = cs.cmd_free_busy(mock_args)

        assert result == cs.EXIT_SUCCESS

    def test_freebusy_multiple_calendars(self, mock_service, mock_args):
        """Test free/busy for multiple calendars."""
        mock_args.start = "2024-01-15 08:00"
        mock_args.end = "2024-01-15 18:00"
        mock_args.calendars = "cal1@example.com,cal2@example.com"
        mock_args.timezone = "America/New_York"
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.freebusy().query().execute.return_value = {
            'calendars': {
                'cal1@example.com': {'busy': []},
                'cal2@example.com': {
                    'busy': [
                        {'start': '2024-01-15T12:00:00Z', 'end': '2024-01-15T13:00:00Z'}
                    ]
                }
            }
        }

        result = cs.cmd_free_busy(mock_args)

        assert result == cs.EXIT_SUCCESS

    def test_freebusy_json_output(self, mock_service, mock_args, capsys):
        """Test free/busy with JSON output."""
        mock_args.start = "2024-01-15 08:00"
        mock_args.end = "2024-01-15 18:00"
        mock_args.calendars = None
        mock_args.timezone = None
        mock_args.calendar_id = None
        mock_args.json = True

        freebusy_data = {
            'calendars': {
                'primary': {'busy': []}
            }
        }
        mock_service.freebusy().query().execute.return_value = freebusy_data

        result = cs.cmd_free_busy(mock_args)

        assert result == cs.EXIT_SUCCESS
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert 'calendars' in output


class TestCreateParser:
    """Tests for the argument parser."""

    def test_parser_creates_successfully(self):
        """Test that parser is created without errors."""
        parser = cs.create_parser()
        assert parser is not None

    def test_parser_has_subcommands(self):
        """Test that all expected subcommands exist."""
        parser = cs.create_parser()

        # Test each subcommand can be parsed
        for cmd in ['create', 'update', 'delete', 'get', 'list', 'free-busy']:
            # These will raise SystemExit if the command doesn't exist
            if cmd == 'create':
                args = parser.parse_args([cmd, 'Test', '--start', '2024-01-15'])
            elif cmd in ['update', 'delete', 'get']:
                args = parser.parse_args([cmd, 'event123'])
            else:
                args = parser.parse_args([cmd])

            assert args.command == cmd

    def test_create_command_args(self):
        """Test create command argument parsing."""
        parser = cs.create_parser()
        args = parser.parse_args([
            'create', 'Meeting',
            '--start', '2024-01-15 14:00',
            '--duration', '60',
            '--attendees', 'a@x.com,b@x.com',
            '--location', 'Room A',
            '--description', 'Test meeting',
            '--recurrence', 'FREQ=WEEKLY;COUNT=5',
            '--notify'
        ])

        assert args.summary == 'Meeting'
        assert args.start == '2024-01-15 14:00'
        assert args.duration == 60
        assert args.attendees == 'a@x.com,b@x.com'
        assert args.location == 'Room A'
        assert args.description == 'Test meeting'
        assert args.recurrence == 'FREQ=WEEKLY;COUNT=5'
        assert args.notify is True

    def test_list_command_args(self):
        """Test list command argument parsing."""
        parser = cs.create_parser()
        args = parser.parse_args([
            'list',
            '--start', 'today',
            '--end', '+7 days',
            '--max-results', '50',
            '--verbose'
        ])

        assert args.start == 'today'
        assert args.end == '+7 days'
        assert args.max_results == 50
        assert args.verbose is True

    def test_freebusy_command_args(self):
        """Test free-busy command argument parsing."""
        parser = cs.create_parser()
        args = parser.parse_args([
            'free-busy',
            '--start', 'tomorrow',
            '--end', '+3 days',
            '--calendars', 'a@x.com,b@x.com',
            '--json'
        ])

        assert args.start == 'tomorrow'
        assert args.end == '+3 days'
        assert args.calendars == 'a@x.com,b@x.com'
        assert args.json is True

    def test_no_command_returns_invalid_args(self):
        """Test that no command shows help (would normally print help)."""
        parser = cs.create_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestMain:
    """Tests for the main entry point."""

    @patch('calendar_schedule.create_parser')
    def test_main_no_command(self, mock_create_parser):
        """Test main with no command exits with invalid args."""
        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.command = None
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        result = cs.main()

        assert result == cs.EXIT_INVALID_ARGS
        mock_parser.print_help.assert_called_once()

    @patch('calendar_schedule.create_parser')
    def test_main_with_command(self, mock_create_parser):
        """Test main with a valid command calls the handler."""
        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.command = 'list'
        mock_func = MagicMock(return_value=cs.EXIT_SUCCESS)
        mock_args.func = mock_func
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        result = cs.main()

        assert result == cs.EXIT_SUCCESS
        mock_func.assert_called_once_with(mock_args)


class TestExitCodes:
    """Tests to verify exit codes are correctly defined."""

    def test_exit_codes_defined(self):
        """Test that all exit codes are defined correctly."""
        assert cs.EXIT_SUCCESS == 0
        assert cs.EXIT_AUTH_ERROR == 1
        assert cs.EXIT_NOT_FOUND == 2
        assert cs.EXIT_API_ERROR == 3
        assert cs.EXIT_INVALID_ARGS == 4

    def test_exit_codes_unique(self):
        """Test that all exit codes are unique."""
        codes = [
            cs.EXIT_SUCCESS,
            cs.EXIT_AUTH_ERROR,
            cs.EXIT_NOT_FOUND,
            cs.EXIT_API_ERROR,
            cs.EXIT_INVALID_ARGS
        ]
        assert len(codes) == len(set(codes))


class TestRecurrenceRuleHandling:
    """Tests for recurrence rule handling."""

    def test_rrule_prefix_added(self, mock_service, mock_args):
        """Test that RRULE: prefix is added if missing."""
        mock_args.summary = "Test"
        mock_args.start = "2024-01-15 14:00"
        mock_args.end = None
        mock_args.duration = 60
        mock_args.description = None
        mock_args.location = None
        mock_args.attendees = None
        mock_args.recurrence = "FREQ=DAILY;COUNT=5"  # No RRULE: prefix
        mock_args.all_day = False
        mock_args.timezone = "UTC"
        mock_args.notify = False
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().insert().execute.return_value = {
            'id': 'test',
            'summary': 'Test'
        }

        cs.cmd_create(mock_args)

        # Verify the call was made with RRULE: prefix
        call_args = mock_service.events().insert.call_args
        body = call_args[1]['body']
        assert body['recurrence'][0].startswith('RRULE:')

    def test_rrule_prefix_not_duplicated(self, mock_service, mock_args):
        """Test that RRULE: prefix is not duplicated if already present."""
        mock_args.summary = "Test"
        mock_args.start = "2024-01-15 14:00"
        mock_args.end = None
        mock_args.duration = 60
        mock_args.description = None
        mock_args.location = None
        mock_args.attendees = None
        mock_args.recurrence = "RRULE:FREQ=DAILY;COUNT=5"  # Already has prefix
        mock_args.all_day = False
        mock_args.timezone = "UTC"
        mock_args.notify = False
        mock_args.calendar_id = None
        mock_args.json = False

        mock_service.events().insert().execute.return_value = {
            'id': 'test',
            'summary': 'Test'
        }

        cs.cmd_create(mock_args)

        call_args = mock_service.events().insert.call_args
        body = call_args[1]['body']
        assert body['recurrence'][0] == 'RRULE:FREQ=DAILY;COUNT=5'
        assert not body['recurrence'][0].startswith('RRULE:RRULE:')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
