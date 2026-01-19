#!/usr/bin/env python3
"""
calendar_schedule - A self-documenting CLI tool for Google Calendar management.

This tool provides comprehensive Google Calendar management capabilities including
creating, updating, deleting events, checking free/busy times, and listing upcoming
events with support for recurring events.

Exit Codes:
    0 - Success
    1 - Authentication error (missing credentials, invalid service account)
    2 - Resource not found (event ID doesn't exist)
    3 - API error (rate limits, server errors, network issues)
    4 - Invalid arguments (bad date format, missing required args)

Environment Variables:
    GOOGLE_APPLICATION_CREDENTIALS - Path to service account JSON key file
    CALENDAR_ID - Default calendar ID (optional, defaults to 'primary')

Examples:
    # Create a simple meeting
    calendar_schedule create "Team Standup" --start "2024-01-15 09:00" --duration 30

    # Create meeting with attendees and location
    calendar_schedule create "Project Review" \\
        --start "2024-01-15 14:00" \\
        --duration 60 \\
        --attendees "alice@example.com,bob@example.com" \\
        --location "Conference Room A" \\
        --description "Quarterly project review meeting"

    # Create recurring weekly meeting
    calendar_schedule create "Weekly Sync" \\
        --start "2024-01-15 10:00" \\
        --duration 45 \\
        --recurrence "FREQ=WEEKLY;BYDAY=MO;COUNT=10"

    # Create all-day event
    calendar_schedule create "Company Holiday" \\
        --start "2024-01-15" \\
        --all-day

    # List upcoming events for next 7 days
    calendar_schedule list --start today --end "+7 days"

    # List events in a specific date range
    calendar_schedule list --start "2024-01-01" --end "2024-01-31" --max-results 50

    # Check free/busy times
    calendar_schedule free-busy --start tomorrow --end "+3 days"

    # Check free/busy for multiple calendars
    calendar_schedule free-busy \\
        --start "2024-01-15 08:00" \\
        --end "2024-01-15 18:00" \\
        --calendars "team@example.com,room@example.com"

    # Update an existing event
    calendar_schedule update EVENT_ID --summary "New Title" --duration 90

    # Delete an event
    calendar_schedule delete EVENT_ID

    # Delete event and notify attendees
    calendar_schedule delete EVENT_ID --notify

    # Get event details
    calendar_schedule get EVENT_ID

Author: Generated with Claude Code
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Exit codes
EXIT_SUCCESS = 0
EXIT_AUTH_ERROR = 1
EXIT_NOT_FOUND = 2
EXIT_API_ERROR = 3
EXIT_INVALID_ARGS = 4

# Google Calendar API imports
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import google.auth
except ImportError as e:
    print(f"Error: Required Google API libraries not installed: {e}", file=sys.stderr)
    print("Install with: pip install google-api-python-client google-auth", file=sys.stderr)
    sys.exit(EXIT_AUTH_ERROR)


# Calendar API scopes
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events',
]


def parse_datetime(date_str: str, reference: Optional[datetime] = None) -> datetime:
    """
    Parse a flexible datetime string into a datetime object.

    Supports:
        - ISO format: "2024-01-15T14:00:00"
        - Date and time: "2024-01-15 14:00"
        - Date only: "2024-01-15"
        - Relative: "today", "tomorrow", "+7 days", "+1 week"

    Args:
        date_str: The date/time string to parse
        reference: Reference datetime for relative dates (defaults to now)

    Returns:
        Parsed datetime object

    Raises:
        ValueError: If the date string cannot be parsed
    """
    if reference is None:
        reference = datetime.now()

    date_str = date_str.strip().lower()

    # Handle relative dates
    if date_str == 'today':
        return reference.replace(hour=0, minute=0, second=0, microsecond=0)
    elif date_str == 'tomorrow':
        return (reference + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif date_str == 'now':
        return reference
    elif date_str.startswith('+'):
        # Parse relative format like "+7 days", "+1 week", "+2 hours"
        parts = date_str[1:].strip().split()
        if len(parts) >= 2:
            try:
                amount = int(parts[0])
                unit = parts[1].rstrip('s')  # Remove plural 's'

                if unit in ('day', 'd'):
                    return reference + timedelta(days=amount)
                elif unit in ('week', 'w'):
                    return reference + timedelta(weeks=amount)
                elif unit in ('hour', 'h'):
                    return reference + timedelta(hours=amount)
                elif unit in ('minute', 'min', 'm'):
                    return reference + timedelta(minutes=amount)
            except ValueError:
                pass

    # Try various datetime formats
    formats = [
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%m/%d/%Y %H:%M',
        '%m/%d/%Y',
        '%d-%m-%Y %H:%M',
        '%d-%m-%Y',
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse date: '{date_str}'. Use formats like '2024-01-15 14:00', 'today', or '+7 days'")


def format_datetime_rfc3339(dt: datetime) -> str:
    """Format datetime to RFC3339 format for Google Calendar API."""
    return dt.strftime('%Y-%m-%dT%H:%M:%S')


def format_date(dt: datetime) -> str:
    """Format datetime to date-only format for all-day events."""
    return dt.strftime('%Y-%m-%d')


def get_credentials():
    """
    Get Google API credentials from environment.

    Looks for GOOGLE_APPLICATION_CREDENTIALS environment variable pointing
    to a service account JSON key file.

    Returns:
        google.oauth2.service_account.Credentials object

    Raises:
        SystemExit: With EXIT_AUTH_ERROR if credentials cannot be loaded
    """
    creds_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

    if not creds_file:
        print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.", file=sys.stderr)
        print("Set it to the path of your service account JSON key file.", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json", file=sys.stderr)
        sys.exit(EXIT_AUTH_ERROR)

    if not os.path.exists(creds_file):
        print(f"Error: Credentials file not found: {creds_file}", file=sys.stderr)
        sys.exit(EXIT_AUTH_ERROR)

    try:
        credentials = service_account.Credentials.from_service_account_file(
            creds_file,
            scopes=SCOPES
        )
        return credentials
    except Exception as e:
        print(f"Error: Failed to load credentials: {e}", file=sys.stderr)
        sys.exit(EXIT_AUTH_ERROR)


def get_calendar_service(credentials=None):
    """
    Build and return the Google Calendar API service.

    Args:
        credentials: Optional credentials object. If None, loads from environment.

    Returns:
        Google Calendar API service object
    """
    if credentials is None:
        credentials = get_credentials()

    try:
        service = build('calendar', 'v3', credentials=credentials)
        return service
    except Exception as e:
        print(f"Error: Failed to build Calendar API service: {e}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)


def get_calendar_id() -> str:
    """Get the calendar ID from environment or default to 'primary'."""
    return os.environ.get('CALENDAR_ID', 'primary')


def cmd_create(args) -> int:
    """
    Create a new calendar event.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    service = get_calendar_service()
    calendar_id = args.calendar_id or get_calendar_id()

    # Parse start time
    try:
        start_dt = parse_datetime(args.start)
    except ValueError as e:
        print(f"Error: Invalid start time - {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS

    # Build event body
    event = {
        'summary': args.summary,
    }

    # Handle all-day events vs timed events
    if args.all_day:
        event['start'] = {'date': format_date(start_dt)}
        if args.end:
            try:
                end_dt = parse_datetime(args.end)
            except ValueError as e:
                print(f"Error: Invalid end time - {e}", file=sys.stderr)
                return EXIT_INVALID_ARGS
        else:
            end_dt = start_dt + timedelta(days=1)
        event['end'] = {'date': format_date(end_dt)}
    else:
        timezone = args.timezone or 'UTC'
        event['start'] = {
            'dateTime': format_datetime_rfc3339(start_dt),
            'timeZone': timezone,
        }

        # Calculate end time
        if args.end:
            try:
                end_dt = parse_datetime(args.end)
            except ValueError as e:
                print(f"Error: Invalid end time - {e}", file=sys.stderr)
                return EXIT_INVALID_ARGS
        elif args.duration:
            end_dt = start_dt + timedelta(minutes=args.duration)
        else:
            # Default to 1 hour
            end_dt = start_dt + timedelta(hours=1)

        event['end'] = {
            'dateTime': format_datetime_rfc3339(end_dt),
            'timeZone': timezone,
        }

    # Optional fields
    if args.description:
        event['description'] = args.description

    if args.location:
        event['location'] = args.location

    if args.attendees:
        attendee_list = [email.strip() for email in args.attendees.split(',')]
        event['attendees'] = [{'email': email} for email in attendee_list]

    if args.recurrence:
        # Ensure RRULE prefix
        rrule = args.recurrence
        if not rrule.upper().startswith('RRULE:'):
            rrule = f'RRULE:{rrule}'
        event['recurrence'] = [rrule]

    # Send notifications
    send_updates = 'all' if args.notify else 'none'

    try:
        created_event = service.events().insert(
            calendarId=calendar_id,
            body=event,
            sendUpdates=send_updates
        ).execute()

        print(f"Event created successfully!")
        print(f"  ID: {created_event['id']}")
        print(f"  Summary: {created_event.get('summary', 'N/A')}")
        print(f"  Link: {created_event.get('htmlLink', 'N/A')}")

        if args.json:
            print("\nFull response:")
            print(json.dumps(created_event, indent=2))

        return EXIT_SUCCESS

    except HttpError as e:
        if e.resp.status == 404:
            print(f"Error: Calendar not found: {calendar_id}", file=sys.stderr)
            return EXIT_NOT_FOUND
        else:
            print(f"Error: API error - {e}", file=sys.stderr)
            return EXIT_API_ERROR


def cmd_update(args) -> int:
    """
    Update an existing calendar event.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    service = get_calendar_service()
    calendar_id = args.calendar_id or get_calendar_id()

    # First, get the existing event
    try:
        event = service.events().get(
            calendarId=calendar_id,
            eventId=args.event_id
        ).execute()
    except HttpError as e:
        if e.resp.status == 404:
            print(f"Error: Event not found: {args.event_id}", file=sys.stderr)
            return EXIT_NOT_FOUND
        else:
            print(f"Error: API error - {e}", file=sys.stderr)
            return EXIT_API_ERROR

    # Update fields if provided
    if args.summary:
        event['summary'] = args.summary

    if args.description:
        event['description'] = args.description

    if args.location:
        event['location'] = args.location

    if args.start:
        try:
            start_dt = parse_datetime(args.start)
        except ValueError as e:
            print(f"Error: Invalid start time - {e}", file=sys.stderr)
            return EXIT_INVALID_ARGS

        timezone = args.timezone or event.get('start', {}).get('timeZone', 'UTC')
        event['start'] = {
            'dateTime': format_datetime_rfc3339(start_dt),
            'timeZone': timezone,
        }

        # If duration is set, update end time
        if args.duration:
            end_dt = start_dt + timedelta(minutes=args.duration)
            event['end'] = {
                'dateTime': format_datetime_rfc3339(end_dt),
                'timeZone': timezone,
            }
    elif args.duration and 'start' in event and 'dateTime' in event['start']:
        # Update duration only
        try:
            start_dt = datetime.fromisoformat(event['start']['dateTime'].replace('Z', '+00:00'))
            # Remove timezone info for calculation
            start_dt = start_dt.replace(tzinfo=None)
        except (ValueError, KeyError):
            print("Error: Cannot update duration - unable to parse existing start time", file=sys.stderr)
            return EXIT_INVALID_ARGS

        end_dt = start_dt + timedelta(minutes=args.duration)
        timezone = event.get('start', {}).get('timeZone', 'UTC')
        event['end'] = {
            'dateTime': format_datetime_rfc3339(end_dt),
            'timeZone': timezone,
        }

    if args.attendees:
        attendee_list = [email.strip() for email in args.attendees.split(',')]
        event['attendees'] = [{'email': email} for email in attendee_list]

    if args.recurrence:
        rrule = args.recurrence
        if not rrule.upper().startswith('RRULE:'):
            rrule = f'RRULE:{rrule}'
        event['recurrence'] = [rrule]

    send_updates = 'all' if args.notify else 'none'

    try:
        updated_event = service.events().update(
            calendarId=calendar_id,
            eventId=args.event_id,
            body=event,
            sendUpdates=send_updates
        ).execute()

        print(f"Event updated successfully!")
        print(f"  ID: {updated_event['id']}")
        print(f"  Summary: {updated_event.get('summary', 'N/A')}")

        if args.json:
            print("\nFull response:")
            print(json.dumps(updated_event, indent=2))

        return EXIT_SUCCESS

    except HttpError as e:
        print(f"Error: API error - {e}", file=sys.stderr)
        return EXIT_API_ERROR


def cmd_delete(args) -> int:
    """
    Delete a calendar event.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    service = get_calendar_service()
    calendar_id = args.calendar_id or get_calendar_id()

    send_updates = 'all' if args.notify else 'none'

    try:
        service.events().delete(
            calendarId=calendar_id,
            eventId=args.event_id,
            sendUpdates=send_updates
        ).execute()

        print(f"Event deleted successfully: {args.event_id}")
        return EXIT_SUCCESS

    except HttpError as e:
        if e.resp.status == 404:
            print(f"Error: Event not found: {args.event_id}", file=sys.stderr)
            return EXIT_NOT_FOUND
        elif e.resp.status == 410:
            print(f"Error: Event already deleted: {args.event_id}", file=sys.stderr)
            return EXIT_NOT_FOUND
        else:
            print(f"Error: API error - {e}", file=sys.stderr)
            return EXIT_API_ERROR


def cmd_get(args) -> int:
    """
    Get details of a specific event.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    service = get_calendar_service()
    calendar_id = args.calendar_id or get_calendar_id()

    try:
        event = service.events().get(
            calendarId=calendar_id,
            eventId=args.event_id
        ).execute()

        if args.json:
            print(json.dumps(event, indent=2))
        else:
            print(f"Event Details:")
            print(f"  ID: {event['id']}")
            print(f"  Summary: {event.get('summary', 'N/A')}")
            print(f"  Status: {event.get('status', 'N/A')}")

            start = event.get('start', {})
            if 'dateTime' in start:
                print(f"  Start: {start['dateTime']}")
            elif 'date' in start:
                print(f"  Start: {start['date']} (all-day)")

            end = event.get('end', {})
            if 'dateTime' in end:
                print(f"  End: {end['dateTime']}")
            elif 'date' in end:
                print(f"  End: {end['date']}")

            if event.get('location'):
                print(f"  Location: {event['location']}")

            if event.get('description'):
                print(f"  Description: {event['description'][:100]}...")

            if event.get('attendees'):
                print(f"  Attendees:")
                for attendee in event['attendees']:
                    status = attendee.get('responseStatus', 'unknown')
                    print(f"    - {attendee.get('email')} ({status})")

            if event.get('recurrence'):
                print(f"  Recurrence: {', '.join(event['recurrence'])}")

            print(f"  Link: {event.get('htmlLink', 'N/A')}")

        return EXIT_SUCCESS

    except HttpError as e:
        if e.resp.status == 404:
            print(f"Error: Event not found: {args.event_id}", file=sys.stderr)
            return EXIT_NOT_FOUND
        else:
            print(f"Error: API error - {e}", file=sys.stderr)
            return EXIT_API_ERROR


def cmd_list(args) -> int:
    """
    List upcoming calendar events.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    service = get_calendar_service()
    calendar_id = args.calendar_id or get_calendar_id()

    # Parse time range
    try:
        if args.start:
            start_dt = parse_datetime(args.start)
        else:
            start_dt = datetime.now()
    except ValueError as e:
        print(f"Error: Invalid start time - {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS

    try:
        if args.end:
            end_dt = parse_datetime(args.end, reference=start_dt)
        else:
            end_dt = start_dt + timedelta(days=7)
    except ValueError as e:
        print(f"Error: Invalid end time - {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS

    time_min = format_datetime_rfc3339(start_dt) + 'Z'
    time_max = format_datetime_rfc3339(end_dt) + 'Z'

    try:
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            maxResults=args.max_results,
            singleEvents=not args.show_recurring,
            orderBy='startTime' if not args.show_recurring else 'updated'
        ).execute()

        events = events_result.get('items', [])

        if args.json:
            print(json.dumps(events, indent=2))
        else:
            if not events:
                print("No upcoming events found.")
            else:
                print(f"Found {len(events)} event(s):\n")
                for event in events:
                    start = event.get('start', {})
                    if 'dateTime' in start:
                        start_str = start['dateTime']
                    else:
                        start_str = start.get('date', 'N/A') + ' (all-day)'

                    summary = event.get('summary', '(No title)')
                    event_id = event['id']

                    print(f"  [{start_str}] {summary}")
                    print(f"    ID: {event_id}")

                    if event.get('location'):
                        print(f"    Location: {event['location']}")

                    if args.verbose and event.get('attendees'):
                        attendees = [a.get('email') for a in event['attendees']]
                        print(f"    Attendees: {', '.join(attendees)}")

                    print()

        return EXIT_SUCCESS

    except HttpError as e:
        if e.resp.status == 404:
            print(f"Error: Calendar not found: {calendar_id}", file=sys.stderr)
            return EXIT_NOT_FOUND
        else:
            print(f"Error: API error - {e}", file=sys.stderr)
            return EXIT_API_ERROR


def cmd_free_busy(args) -> int:
    """
    Check free/busy times for calendars.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    service = get_calendar_service()

    # Parse time range
    try:
        if args.start:
            start_dt = parse_datetime(args.start)
        else:
            start_dt = datetime.now()
    except ValueError as e:
        print(f"Error: Invalid start time - {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS

    try:
        if args.end:
            end_dt = parse_datetime(args.end, reference=start_dt)
        else:
            end_dt = start_dt + timedelta(days=1)
    except ValueError as e:
        print(f"Error: Invalid end time - {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS

    # Build calendar list
    if args.calendars:
        calendar_ids = [cal.strip() for cal in args.calendars.split(',')]
    else:
        calendar_ids = [args.calendar_id or get_calendar_id()]

    items = [{'id': cal_id} for cal_id in calendar_ids]

    body = {
        'timeMin': format_datetime_rfc3339(start_dt) + 'Z',
        'timeMax': format_datetime_rfc3339(end_dt) + 'Z',
        'items': items,
    }

    if args.timezone:
        body['timeZone'] = args.timezone

    try:
        freebusy_result = service.freebusy().query(body=body).execute()

        if args.json:
            print(json.dumps(freebusy_result, indent=2))
        else:
            print(f"Free/Busy Information")
            print(f"  Time Range: {start_dt} to {end_dt}")
            print()

            calendars = freebusy_result.get('calendars', {})

            for cal_id, cal_data in calendars.items():
                print(f"Calendar: {cal_id}")

                if cal_data.get('errors'):
                    for error in cal_data['errors']:
                        print(f"  Error: {error.get('reason', 'Unknown error')}")
                    continue

                busy_times = cal_data.get('busy', [])

                if not busy_times:
                    print("  Status: Free (no busy times)")
                else:
                    print(f"  Busy periods ({len(busy_times)}):")
                    for busy in busy_times:
                        start_str = busy.get('start', 'N/A')
                        end_str = busy.get('end', 'N/A')
                        print(f"    - {start_str} to {end_str}")

                print()

        return EXIT_SUCCESS

    except HttpError as e:
        print(f"Error: API error - {e}", file=sys.stderr)
        return EXIT_API_ERROR


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser with all subcommands.

    Returns:
        Configured ArgumentParser object
    """
    # Main parser
    parser = argparse.ArgumentParser(
        prog='calendar_schedule',
        description=textwrap.dedent('''
            A self-documenting CLI tool for Google Calendar management.

            Supports creating, updating, deleting events, checking free/busy times,
            and listing upcoming events with support for recurring events.
        '''),
        epilog=textwrap.dedent('''
            Exit Codes:
                0  Success
                1  Authentication error
                2  Resource not found
                3  API error
                4  Invalid arguments

            Environment Variables:
                GOOGLE_APPLICATION_CREDENTIALS  Path to service account JSON key file (required)
                CALENDAR_ID                     Default calendar ID (optional, defaults to 'primary')

            Examples:
                %(prog)s create "Meeting" --start "2024-01-15 14:00" --duration 60
                %(prog)s list --start today --end "+7 days"
                %(prog)s free-busy --start tomorrow --end "+3 days"
                %(prog)s delete EVENT_ID

            For more detailed examples, use: %(prog)s <command> --help
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    subparsers = parser.add_subparsers(
        title='commands',
        dest='command',
        description='Available commands',
        metavar='<command>'
    )

    # Common arguments function
    def add_common_args(subparser):
        subparser.add_argument(
            '--calendar-id',
            help='Calendar ID (default: primary or CALENDAR_ID env var)'
        )
        subparser.add_argument(
            '--json',
            action='store_true',
            help='Output full JSON response'
        )

    # CREATE command
    create_parser = subparsers.add_parser(
        'create',
        help='Create a new calendar event',
        description='Create a new event with optional attendees, location, and recurrence.',
        epilog=textwrap.dedent('''
            Examples:
                # Simple 1-hour meeting
                %(prog)s "Team Meeting" --start "2024-01-15 14:00"

                # Meeting with duration and attendees
                %(prog)s "Project Review" --start "2024-01-15 14:00" --duration 60 \\
                    --attendees "alice@example.com,bob@example.com"

                # Recurring weekly meeting
                %(prog)s "Weekly Sync" --start "2024-01-15 10:00" --duration 45 \\
                    --recurrence "FREQ=WEEKLY;BYDAY=MO;COUNT=10"

                # All-day event
                %(prog)s "Company Holiday" --start "2024-01-15" --all-day

            Recurrence Rule Examples (RRULE format):
                FREQ=DAILY;COUNT=5                    Daily for 5 days
                FREQ=WEEKLY;BYDAY=MO,WE,FR            Every Mon, Wed, Fri
                FREQ=WEEKLY;BYDAY=TU;COUNT=10         Every Tuesday, 10 times
                FREQ=MONTHLY;BYMONTHDAY=15            Monthly on the 15th
                FREQ=YEARLY;BYMONTH=1;BYMONTHDAY=1    Every January 1st
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    create_parser.add_argument(
        'summary',
        help='Event title/summary'
    )
    create_parser.add_argument(
        '--start', '-s',
        required=True,
        help='Start time (e.g., "2024-01-15 14:00", "today", "tomorrow", "+1 day")'
    )
    create_parser.add_argument(
        '--end', '-e',
        help='End time (default: start + duration or 1 hour)'
    )
    create_parser.add_argument(
        '--duration', '-d',
        type=int,
        help='Duration in minutes (default: 60)'
    )
    create_parser.add_argument(
        '--description',
        help='Event description'
    )
    create_parser.add_argument(
        '--location', '-l',
        help='Event location'
    )
    create_parser.add_argument(
        '--attendees', '-a',
        help='Comma-separated list of attendee emails'
    )
    create_parser.add_argument(
        '--recurrence', '-r',
        help='Recurrence rule in RRULE format (e.g., "FREQ=WEEKLY;BYDAY=MO;COUNT=10")'
    )
    create_parser.add_argument(
        '--all-day',
        action='store_true',
        help='Create an all-day event'
    )
    create_parser.add_argument(
        '--timezone', '-tz',
        help='Timezone (default: UTC)'
    )
    create_parser.add_argument(
        '--notify', '-n',
        action='store_true',
        help='Send email notifications to attendees'
    )
    add_common_args(create_parser)
    create_parser.set_defaults(func=cmd_create)

    # UPDATE command
    update_parser = subparsers.add_parser(
        'update',
        help='Update an existing calendar event',
        description='Update one or more fields of an existing event.',
        epilog=textwrap.dedent('''
            Examples:
                # Change event title
                %(prog)s EVENT_ID --summary "New Meeting Title"

                # Update time and duration
                %(prog)s EVENT_ID --start "2024-01-16 15:00" --duration 90

                # Add attendees
                %(prog)s EVENT_ID --attendees "new@example.com,other@example.com"
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    update_parser.add_argument(
        'event_id',
        help='Event ID to update'
    )
    update_parser.add_argument(
        '--summary',
        help='New event title'
    )
    update_parser.add_argument(
        '--start', '-s',
        help='New start time'
    )
    update_parser.add_argument(
        '--duration', '-d',
        type=int,
        help='New duration in minutes'
    )
    update_parser.add_argument(
        '--description',
        help='New event description'
    )
    update_parser.add_argument(
        '--location', '-l',
        help='New event location'
    )
    update_parser.add_argument(
        '--attendees', '-a',
        help='New comma-separated list of attendee emails'
    )
    update_parser.add_argument(
        '--recurrence', '-r',
        help='New recurrence rule'
    )
    update_parser.add_argument(
        '--timezone', '-tz',
        help='Timezone'
    )
    update_parser.add_argument(
        '--notify', '-n',
        action='store_true',
        help='Send email notifications to attendees'
    )
    add_common_args(update_parser)
    update_parser.set_defaults(func=cmd_update)

    # DELETE command
    delete_parser = subparsers.add_parser(
        'delete',
        help='Delete a calendar event',
        description='Delete an event by its ID.',
        epilog=textwrap.dedent('''
            Examples:
                # Delete an event
                %(prog)s EVENT_ID

                # Delete and notify attendees
                %(prog)s EVENT_ID --notify
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    delete_parser.add_argument(
        'event_id',
        help='Event ID to delete'
    )
    delete_parser.add_argument(
        '--notify', '-n',
        action='store_true',
        help='Send cancellation emails to attendees'
    )
    add_common_args(delete_parser)
    delete_parser.set_defaults(func=cmd_delete)

    # GET command
    get_parser = subparsers.add_parser(
        'get',
        help='Get details of a specific event',
        description='Retrieve and display details of a calendar event.',
        epilog=textwrap.dedent('''
            Examples:
                # Get event details
                %(prog)s EVENT_ID

                # Get event as JSON
                %(prog)s EVENT_ID --json
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser.add_argument(
        'event_id',
        help='Event ID to retrieve'
    )
    add_common_args(get_parser)
    get_parser.set_defaults(func=cmd_get)

    # LIST command
    list_parser = subparsers.add_parser(
        'list',
        help='List upcoming calendar events',
        description='List events within a time range.',
        epilog=textwrap.dedent('''
            Examples:
                # List events for next 7 days
                %(prog)s --start today --end "+7 days"

                # List events for a specific date range
                %(prog)s --start "2024-01-01" --end "2024-01-31"

                # List with more results
                %(prog)s --start today --end "+30 days" --max-results 100

                # Show recurring event series (not expanded)
                %(prog)s --start today --end "+30 days" --show-recurring
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    list_parser.add_argument(
        '--start', '-s',
        help='Start of time range (default: now)'
    )
    list_parser.add_argument(
        '--end', '-e',
        help='End of time range (default: +7 days)'
    )
    list_parser.add_argument(
        '--max-results', '-m',
        type=int,
        default=25,
        help='Maximum number of events to return (default: 25)'
    )
    list_parser.add_argument(
        '--show-recurring',
        action='store_true',
        help='Show recurring events as series (not expanded to instances)'
    )
    list_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show additional details like attendees'
    )
    add_common_args(list_parser)
    list_parser.set_defaults(func=cmd_list)

    # FREE-BUSY command
    freebusy_parser = subparsers.add_parser(
        'free-busy',
        help='Check free/busy times for calendars',
        description='Query free/busy information for one or more calendars.',
        epilog=textwrap.dedent('''
            Examples:
                # Check your free/busy for tomorrow
                %(prog)s --start tomorrow --end "+1 day"

                # Check multiple calendars
                %(prog)s --start "2024-01-15 08:00" --end "2024-01-15 18:00" \\
                    --calendars "team@example.com,room@resource.calendar.google.com"

                # Check for next 3 days
                %(prog)s --start today --end "+3 days"
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    freebusy_parser.add_argument(
        '--start', '-s',
        help='Start of time range (default: now)'
    )
    freebusy_parser.add_argument(
        '--end', '-e',
        help='End of time range (default: +1 day)'
    )
    freebusy_parser.add_argument(
        '--calendars', '-c',
        help='Comma-separated list of calendar IDs to check'
    )
    freebusy_parser.add_argument(
        '--timezone', '-tz',
        help='Timezone for the query'
    )
    add_common_args(freebusy_parser)
    freebusy_parser.set_defaults(func=cmd_free_busy)

    return parser


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return EXIT_INVALID_ARGS

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
