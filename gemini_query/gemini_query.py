#!/usr/bin/env python3
"""
gemini_query - Analyze media files with Google Gemini API

A CLI tool for AI agents to analyze PDFs, images, and audio files using
Google's Gemini multimodal models. Streams responses to stdout.

Environment Variables:
    GOOGLE_API_KEY: Your Google AI API key (required)

Supported Formats:
    Images: JPEG, PNG, GIF, WebP, HEIC, HEIF
    Audio:  WAV, MP3, AIFF, AAC, OGG, FLAC
    Documents: PDF (up to 3600 pages)

Usage:
    gemini_query -f document.pdf -p "Summarize this document"
    gemini_query -f image.png -p "Describe what you see"
    gemini_query -f audio.mp3 -p "Transcribe this audio"
    gemini_query -f report.pdf -p "Extract all tables as JSON" --json

Exit Codes:
    0: Success
    1: Missing API key
    2: File not found or unsupported format
    3: API error
    4: Invalid arguments
"""

import argparse
import mimetypes
import os
import sys
from pathlib import Path
from typing import Optional, Generator

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai package not installed. Run: pip install google-genai", file=sys.stderr)
    sys.exit(1)


# Supported MIME types by category
SUPPORTED_IMAGES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "image/heic", "image/heif"
}

SUPPORTED_AUDIO = {
    "audio/wav", "audio/x-wav", "audio/mp3", "audio/mpeg", "audio/aiff",
    "audio/aac", "audio/ogg", "audio/flac"
}

SUPPORTED_DOCUMENTS = {
    "application/pdf"
}

ALL_SUPPORTED = SUPPORTED_IMAGES | SUPPORTED_AUDIO | SUPPORTED_DOCUMENTS

# File size threshold for using Files API (20MB)
INLINE_SIZE_LIMIT = 20 * 1024 * 1024

# Default model
DEFAULT_MODEL = "gemini-2.5-flash"


def get_api_key() -> str:
    """
    Get API key from environment.

    Returns:
        API key string

    Raises:
        SystemExit: If GOOGLE_API_KEY not set
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set", file=sys.stderr)
        print("\nGet your API key from: https://aistudio.google.com/apikey", file=sys.stderr)
        sys.exit(1)
    return api_key


def detect_mime_type(filepath: Path) -> str:
    """
    Detect MIME type of a file.

    Args:
        filepath: Path to the file

    Returns:
        MIME type string

    Raises:
        ValueError: If MIME type cannot be determined or is unsupported
    """
    mime_type, _ = mimetypes.guess_type(str(filepath))

    # Handle common extensions that might not be in mimetypes
    extension_map = {
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aiff": "audio/aiff",
        ".heic": "image/heic",
        ".heif": "image/heif",
    }

    if mime_type is None:
        suffix = filepath.suffix.lower()
        mime_type = extension_map.get(suffix)

    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for: {filepath}")

    if mime_type not in ALL_SUPPORTED:
        raise ValueError(
            f"Unsupported MIME type: {mime_type}\n"
            f"Supported: images (JPEG/PNG/GIF/WebP), audio (WAV/MP3/AAC/OGG/FLAC), PDF"
        )

    return mime_type


def query_gemini_stream(
    client: genai.Client,
    filepath: Path,
    prompt: str,
    model: str = DEFAULT_MODEL,
    use_files_api: bool = False
) -> Generator[str, None, None]:
    """
    Query Gemini with a media file and stream the response.

    Args:
        client: Initialized Gemini client
        filepath: Path to the media file
        prompt: Query prompt
        model: Model name to use
        use_files_api: Force using Files API for upload

    Yields:
        Response text chunks
    """
    mime_type = detect_mime_type(filepath)
    file_size = filepath.stat().st_size

    # Decide whether to use inline data or Files API
    if use_files_api or file_size > INLINE_SIZE_LIMIT:
        # Upload via Files API
        uploaded_file = client.files.upload(file=filepath)
        contents = [uploaded_file, prompt]
    else:
        # Use inline data
        data = filepath.read_bytes()
        part = types.Part.from_bytes(data=data, mime_type=mime_type)
        contents = [part, prompt]

    # Stream the response
    response = client.models.generate_content_stream(
        model=model,
        contents=contents
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text


def query_gemini(
    client: genai.Client,
    filepath: Path,
    prompt: str,
    model: str = DEFAULT_MODEL,
    use_files_api: bool = False
) -> str:
    """
    Query Gemini with a media file and return the full response.

    Args:
        client: Initialized Gemini client
        filepath: Path to the media file
        prompt: Query prompt
        model: Model name to use
        use_files_api: Force using Files API for upload

    Returns:
        Complete response text
    """
    mime_type = detect_mime_type(filepath)
    file_size = filepath.stat().st_size

    if use_files_api or file_size > INLINE_SIZE_LIMIT:
        uploaded_file = client.files.upload(file=filepath)
        contents = [uploaded_file, prompt]
    else:
        data = filepath.read_bytes()
        part = types.Part.from_bytes(data=data, mime_type=mime_type)
        contents = [part, prompt]

    response = client.models.generate_content(
        model=model,
        contents=contents
    )

    return response.text


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="gemini_query",
        description="Analyze media files (images, PDFs, audio) with Google Gemini API. "
                    "Streams responses to stdout for efficient processing.",
        epilog="""
Examples:
  %(prog)s -f document.pdf -p "Summarize this document"
  %(prog)s -f screenshot.png -p "Extract all text from this image"
  %(prog)s -f meeting.mp3 -p "Transcribe and list action items"
  %(prog)s -f report.pdf -p "Extract tables as JSON" --json
  %(prog)s -f chart.png -p "Describe trends" --model gemini-2.5-pro

Supported Formats:
  Images:    JPEG, PNG, GIF, WebP, HEIC, HEIF
  Audio:     WAV, MP3, AIFF, AAC, OGG, FLAC
  Documents: PDF (up to 3600 pages)

Environment Variables:
  GOOGLE_API_KEY  Your Google AI API key (get from aistudio.google.com)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-f", "--file",
        required=True,
        type=Path,
        help="Path to the media file to analyze"
    )

    parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="Query prompt describing what to extract or analyze"
    )

    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})"
    )

    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Return complete response instead of streaming"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Hint to model to return JSON output (adds instruction to prompt)"
    )

    parser.add_argument(
        "--use-files-api",
        action="store_true",
        help="Force using Files API for upload (automatic for files >20MB)"
    )

    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Show token count before processing"
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate file exists
    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 2

    # Validate MIME type early
    try:
        detect_mime_type(args.file)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Get API key and initialize client
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    # Modify prompt for JSON output if requested
    prompt = args.prompt
    if args.json:
        prompt = f"{prompt}\n\nRespond with valid JSON only, no markdown code blocks."

    # Show token count if requested
    if args.show_tokens:
        try:
            mime_type = detect_mime_type(args.file)
            data = args.file.read_bytes()
            part = types.Part.from_bytes(data=data, mime_type=mime_type)
            token_response = client.models.count_tokens(
                model=args.model,
                contents=[part, prompt]
            )
            print(f"Input tokens: {token_response.total_tokens}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not count tokens: {e}", file=sys.stderr)

    try:
        if args.no_stream:
            # Return complete response
            result = query_gemini(
                client=client,
                filepath=args.file,
                prompt=prompt,
                model=args.model,
                use_files_api=args.use_files_api
            )
            print(result)
        else:
            # Stream response to stdout
            for chunk in query_gemini_stream(
                client=client,
                filepath=args.file,
                prompt=prompt,
                model=args.model,
                use_files_api=args.use_files_api
            ):
                print(chunk, end="", flush=True)
            print()  # Final newline

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
