#!/usr/bin/env python3
"""Tests for gemini_query CLI tool."""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Import the module under test
import gemini_query


class TestGetApiKey:
    """Tests for get_api_key function."""

    def test_returns_api_key_when_set(self, monkeypatch):
        """Should return API key from environment."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-123")

        result = gemini_query.get_api_key()

        assert result == "test-api-key-123"

    def test_exits_when_not_set(self, monkeypatch):
        """Should exit with code 1 when API key is missing."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            gemini_query.get_api_key()

        assert exc_info.value.code == 1


class TestDetectMimeType:
    """Tests for detect_mime_type function."""

    def test_detect_pdf(self, tmp_path):
        """Should detect PDF MIME type."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        result = gemini_query.detect_mime_type(pdf_file)

        assert result == "application/pdf"

    def test_detect_jpeg(self, tmp_path):
        """Should detect JPEG MIME type."""
        jpg_file = tmp_path / "test.jpg"
        jpg_file.write_bytes(b"\xff\xd8\xff")

        result = gemini_query.detect_mime_type(jpg_file)

        assert result == "image/jpeg"

    def test_detect_png(self, tmp_path):
        """Should detect PNG MIME type."""
        png_file = tmp_path / "test.png"
        png_file.write_bytes(b"\x89PNG")

        result = gemini_query.detect_mime_type(png_file)

        assert result == "image/png"

    def test_detect_mp3(self, tmp_path):
        """Should detect MP3 MIME type."""
        mp3_file = tmp_path / "test.mp3"
        mp3_file.write_bytes(b"ID3")

        result = gemini_query.detect_mime_type(mp3_file)

        assert result in ("audio/mp3", "audio/mpeg")

    def test_detect_wav(self, tmp_path):
        """Should detect WAV MIME type."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF")

        result = gemini_query.detect_mime_type(wav_file)

        assert result in ("audio/wav", "audio/x-wav")

    def test_unsupported_format(self, tmp_path):
        """Should raise ValueError for unsupported formats."""
        txt_file = tmp_path / "test.xyz"
        txt_file.write_text("content")

        with pytest.raises(ValueError) as exc_info:
            gemini_query.detect_mime_type(txt_file)

        assert "Unsupported" in str(exc_info.value)

    def test_unsupported_mime_type(self, tmp_path):
        """Should raise ValueError for unsupported MIME types."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("content")

        with pytest.raises(ValueError) as exc_info:
            gemini_query.detect_mime_type(txt_file)

        assert "Unsupported MIME type" in str(exc_info.value)


class TestQueryGemini:
    """Tests for query_gemini function."""

    def test_query_with_inline_data(self, tmp_path):
        """Should use inline data for small files."""
        # Create a small test file
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 100)

        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "This is a test image"
        mock_client.models.generate_content.return_value = mock_response

        result = gemini_query.query_gemini(
            client=mock_client,
            filepath=test_file,
            prompt="Describe this image"
        )

        assert result == "This is a test image"
        mock_client.models.generate_content.assert_called_once()

    def test_query_with_files_api(self, tmp_path):
        """Should use Files API when forced."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4" + b"x" * 100)

        mock_client = Mock()
        mock_uploaded = Mock()
        mock_client.files.upload.return_value = mock_uploaded
        mock_response = Mock()
        mock_response.text = "Document summary"
        mock_client.models.generate_content.return_value = mock_response

        result = gemini_query.query_gemini(
            client=mock_client,
            filepath=test_file,
            prompt="Summarize",
            use_files_api=True
        )

        assert result == "Document summary"
        mock_client.files.upload.assert_called_once()


class TestQueryGeminiStream:
    """Tests for query_gemini_stream function."""

    def test_stream_response(self, tmp_path):
        """Should yield chunks from streaming response."""
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 100)

        mock_client = Mock()

        # Create mock chunks
        chunk1 = Mock()
        chunk1.text = "Hello "
        chunk2 = Mock()
        chunk2.text = "world!"

        mock_client.models.generate_content_stream.return_value = [chunk1, chunk2]

        result = list(gemini_query.query_gemini_stream(
            client=mock_client,
            filepath=test_file,
            prompt="Describe"
        ))

        assert result == ["Hello ", "world!"]


class TestCreateParser:
    """Tests for argument parser."""

    def test_requires_file_and_prompt(self):
        """Should require --file and --prompt arguments."""
        parser = gemini_query.create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([])

        with pytest.raises(SystemExit):
            parser.parse_args(["-f", "test.pdf"])

        with pytest.raises(SystemExit):
            parser.parse_args(["-p", "prompt"])

    def test_accepts_short_flags(self):
        """Should accept -f and -p short flags."""
        parser = gemini_query.create_parser()
        args = parser.parse_args(["-f", "test.pdf", "-p", "Summarize"])

        assert args.file == Path("test.pdf")
        assert args.prompt == "Summarize"

    def test_accepts_long_flags(self):
        """Should accept --file and --prompt long flags."""
        parser = gemini_query.create_parser()
        args = parser.parse_args(["--file", "test.pdf", "--prompt", "Summarize"])

        assert args.file == Path("test.pdf")
        assert args.prompt == "Summarize"

    def test_model_flag(self):
        """Should parse --model flag."""
        parser = gemini_query.create_parser()
        args = parser.parse_args([
            "-f", "test.pdf", "-p", "Sum", "-m", "gemini-2.5-pro"
        ])

        assert args.model == "gemini-2.5-pro"

    def test_no_stream_flag(self):
        """Should parse --no-stream flag."""
        parser = gemini_query.create_parser()
        args = parser.parse_args(["-f", "test.pdf", "-p", "Sum", "--no-stream"])

        assert args.no_stream is True

    def test_json_flag(self):
        """Should parse --json flag."""
        parser = gemini_query.create_parser()
        args = parser.parse_args(["-f", "test.pdf", "-p", "Sum", "--json"])

        assert args.json is True

    def test_use_files_api_flag(self):
        """Should parse --use-files-api flag."""
        parser = gemini_query.create_parser()
        args = parser.parse_args(["-f", "test.pdf", "-p", "Sum", "--use-files-api"])

        assert args.use_files_api is True

    def test_default_model(self):
        """Should use default model."""
        parser = gemini_query.create_parser()
        args = parser.parse_args(["-f", "test.pdf", "-p", "Sum"])

        assert args.model == gemini_query.DEFAULT_MODEL


class TestMain:
    """Tests for main function."""

    def test_file_not_found(self, monkeypatch, capsys):
        """Should return error code 2 for missing file."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setattr(sys, "argv", [
            "gemini_query", "-f", "/nonexistent/file.pdf", "-p", "Summarize"
        ])

        result = gemini_query.main()

        assert result == 2
        captured = capsys.readouterr()
        assert "File not found" in captured.err

    def test_unsupported_format(self, monkeypatch, capsys, tmp_path):
        """Should return error code 2 for unsupported format."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setattr(sys, "argv", [
            "gemini_query", "-f", str(test_file), "-p", "Summarize"
        ])

        result = gemini_query.main()

        assert result == 2
        captured = capsys.readouterr()
        assert "Unsupported" in captured.err

    def test_successful_query_no_stream(self, monkeypatch, capsys, tmp_path):
        """Should return 0 on successful query."""
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 100)

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setattr(sys, "argv", [
            "gemini_query", "-f", str(test_file), "-p", "Describe", "--no-stream"
        ])

        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "A test image"
        mock_client.models.generate_content.return_value = mock_response

        with patch("gemini_query.genai.Client", return_value=mock_client):
            result = gemini_query.main()

        assert result == 0
        captured = capsys.readouterr()
        assert "A test image" in captured.out

    def test_json_flag_modifies_prompt(self, monkeypatch, tmp_path):
        """Should add JSON instruction to prompt when --json is used."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4" + b"x" * 100)

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setattr(sys, "argv", [
            "gemini_query", "-f", str(test_file), "-p", "Extract data", "--json", "--no-stream"
        ])

        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = '{"key": "value"}'
        mock_client.models.generate_content.return_value = mock_response

        with patch("gemini_query.genai.Client", return_value=mock_client):
            gemini_query.main()

        # Check that the prompt was modified
        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs["contents"]
        prompt = contents[1]
        assert "JSON" in prompt


class TestSupportedFormats:
    """Tests for supported format constants."""

    def test_supported_images(self):
        """Should include common image formats."""
        assert "image/jpeg" in gemini_query.SUPPORTED_IMAGES
        assert "image/png" in gemini_query.SUPPORTED_IMAGES
        assert "image/gif" in gemini_query.SUPPORTED_IMAGES
        assert "image/webp" in gemini_query.SUPPORTED_IMAGES

    def test_supported_audio(self):
        """Should include common audio formats."""
        assert "audio/mp3" in gemini_query.SUPPORTED_AUDIO
        assert "audio/wav" in gemini_query.SUPPORTED_AUDIO
        assert "audio/flac" in gemini_query.SUPPORTED_AUDIO

    def test_supported_documents(self):
        """Should include PDF format."""
        assert "application/pdf" in gemini_query.SUPPORTED_DOCUMENTS

    def test_all_supported_combined(self):
        """Should combine all supported types."""
        all_types = gemini_query.ALL_SUPPORTED
        assert "image/png" in all_types
        assert "audio/mp3" in all_types
        assert "application/pdf" in all_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
