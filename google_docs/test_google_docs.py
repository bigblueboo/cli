#!/usr/bin/env python3
"""Tests for google_docs CLI tool."""

import json
import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
import google_docs


class TestGetCredentials:
    """Tests for get_credentials function."""

    def test_exits_when_env_not_set(self, monkeypatch):
        """Should exit with code 1 when credentials path not set."""
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            google_docs.get_credentials()

        assert exc_info.value.code == 1

    def test_exits_when_file_not_found(self, monkeypatch, tmp_path):
        """Should exit with code 1 when credentials file doesn't exist."""
        monkeypatch.setenv(
            "GOOGLE_APPLICATION_CREDENTIALS",
            str(tmp_path / "nonexistent.json")
        )

        with pytest.raises(SystemExit) as exc_info:
            google_docs.get_credentials()

        assert exc_info.value.code == 1

    def test_loads_valid_credentials(self, monkeypatch, tmp_path):
        """Should load credentials from valid service account file."""
        # Create a mock service account file
        creds_file = tmp_path / "service-account.json"
        creds_data = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "key-id",
            "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA2Z3qX2BTLS4e0v/2\n-----END RSA PRIVATE KEY-----\n",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "123456789",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token"
        }
        creds_file.write_text(json.dumps(creds_data))

        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds_file))

        # Mock the credentials loading
        mock_creds = Mock()
        with patch("google_docs.service_account.Credentials.from_service_account_file",
                   return_value=mock_creds):
            result = google_docs.get_credentials()

        assert result == mock_creds


class TestDocsOperations:
    """Tests for Google Docs operations."""

    def test_docs_read(self):
        """Should extract text from document."""
        mock_credentials = Mock()
        mock_service = Mock()
        mock_documents = Mock()

        # Mock document structure
        mock_doc = {
            "body": {
                "content": [
                    {
                        "paragraph": {
                            "elements": [
                                {"textRun": {"content": "Hello "}},
                                {"textRun": {"content": "World!"}}
                            ]
                        }
                    }
                ]
            }
        }

        mock_service.documents.return_value.get.return_value.execute.return_value = mock_doc

        with patch("google_docs.build", return_value=mock_service):
            result = google_docs.docs_read("doc-id-123", mock_credentials)

        assert result == "Hello World!"

    def test_docs_append(self):
        """Should append text to document."""
        mock_credentials = Mock()
        mock_service = Mock()

        # Mock document for getting end index
        mock_doc = {
            "body": {
                "content": [
                    {"endIndex": 10}
                ]
            }
        }
        mock_service.documents.return_value.get.return_value.execute.return_value = mock_doc

        # Mock batch update response
        mock_result = {"replies": []}
        mock_service.documents.return_value.batchUpdate.return_value.execute.return_value = mock_result

        with patch("google_docs.build", return_value=mock_service):
            result = google_docs.docs_append("doc-id-123", "New text", mock_credentials)

        assert result == mock_result
        mock_service.documents.return_value.batchUpdate.assert_called_once()

    def test_docs_create(self):
        """Should create new document."""
        mock_credentials = Mock()
        mock_service = Mock()

        mock_doc = {"documentId": "new-doc-id"}
        mock_service.documents.return_value.create.return_value.execute.return_value = mock_doc

        with patch("google_docs.build", return_value=mock_service):
            result = google_docs.docs_create("Test Document", mock_credentials)

        assert result["documentId"] == "new-doc-id"
        assert "url" in result


class TestSheetsOperations:
    """Tests for Google Sheets operations."""

    def test_sheets_read(self):
        """Should read spreadsheet data."""
        mock_credentials = Mock()
        mock_service = Mock()

        mock_data = {
            "values": [
                ["A1", "B1", "C1"],
                ["A2", "B2", "C2"]
            ]
        }
        mock_service.spreadsheets.return_value.values.return_value.get.return_value.execute.return_value = mock_data

        with patch("google_docs.build", return_value=mock_service):
            result = google_docs.sheets_read("sheet-id-123", mock_credentials)

        assert len(result) == 2
        assert result[0] == ["A1", "B1", "C1"]

    def test_sheets_read_with_range(self):
        """Should read specific range."""
        mock_credentials = Mock()
        mock_service = Mock()

        mock_data = {"values": [["X"]]}
        mock_service.spreadsheets.return_value.values.return_value.get.return_value.execute.return_value = mock_data

        with patch("google_docs.build", return_value=mock_service):
            result = google_docs.sheets_read(
                "sheet-id", mock_credentials, range_notation="Sheet1!A1"
            )

        # Verify the range was passed
        mock_service.spreadsheets.return_value.values.return_value.get.assert_called_once()

    def test_sheets_append(self):
        """Should append rows to spreadsheet."""
        mock_credentials = Mock()
        mock_service = Mock()

        mock_result = {
            "updates": {
                "updatedRange": "Sheet1!A3:B4",
                "updatedRows": 2,
                "updatedCells": 4
            }
        }
        mock_service.spreadsheets.return_value.values.return_value.append.return_value.execute.return_value = mock_result

        with patch("google_docs.build", return_value=mock_service):
            result = google_docs.sheets_append(
                "sheet-id",
                [["a", "b"], ["c", "d"]],
                mock_credentials
            )

        assert result["updatedRows"] == 2
        assert result["updatedCells"] == 4

    def test_sheets_write(self):
        """Should write to specific cells."""
        mock_credentials = Mock()
        mock_service = Mock()

        mock_result = {
            "updatedRange": "Sheet1!B2:C3",
            "updatedRows": 2,
            "updatedCells": 4
        }
        mock_service.spreadsheets.return_value.values.return_value.update.return_value.execute.return_value = mock_result

        with patch("google_docs.build", return_value=mock_service):
            result = google_docs.sheets_write(
                "sheet-id",
                [["x", "y"], ["z", "w"]],
                mock_credentials,
                range_notation="B2"
            )

        assert result["updatedRange"] == "Sheet1!B2:C3"

    def test_sheets_create(self):
        """Should create new spreadsheet."""
        mock_credentials = Mock()
        mock_service = Mock()

        mock_result = {
            "spreadsheetId": "new-sheet-id",
            "spreadsheetUrl": "https://docs.google.com/spreadsheets/d/new-sheet-id"
        }
        mock_service.spreadsheets.return_value.create.return_value.execute.return_value = mock_result

        with patch("google_docs.build", return_value=mock_service):
            result = google_docs.sheets_create("Test Sheet", mock_credentials)

        assert result["spreadsheetId"] == "new-sheet-id"
        assert "url" in result

    def test_sheets_create_with_sheet_names(self):
        """Should create spreadsheet with custom sheet names."""
        mock_credentials = Mock()
        mock_service = Mock()

        mock_result = {
            "spreadsheetId": "new-sheet-id",
            "spreadsheetUrl": "https://docs.google.com/spreadsheets/d/new-sheet-id"
        }
        mock_service.spreadsheets.return_value.create.return_value.execute.return_value = mock_result

        with patch("google_docs.build", return_value=mock_service):
            result = google_docs.sheets_create(
                "Test Sheet",
                mock_credentials,
                sheet_names=["Data", "Summary"]
            )

        # Verify create was called with sheet names
        call_args = mock_service.spreadsheets.return_value.create.call_args
        body = call_args.kwargs["body"]
        assert "sheets" in body
        assert len(body["sheets"]) == 2

    def test_sheets_clear(self):
        """Should clear cells in spreadsheet."""
        mock_credentials = Mock()
        mock_service = Mock()

        mock_result = {"clearedRange": "Sheet1!A1:Z100"}
        mock_service.spreadsheets.return_value.values.return_value.clear.return_value.execute.return_value = mock_result

        with patch("google_docs.build", return_value=mock_service):
            result = google_docs.sheets_clear("sheet-id", mock_credentials, "A1:Z100")

        assert result["clearedRange"] == "Sheet1!A1:Z100"


class TestCreateParser:
    """Tests for argument parser."""

    def test_requires_service_subcommand(self):
        """Should require doc or sheet subcommand."""
        parser = google_docs.create_parser()
        args = parser.parse_args([])
        assert args.service is None

    def test_doc_read_command(self):
        """Should parse doc read command."""
        parser = google_docs.create_parser()
        args = parser.parse_args(["doc", "read", "doc-id-123"])

        assert args.service == "doc"
        assert args.action == "read"
        assert args.doc_id == "doc-id-123"

    def test_doc_append_command(self):
        """Should parse doc append command."""
        parser = google_docs.create_parser()
        args = parser.parse_args(["doc", "append", "doc-id", "New text"])

        assert args.service == "doc"
        assert args.action == "append"
        assert args.doc_id == "doc-id"
        assert args.text == "New text"

    def test_doc_create_command(self):
        """Should parse doc create command."""
        parser = google_docs.create_parser()
        args = parser.parse_args(["doc", "create", "My Document"])

        assert args.service == "doc"
        assert args.action == "create"
        assert args.title == "My Document"

    def test_doc_create_with_content(self):
        """Should parse doc create with initial content."""
        parser = google_docs.create_parser()
        args = parser.parse_args([
            "doc", "create", "My Doc", "--content", "Initial text"
        ])

        assert args.content == "Initial text"

    def test_sheet_read_command(self):
        """Should parse sheet read command."""
        parser = google_docs.create_parser()
        args = parser.parse_args(["sheet", "read", "sheet-id"])

        assert args.service == "sheet"
        assert args.action == "read"
        assert args.sheet_id == "sheet-id"

    def test_sheet_read_with_range(self):
        """Should parse sheet read with range."""
        parser = google_docs.create_parser()
        args = parser.parse_args([
            "sheet", "read", "sheet-id", "--range", "A1:D10"
        ])

        assert args.range == "A1:D10"

    def test_sheet_read_json_flag(self):
        """Should parse --json flag for sheet read."""
        parser = google_docs.create_parser()
        args = parser.parse_args(["sheet", "read", "sheet-id", "--json"])

        assert args.json is True

    def test_sheet_append_command(self):
        """Should parse sheet append command."""
        parser = google_docs.create_parser()
        data = '[["a","b"],["c","d"]]'
        args = parser.parse_args([
            "sheet", "append", "sheet-id", "--data", data
        ])

        assert args.service == "sheet"
        assert args.action == "append"
        assert args.data == data

    def test_sheet_write_command(self):
        """Should parse sheet write command."""
        parser = google_docs.create_parser()
        args = parser.parse_args([
            "sheet", "write", "sheet-id", "--range", "B2", "--data", '[["x"]]'
        ])

        assert args.action == "write"
        assert args.range == "B2"

    def test_sheet_create_command(self):
        """Should parse sheet create command."""
        parser = google_docs.create_parser()
        args = parser.parse_args(["sheet", "create", "My Spreadsheet"])

        assert args.action == "create"
        assert args.title == "My Spreadsheet"

    def test_sheet_create_with_sheets(self):
        """Should parse sheet create with sheet names."""
        parser = google_docs.create_parser()
        args = parser.parse_args([
            "sheet", "create", "My Sheet", "--sheets", "Data,Summary,Charts"
        ])

        assert args.sheets == "Data,Summary,Charts"

    def test_sheet_clear_command(self):
        """Should parse sheet clear command."""
        parser = google_docs.create_parser()
        args = parser.parse_args([
            "sheet", "clear", "sheet-id", "--range", "A1:Z100"
        ])

        assert args.action == "clear"
        assert args.range == "A1:Z100"


class TestMain:
    """Tests for main function."""

    def test_no_service_shows_help(self, monkeypatch, capsys):
        """Should return 4 when no service specified."""
        monkeypatch.setattr(sys, "argv", ["google_docs"])

        result = google_docs.main()

        assert result == 4

    def test_doc_read_success(self, monkeypatch, capsys, tmp_path):
        """Should read document and print content."""
        # Setup credentials
        creds_file = tmp_path / "creds.json"
        creds_file.write_text('{"type": "service_account"}')
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds_file))
        monkeypatch.setattr(sys, "argv", ["google_docs", "doc", "read", "doc-id"])

        mock_creds = Mock()
        mock_service = Mock()
        mock_doc = {
            "body": {
                "content": [
                    {"paragraph": {"elements": [{"textRun": {"content": "Doc content"}}]}}
                ]
            }
        }
        mock_service.documents.return_value.get.return_value.execute.return_value = mock_doc

        with patch("google_docs.service_account.Credentials.from_service_account_file",
                   return_value=mock_creds):
            with patch("google_docs.build", return_value=mock_service):
                result = google_docs.main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Doc content" in captured.out

    def test_sheet_read_json_output(self, monkeypatch, capsys, tmp_path):
        """Should output JSON when --json flag is used."""
        creds_file = tmp_path / "creds.json"
        creds_file.write_text('{"type": "service_account"}')
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds_file))
        monkeypatch.setattr(sys, "argv", [
            "google_docs", "sheet", "read", "sheet-id", "--json"
        ])

        mock_creds = Mock()
        mock_service = Mock()
        mock_data = {"values": [["a", "b"], ["c", "d"]]}
        mock_service.spreadsheets.return_value.values.return_value.get.return_value.execute.return_value = mock_data

        with patch("google_docs.service_account.Credentials.from_service_account_file",
                   return_value=mock_creds):
            with patch("google_docs.build", return_value=mock_service):
                result = google_docs.main()

        assert result == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == [["a", "b"], ["c", "d"]]

    def test_handles_http_404_error(self, monkeypatch, capsys, tmp_path):
        """Should return 2 for document not found."""
        from googleapiclient.errors import HttpError

        creds_file = tmp_path / "creds.json"
        creds_file.write_text('{"type": "service_account"}')
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds_file))
        monkeypatch.setattr(sys, "argv", ["google_docs", "doc", "read", "bad-id"])

        mock_creds = Mock()
        mock_service = Mock()

        # Create mock HTTP error
        mock_resp = Mock()
        mock_resp.status = 404
        http_error = HttpError(mock_resp, b"Not Found")

        mock_service.documents.return_value.get.return_value.execute.side_effect = http_error

        with patch("google_docs.service_account.Credentials.from_service_account_file",
                   return_value=mock_creds):
            with patch("google_docs.build", return_value=mock_service):
                result = google_docs.main()

        assert result == 2

    def test_handles_invalid_json_data(self, monkeypatch, capsys, tmp_path):
        """Should return 4 for invalid JSON in --data."""
        creds_file = tmp_path / "creds.json"
        creds_file.write_text('{"type": "service_account"}')
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds_file))
        monkeypatch.setattr(sys, "argv", [
            "google_docs", "sheet", "append", "sheet-id", "--data", "not valid json"
        ])

        mock_creds = Mock()

        with patch("google_docs.service_account.Credentials.from_service_account_file",
                   return_value=mock_creds):
            with patch("google_docs.build"):
                result = google_docs.main()

        assert result == 4
        captured = capsys.readouterr()
        assert "Invalid JSON" in captured.err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
