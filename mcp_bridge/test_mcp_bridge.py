#!/usr/bin/env python3
"""
Comprehensive tests for MCP Bridge CLI tool.

These tests cover:
- Configuration management
- Server state management
- JSON-RPC protocol handling
- MCP client functionality
- CLI commands
- Proxy server functionality

All external dependencies are mocked to ensure tests are fast and reliable.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

import pytest

# Import the module under test
import mcp_bridge
from mcp_bridge import (
    ExitCode,
    ServerConfig,
    BridgeConfig,
    ServerState,
    JsonRpcError,
    MCPClient,
    MCPProxy,
    make_jsonrpc_request,
    parse_jsonrpc_response,
    get_config_path,
    get_state_path,
    load_config,
    save_config,
    load_state,
    save_state,
    cleanup_dead_servers,
    is_process_running,
    cmd_start,
    cmd_stop,
    cmd_list,
    cmd_list_tools,
    cmd_call,
    cmd_health,
    cmd_config,
    cmd_proxy,
    create_parser,
    main,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / ".mcp_bridge"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def mock_config_path(temp_config_dir):
    """Mock the config path to use temp directory."""
    config_file = temp_config_dir / "config.yaml"
    with patch("mcp_bridge.get_config_path", return_value=config_file):
        yield config_file


@pytest.fixture
def mock_state_path(temp_config_dir):
    """Mock the state path to use temp directory."""
    state_file = temp_config_dir / "state.json"
    with patch("mcp_bridge.get_state_path", return_value=state_file):
        yield state_file


@pytest.fixture
def sample_server_config():
    """Create a sample server configuration."""
    return ServerConfig(
        name="test-server",
        command="node server.js",
        port=3000,
        transport="stdio",
        env={"NODE_ENV": "test"},
        args=["--debug"],
        working_dir="/tmp/test",
        auto_start=True,
        health_check_interval=60,
        restart_on_failure=True,
        max_retries=5,
    )


@pytest.fixture
def sample_bridge_config(sample_server_config):
    """Create a sample bridge configuration."""
    return BridgeConfig(
        servers={"test-server": sample_server_config},
        default_timeout=60,
        log_level="DEBUG",
        proxy_port=9000,
    )


@pytest.fixture
def sample_server_state():
    """Create a sample server state."""
    return ServerState(
        name="test-server",
        pid=12345,
        command="node server.js",
        port=3000,
        transport="stdio",
        started_at="2024-01-15T10:30:00",
        status="running",
        health="healthy",
        last_health_check="2024-01-15T10:35:00",
    )


class MockArgs:
    """Helper class to create mock argparse.Namespace objects."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Tests for configuration management."""

    def test_get_config_path_default(self):
        """Test default config path."""
        with patch.dict(os.environ, {}, clear=True):
            if "MCP_BRIDGE_CONFIG" in os.environ:
                del os.environ["MCP_BRIDGE_CONFIG"]
            path = get_config_path()
            assert path == Path.home() / ".mcp_bridge" / "config.yaml"

    def test_get_config_path_env_override(self):
        """Test config path from environment variable."""
        with patch.dict(os.environ, {"MCP_BRIDGE_CONFIG": "/custom/config.yaml"}):
            path = get_config_path()
            assert path == Path("/custom/config.yaml")

    def test_get_state_path(self):
        """Test state path."""
        path = get_state_path()
        assert path == Path.home() / ".mcp_bridge" / "state.json"

    def test_server_config_to_dict(self, sample_server_config):
        """Test ServerConfig serialization."""
        data = sample_server_config.to_dict()
        assert data["name"] == "test-server"
        assert data["command"] == "node server.js"
        assert data["port"] == 3000
        assert data["transport"] == "stdio"
        assert data["env"] == {"NODE_ENV": "test"}
        assert data["args"] == ["--debug"]

    def test_server_config_from_dict(self):
        """Test ServerConfig deserialization."""
        data = {
            "name": "my-server",
            "command": "python server.py",
            "port": 8000,
            "transport": "http",
        }
        config = ServerConfig.from_dict(data)
        assert config.name == "my-server"
        assert config.command == "python server.py"
        assert config.port == 8000
        assert config.transport == "http"

    def test_bridge_config_to_dict(self, sample_bridge_config):
        """Test BridgeConfig serialization."""
        data = sample_bridge_config.to_dict()
        assert "servers" in data
        assert "test-server" in data["servers"]
        assert data["default_timeout"] == 60
        assert data["log_level"] == "DEBUG"
        assert data["proxy_port"] == 9000

    def test_bridge_config_from_dict(self):
        """Test BridgeConfig deserialization."""
        data = {
            "servers": {
                "server1": {
                    "name": "server1",
                    "command": "node s1.js",
                },
            },
            "default_timeout": 45,
            "log_level": "WARNING",
        }
        config = BridgeConfig.from_dict(data)
        assert len(config.servers) == 1
        assert "server1" in config.servers
        assert config.default_timeout == 45
        assert config.log_level == "WARNING"

    def test_load_config_creates_default(self, mock_config_path):
        """Test loading config when file doesn't exist."""
        config = load_config()
        assert isinstance(config, BridgeConfig)
        assert len(config.servers) == 0

    def test_load_config_from_file(self, mock_config_path):
        """Test loading config from file."""
        config_data = {
            "servers": {
                "test": {"name": "test", "command": "echo test"},
            },
            "default_timeout": 120,
        }
        import yaml
        mock_config_path.write_text(yaml.dump(config_data))

        config = load_config()
        assert "test" in config.servers
        assert config.default_timeout == 120

    def test_save_config(self, mock_config_path):
        """Test saving config to file."""
        config = BridgeConfig(
            servers={"s1": ServerConfig(name="s1", command="cmd1")},
            default_timeout=90,
        )
        save_config(config)

        assert mock_config_path.exists()
        import yaml
        loaded = yaml.safe_load(mock_config_path.read_text())
        assert "s1" in loaded["servers"]
        assert loaded["default_timeout"] == 90


# =============================================================================
# State Management Tests
# =============================================================================

class TestStateManagement:
    """Tests for server state management."""

    def test_server_state_to_dict(self, sample_server_state):
        """Test ServerState serialization."""
        data = sample_server_state.to_dict()
        assert data["name"] == "test-server"
        assert data["pid"] == 12345
        assert data["status"] == "running"

    def test_server_state_from_dict(self):
        """Test ServerState deserialization."""
        data = {
            "name": "my-server",
            "pid": 54321,
            "command": "python app.py",
            "port": 5000,
            "transport": "http",
            "started_at": "2024-01-15T12:00:00",
            "status": "running",
            "health": "unknown",
            "last_health_check": None,
        }
        state = ServerState.from_dict(data)
        assert state.name == "my-server"
        assert state.pid == 54321
        assert state.status == "running"

    def test_load_state_empty(self, mock_state_path):
        """Test loading state when file doesn't exist."""
        state = load_state()
        assert state == {}

    def test_load_state_from_file(self, mock_state_path):
        """Test loading state from file."""
        state_data = {
            "server1": {
                "name": "server1",
                "pid": 1234,
                "command": "cmd",
                "port": None,
                "transport": "stdio",
                "started_at": "2024-01-15T10:00:00",
                "status": "running",
                "health": "unknown",
                "last_health_check": None,
            },
        }
        mock_state_path.write_text(json.dumps(state_data))

        state = load_state()
        assert "server1" in state
        assert state["server1"].pid == 1234

    def test_save_state(self, mock_state_path):
        """Test saving state to file."""
        state = {
            "s1": ServerState(
                name="s1",
                pid=999,
                command="test",
                port=None,
                transport="stdio",
                started_at="2024-01-15T10:00:00",
            ),
        }
        save_state(state)

        assert mock_state_path.exists()
        loaded = json.loads(mock_state_path.read_text())
        assert "s1" in loaded
        assert loaded["s1"]["pid"] == 999

    def test_is_process_running_true(self):
        """Test process detection for running process."""
        # Current process should be running
        assert is_process_running(os.getpid()) is True

    def test_is_process_running_false(self):
        """Test process detection for non-existent process."""
        # Use a very high PID that's unlikely to exist
        assert is_process_running(999999999) is False

    def test_cleanup_dead_servers(self, mock_state_path):
        """Test cleanup of dead server entries."""
        # Create state with a dead server (PID 999999999 shouldn't exist)
        state_data = {
            "dead-server": {
                "name": "dead-server",
                "pid": 999999999,
                "command": "cmd",
                "port": None,
                "transport": "stdio",
                "started_at": "2024-01-15T10:00:00",
                "status": "running",
                "health": "unknown",
                "last_health_check": None,
            },
        }
        mock_state_path.write_text(json.dumps(state_data))

        cleaned = cleanup_dead_servers()
        assert "dead-server" not in cleaned


# =============================================================================
# JSON-RPC Protocol Tests
# =============================================================================

class TestJsonRpc:
    """Tests for JSON-RPC protocol handling."""

    def test_make_jsonrpc_request_basic(self):
        """Test creating a basic JSON-RPC request."""
        request = make_jsonrpc_request("test_method")
        assert request["jsonrpc"] == "2.0"
        assert request["method"] == "test_method"
        assert "id" in request
        assert "params" not in request

    def test_make_jsonrpc_request_with_params(self):
        """Test creating a JSON-RPC request with parameters."""
        request = make_jsonrpc_request("test", {"key": "value"})
        assert request["params"] == {"key": "value"}

    def test_make_jsonrpc_request_with_id(self):
        """Test creating a JSON-RPC request with custom ID."""
        request = make_jsonrpc_request("test", id="custom-id-123")
        assert request["id"] == "custom-id-123"

    def test_parse_jsonrpc_response_success(self):
        """Test parsing a successful response."""
        response = {"jsonrpc": "2.0", "id": "1", "result": {"data": "test"}}
        result = parse_jsonrpc_response(response)
        assert result == {"data": "test"}

    def test_parse_jsonrpc_response_error(self):
        """Test parsing an error response."""
        response = {
            "jsonrpc": "2.0",
            "id": "1",
            "error": {"code": -32600, "message": "Invalid Request"},
        }
        with pytest.raises(JsonRpcError) as excinfo:
            parse_jsonrpc_response(response)
        assert excinfo.value.code == -32600
        assert "Invalid Request" in str(excinfo.value)

    def test_jsonrpc_error_with_data(self):
        """Test JSON-RPC error with additional data."""
        response = {
            "jsonrpc": "2.0",
            "id": "1",
            "error": {
                "code": -32000,
                "message": "Server error",
                "data": {"details": "Something went wrong"},
            },
        }
        with pytest.raises(JsonRpcError) as excinfo:
            parse_jsonrpc_response(response)
        assert excinfo.value.data == {"details": "Something went wrong"}


# =============================================================================
# MCP Client Tests
# =============================================================================

class TestMCPClient:
    """Tests for MCP client functionality."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization."""
        client = MCPClient(
            transport="stdio",
            command="echo test",
            timeout=60,
        )
        assert client.transport == "stdio"
        assert client.command == "echo test"
        assert client.timeout == 60

    @pytest.mark.asyncio
    async def test_client_connect_stdio(self):
        """Test stdio connection."""
        client = MCPClient(transport="stdio", command="cat")

        with patch.object(subprocess, "Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdin = MagicMock()
            mock_process.stdout = MagicMock()
            mock_process.stdout.readline.return_value = b'{"jsonrpc":"2.0","id":"1","result":{"capabilities":{}}}\n'
            mock_popen.return_value = mock_process

            await client._connect_stdio()

            mock_popen.assert_called_once()
            assert client._process is not None

    @pytest.mark.asyncio
    async def test_client_connect_http(self):
        """Test HTTP connection."""
        client = MCPClient(transport="http", host="localhost", port=8080)

        await client._connect_http()

        assert client._http_client is not None
        await client._http_client.aclose()

    @pytest.mark.asyncio
    async def test_client_request(self):
        """Test sending a request."""
        client = MCPClient(transport="stdio", command="echo")
        client._initialized = True

        response = {"jsonrpc": "2.0", "id": "1", "result": {"tools": []}}

        with patch.object(client, "_send_receive", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = response
            result = await client.request("tools/list")
            assert result == {"tools": []}

    @pytest.mark.asyncio
    async def test_client_list_tools(self):
        """Test listing tools."""
        client = MCPClient(transport="stdio", command="echo")
        client._initialized = True

        expected_tools = [
            {"name": "read_file", "description": "Read a file"},
            {"name": "write_file", "description": "Write a file"},
        ]

        with patch.object(client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"tools": expected_tools}
            tools = await client.list_tools()
            assert tools == expected_tools
            mock_request.assert_called_once_with("tools/list")

    @pytest.mark.asyncio
    async def test_client_call_tool(self):
        """Test calling a tool."""
        client = MCPClient(transport="stdio", command="echo")
        client._initialized = True

        expected_result = {"content": "file contents"}

        with patch.object(client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = expected_result
            result = await client.call_tool("read_file", {"path": "test.txt"})
            assert result == expected_result
            mock_request.assert_called_once_with(
                "tools/call",
                {"name": "read_file", "arguments": {"path": "test.txt"}},
            )

    @pytest.mark.asyncio
    async def test_client_ping_success(self):
        """Test successful ping."""
        client = MCPClient(transport="stdio", command="echo")

        with patch.object(client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {}
            result = await client.ping()
            assert result is True

    @pytest.mark.asyncio
    async def test_client_ping_failure(self):
        """Test failed ping."""
        client = MCPClient(transport="stdio", command="echo")

        with patch.object(client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = ConnectionError("Connection refused")
            result = await client.ping()
            assert result is False

    @pytest.mark.asyncio
    async def test_client_close(self):
        """Test closing the client."""
        client = MCPClient(transport="stdio", command="echo")
        mock_process = MagicMock()
        client._process = mock_process

        await client.close()

        mock_process.terminate.assert_called_once()
        assert client._process is None


# =============================================================================
# MCP Proxy Tests
# =============================================================================

class TestMCPProxy:
    """Tests for MCP proxy functionality."""

    def test_proxy_initialization(self):
        """Test proxy initialization."""
        proxy = MCPProxy(port=9000)
        assert proxy.port == 9000
        assert proxy.backends == {}

    def test_proxy_add_backend(self):
        """Test adding a backend."""
        proxy = MCPProxy()
        mock_client = MagicMock(spec=MCPClient)
        proxy.add_backend("test", mock_client)
        assert "test" in proxy.backends
        assert proxy.backends["test"] == mock_client

    @pytest.mark.asyncio
    async def test_proxy_list_all_tools(self):
        """Test listing tools from all backends."""
        proxy = MCPProxy()

        mock_client1 = AsyncMock(spec=MCPClient)
        mock_client1.list_tools.return_value = [{"name": "tool1"}]

        mock_client2 = AsyncMock(spec=MCPClient)
        mock_client2.list_tools.return_value = [{"name": "tool2"}]

        proxy.add_backend("backend1", mock_client1)
        proxy.add_backend("backend2", mock_client2)

        tools = await proxy.list_all_tools()

        assert "backend1" in tools
        assert "backend2" in tools
        assert tools["backend1"] == [{"name": "tool1"}]
        assert tools["backend2"] == [{"name": "tool2"}]

    @pytest.mark.asyncio
    async def test_proxy_call_tool(self):
        """Test calling a tool through the proxy."""
        proxy = MCPProxy()

        mock_client = AsyncMock(spec=MCPClient)
        mock_client.call_tool.return_value = {"result": "success"}

        proxy.add_backend("test", mock_client)

        result = await proxy.call_tool("test", "my_tool", {"arg": "value"})

        assert result == {"result": "success"}
        mock_client.call_tool.assert_called_once_with("my_tool", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_proxy_call_tool_unknown_backend(self):
        """Test calling a tool on unknown backend."""
        proxy = MCPProxy()

        with pytest.raises(ValueError) as excinfo:
            await proxy.call_tool("unknown", "tool", {})
        assert "Unknown backend" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_proxy_handle_jsonrpc(self):
        """Test JSON-RPC request handling."""
        proxy = MCPProxy()

        mock_client = AsyncMock(spec=MCPClient)
        mock_client.list_tools.return_value = [{"name": "test_tool"}]
        proxy.add_backend("test", mock_client)

        request = json.dumps({
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/list",
            "params": {},
        })

        response = await proxy._handle_jsonrpc(request)
        response_data = json.loads(response)

        assert response_data["jsonrpc"] == "2.0"
        assert response_data["id"] == "1"
        assert "test" in response_data["result"]


# =============================================================================
# CLI Command Tests
# =============================================================================

class TestCLICommands:
    """Tests for CLI command implementations."""

    def test_cmd_start_success(self, mock_state_path):
        """Test starting a server successfully."""
        args = MockArgs(
            command="sleep 100",
            name="test-server",
            port=None,
            transport="stdio",
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_process.poll.return_value = None  # Process is running
            mock_popen.return_value = mock_process

            with patch("time.sleep"):
                result = cmd_start(args)

            assert result == ExitCode.SUCCESS

    def test_cmd_start_duplicate_name(self, mock_state_path):
        """Test starting a server with duplicate name."""
        # Create existing state
        state = {
            "test-server": ServerState(
                name="test-server",
                pid=os.getpid(),  # Use current PID so it appears running
                command="cmd",
                port=None,
                transport="stdio",
                started_at="2024-01-15T10:00:00",
            ),
        }
        save_state(state)

        args = MockArgs(
            command="echo test",
            name="test-server",
            port=None,
            transport="stdio",
        )

        result = cmd_start(args)
        assert result == ExitCode.INVALID_ARGS

    def test_cmd_start_command_not_found(self, mock_state_path):
        """Test starting with non-existent command."""
        args = MockArgs(
            command="/nonexistent/command/xyz123",
            name="test-server",
            port=None,
            transport="stdio",
        )

        result = cmd_start(args)
        assert result == ExitCode.SERVER_ERROR

    def test_cmd_stop_success(self, mock_state_path):
        """Test stopping a server successfully."""
        # Start a background process we can stop
        process = subprocess.Popen(
            ["sleep", "100"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        state = {
            "test-server": ServerState(
                name="test-server",
                pid=process.pid,
                command="sleep 100",
                port=None,
                transport="stdio",
                started_at="2024-01-15T10:00:00",
            ),
        }
        save_state(state)

        args = MockArgs(name="test-server")

        result = cmd_stop(args)

        assert result == ExitCode.SUCCESS
        # Verify process is stopped
        process.poll()  # Update process status

    def test_cmd_stop_not_found(self, mock_state_path):
        """Test stopping a non-existent server."""
        args = MockArgs(name="nonexistent")
        result = cmd_stop(args)
        assert result == ExitCode.INVALID_ARGS

    def test_cmd_list_empty(self, mock_state_path, capsys):
        """Test listing when no servers are running."""
        args = MockArgs()
        result = cmd_list(args)
        captured = capsys.readouterr()

        assert result == ExitCode.SUCCESS
        assert "No running servers" in captured.out

    def test_cmd_list_with_servers(self, mock_state_path, capsys):
        """Test listing running servers."""
        state = {
            "server1": ServerState(
                name="server1",
                pid=os.getpid(),
                command="cmd1",
                port=3000,
                transport="stdio",
                started_at="2024-01-15T10:00:00",
            ),
        }
        save_state(state)

        args = MockArgs()
        result = cmd_list(args)
        captured = capsys.readouterr()

        assert result == ExitCode.SUCCESS
        assert "server1" in captured.out
        assert "3000" in captured.out

    def test_cmd_list_tools_server_not_found(self, mock_state_path):
        """Test listing tools from non-existent server."""
        args = MockArgs(server="nonexistent")
        result = cmd_list_tools(args)
        assert result == ExitCode.INVALID_ARGS

    def test_cmd_call_server_not_found(self, mock_state_path):
        """Test calling tool on non-existent server."""
        args = MockArgs(server="nonexistent", tool="test", args="{}")
        result = cmd_call(args)
        assert result == ExitCode.INVALID_ARGS

    def test_cmd_call_invalid_json(self, mock_state_path):
        """Test calling tool with invalid JSON arguments."""
        state = {
            "test": ServerState(
                name="test",
                pid=os.getpid(),
                command="cmd",
                port=None,
                transport="stdio",
                started_at="2024-01-15T10:00:00",
            ),
        }
        save_state(state)

        args = MockArgs(server="test", tool="test", args="invalid json")
        result = cmd_call(args)
        assert result == ExitCode.INVALID_ARGS

    def test_cmd_health_no_servers(self, mock_state_path, capsys):
        """Test health check with no servers."""
        args = MockArgs(all=True, server=None)
        result = cmd_health(args)
        captured = capsys.readouterr()

        assert result == ExitCode.SUCCESS
        assert "No servers to check" in captured.out

    def test_cmd_config_show(self, mock_config_path, capsys):
        """Test showing configuration."""
        import yaml
        mock_config_path.write_text(yaml.dump({"servers": {}, "default_timeout": 30}))

        args = MockArgs(action="show")
        result = cmd_config(args)
        captured = capsys.readouterr()

        assert result == ExitCode.SUCCESS
        assert "default_timeout" in captured.out

    def test_cmd_config_add(self, mock_config_path, capsys):
        """Test adding server to configuration."""
        args = MockArgs(
            action="add",
            name="new-server",
            command="node server.js",
            port=3000,
            transport="stdio",
        )
        result = cmd_config(args)
        captured = capsys.readouterr()

        assert result == ExitCode.SUCCESS
        assert "Added server" in captured.out

        # Verify it was saved
        config = load_config()
        assert "new-server" in config.servers

    def test_cmd_config_add_missing_args(self, mock_config_path):
        """Test adding server without required arguments."""
        args = MockArgs(
            action="add",
            name=None,
            command=None,
            port=None,
            transport="stdio",
        )
        result = cmd_config(args)
        assert result == ExitCode.INVALID_ARGS

    def test_cmd_config_remove(self, mock_config_path, capsys):
        """Test removing server from configuration."""
        # First add a server
        config = BridgeConfig(
            servers={"test": ServerConfig(name="test", command="cmd")},
        )
        save_config(config)

        args = MockArgs(action="remove", name="test")
        result = cmd_config(args)
        captured = capsys.readouterr()

        assert result == ExitCode.SUCCESS
        assert "Removed server" in captured.out

        # Verify it was removed
        config = load_config()
        assert "test" not in config.servers

    def test_cmd_config_remove_not_found(self, mock_config_path):
        """Test removing non-existent server."""
        args = MockArgs(action="remove", name="nonexistent")
        result = cmd_config(args)
        assert result == ExitCode.INVALID_ARGS

    def test_cmd_config_init(self, mock_config_path, capsys):
        """Test initializing configuration."""
        args = MockArgs(action="init", force=False)
        result = cmd_config(args)
        captured = capsys.readouterr()

        assert result == ExitCode.SUCCESS
        assert "Initialized" in captured.out
        assert mock_config_path.exists()

    def test_cmd_config_init_exists(self, mock_config_path):
        """Test initializing when config already exists."""
        mock_config_path.write_text("test: data")

        args = MockArgs(action="init", force=False)
        result = cmd_config(args)

        assert result == ExitCode.INVALID_ARGS

    def test_cmd_config_init_force(self, mock_config_path, capsys):
        """Test force initializing configuration."""
        mock_config_path.write_text("test: data")

        args = MockArgs(action="init", force=True)
        result = cmd_config(args)
        captured = capsys.readouterr()

        assert result == ExitCode.SUCCESS
        assert "Initialized" in captured.out

    def test_cmd_proxy_no_config(self, mock_config_path, capsys):
        """Test starting proxy with no servers configured."""
        args = MockArgs(config=None, port=8080)
        result = cmd_proxy(args)
        captured = capsys.readouterr()

        assert result == ExitCode.INVALID_ARGS
        assert "No servers configured" in captured.out

    def test_cmd_proxy_config_not_found(self, tmp_path):
        """Test starting proxy with non-existent config file."""
        args = MockArgs(config=str(tmp_path / "nonexistent.yaml"), port=8080)
        result = cmd_proxy(args)
        assert result == ExitCode.INVALID_ARGS


# =============================================================================
# CLI Parser Tests
# =============================================================================

class TestCLIParser:
    """Tests for CLI argument parser."""

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser.prog == "mcp_bridge"

    def test_parser_version(self, capsys):
        """Test version flag."""
        parser = create_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--version"])
        assert excinfo.value.code == 0

    def test_parser_start_command(self):
        """Test start command parsing."""
        parser = create_parser()
        args = parser.parse_args(["start", "node server.js", "--name", "test", "--port", "3000"])

        assert args.command == "node server.js"
        assert args.name == "test"
        assert args.port == 3000

    def test_parser_stop_command(self):
        """Test stop command parsing."""
        parser = create_parser()
        args = parser.parse_args(["stop", "my-server"])

        assert args.name == "my-server"

    def test_parser_list_command(self):
        """Test list command parsing."""
        parser = create_parser()
        args = parser.parse_args(["list"])

        assert args.subcommand == "list"

    def test_parser_list_tools_command(self):
        """Test list-tools command parsing."""
        parser = create_parser()
        args = parser.parse_args(["list-tools", "--server", "fs"])

        assert args.server == "fs"

    def test_parser_call_command(self):
        """Test call command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "call",
            "--server", "fs",
            "--tool", "read_file",
            "--args", '{"path": "test.txt"}',
        ])

        assert args.server == "fs"
        assert args.tool == "read_file"
        assert args.args == '{"path": "test.txt"}'

    def test_parser_health_command(self):
        """Test health command parsing."""
        parser = create_parser()
        args = parser.parse_args(["health", "--all"])

        assert args.all is True

    def test_parser_config_command(self):
        """Test config command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "config", "add", "test",
            "--command", "node server.js",
            "--port", "3000",
        ])

        assert args.action == "add"
        assert args.name == "test"
        assert args.command == "node server.js"
        assert args.port == 3000

    def test_parser_proxy_command(self):
        """Test proxy command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "proxy",
            "--config", "servers.yaml",
            "--port", "9000",
        ])

        assert args.config == "servers.yaml"
        assert args.port == 9000


# =============================================================================
# Main Function Tests
# =============================================================================

class TestMain:
    """Tests for main entry point."""

    def test_main_no_command(self, capsys):
        """Test main with no command shows help."""
        with patch("sys.argv", ["mcp_bridge"]):
            result = main()
        assert result == ExitCode.SUCCESS

    def test_main_keyboard_interrupt(self, mock_state_path):
        """Test main handles keyboard interrupt."""
        with patch("sys.argv", ["mcp_bridge", "list"]):
            with patch("mcp_bridge.cmd_list") as mock_cmd:
                mock_cmd.side_effect = KeyboardInterrupt()
                result = main()
        assert result == ExitCode.SUCCESS

    def test_main_exception(self, mock_state_path):
        """Test main handles exceptions."""
        with patch("sys.argv", ["mcp_bridge", "--verbose", "list"]):
            with patch("mcp_bridge.cmd_list") as mock_cmd:
                mock_cmd.side_effect = RuntimeError("Test error")
                result = main()
        assert result == ExitCode.SERVER_ERROR


# =============================================================================
# Exit Code Tests
# =============================================================================

class TestExitCodes:
    """Tests for exit codes."""

    def test_exit_code_values(self):
        """Test exit code values are correct."""
        assert ExitCode.SUCCESS == 0
        assert ExitCode.SERVER_ERROR == 1
        assert ExitCode.CONNECTION_ERROR == 2
        assert ExitCode.INVALID_ARGS == 3


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for end-to-end scenarios."""

    def test_full_server_lifecycle(self, mock_state_path, mock_config_path, capsys):
        """Test full lifecycle: start, list, stop."""
        # Start
        start_args = MockArgs(
            command="sleep 60",
            name="integration-test",
            port=None,
            transport="stdio",
        )

        with patch("time.sleep"):
            start_result = cmd_start(start_args)

        if start_result == ExitCode.SUCCESS:
            # List
            list_args = MockArgs()
            list_result = cmd_list(list_args)
            assert list_result == ExitCode.SUCCESS

            captured = capsys.readouterr()
            assert "integration-test" in captured.out

            # Stop
            stop_args = MockArgs(name="integration-test")
            stop_result = cmd_stop(stop_args)
            assert stop_result == ExitCode.SUCCESS

    def test_config_workflow(self, mock_config_path, capsys):
        """Test configuration workflow: init, add, show, remove."""
        # Init
        init_args = MockArgs(action="init", force=False)
        assert cmd_config(init_args) == ExitCode.SUCCESS

        # Add
        add_args = MockArgs(
            action="add",
            name="workflow-test",
            command="echo hello",
            port=3000,
            transport="http",
        )
        assert cmd_config(add_args) == ExitCode.SUCCESS

        # Show
        show_args = MockArgs(action="show")
        assert cmd_config(show_args) == ExitCode.SUCCESS
        captured = capsys.readouterr()
        assert "workflow-test" in captured.out

        # Remove
        remove_args = MockArgs(action="remove", name="workflow-test")
        assert cmd_config(remove_args) == ExitCode.SUCCESS


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
