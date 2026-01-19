#!/usr/bin/env python3
"""
MCP Bridge - A comprehensive CLI tool for managing Model Context Protocol (MCP) servers.

This tool provides a unified interface for:
- Starting and stopping MCP server processes
- Listing available tools from connected servers
- Testing tool calls with arbitrary arguments
- Proxying multiple MCP servers behind a single endpoint
- Health monitoring and server management
- Configuration management via YAML files

MCP (Model Context Protocol) is an open protocol enabling LLM applications to integrate
with external data sources and tools via JSON-RPC 2.0 messages over stdio or HTTP/SSE.

Exit codes:
    0 - Success
    1 - Server error (server crashed, failed to start, etc.)
    2 - Connection error (timeout, refused, network issues)
    3 - Invalid arguments (bad CLI args, invalid config, etc.)

Configuration:
    Default config location: ~/.mcp_bridge/config.yaml
    Override with: MCP_BRIDGE_CONFIG environment variable

Example usage:
    # Start a server from a local script
    mcp_bridge start ./servers/filesystem-server.js --name fs --port 3000

    # Start a server via npx
    mcp_bridge start "npx @anthropic/mcp-server-filesystem" --name fs

    # Stop a running server
    mcp_bridge stop fs

    # List all running servers
    mcp_bridge list

    # List tools available from a specific server
    mcp_bridge list-tools --server fs

    # Call a tool on a server
    mcp_bridge call --server fs --tool read_file --args '{"path": "test.txt"}'

    # Check health of all servers
    mcp_bridge health --all

    # Add a server to configuration
    mcp_bridge config add fs --command "node server.js" --port 3000

    # Start a proxy aggregating multiple servers
    mcp_bridge proxy --config servers.yaml --port 8080

For more detailed help on any command, use:
    mcp_bridge <command> --help
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
import yaml
import websockets

# Version info
__version__ = "1.0.0"
__author__ = "MCP Bridge Contributors"

# =============================================================================
# Exit Codes
# =============================================================================

class ExitCode(IntEnum):
    """Exit codes for the MCP Bridge CLI."""
    SUCCESS = 0
    SERVER_ERROR = 1
    CONNECTION_ERROR = 2
    INVALID_ARGS = 3


# =============================================================================
# Configuration
# =============================================================================

def get_config_path() -> Path:
    """
    Get the configuration file path.

    Returns the path from MCP_BRIDGE_CONFIG environment variable if set,
    otherwise returns the default ~/.mcp_bridge/config.yaml

    Returns:
        Path: The configuration file path
    """
    env_path = os.environ.get("MCP_BRIDGE_CONFIG")
    if env_path:
        return Path(env_path)
    return Path.home() / ".mcp_bridge" / "config.yaml"


def get_state_path() -> Path:
    """
    Get the state file path for tracking running servers.

    Returns:
        Path: The state file path (~/.mcp_bridge/state.json)
    """
    return Path.home() / ".mcp_bridge" / "state.json"


@dataclass
class ServerConfig:
    """Configuration for a single MCP server."""
    name: str
    command: str
    port: Optional[int] = None
    transport: str = "stdio"  # stdio, http, websocket
    env: Dict[str, str] = field(default_factory=dict)
    args: List[str] = field(default_factory=list)
    working_dir: Optional[str] = None
    auto_start: bool = False
    health_check_interval: int = 30  # seconds
    restart_on_failure: bool = False
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BridgeConfig:
    """Main configuration for MCP Bridge."""
    servers: Dict[str, ServerConfig] = field(default_factory=dict)
    default_timeout: int = 30
    log_level: str = "INFO"
    proxy_port: int = 8080

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "servers": {name: srv.to_dict() for name, srv in self.servers.items()},
            "default_timeout": self.default_timeout,
            "log_level": self.log_level,
            "proxy_port": self.proxy_port,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BridgeConfig":
        """Create from dictionary."""
        servers = {}
        for name, srv_data in data.get("servers", {}).items():
            srv_data["name"] = name
            servers[name] = ServerConfig.from_dict(srv_data)
        return cls(
            servers=servers,
            default_timeout=data.get("default_timeout", 30),
            log_level=data.get("log_level", "INFO"),
            proxy_port=data.get("proxy_port", 8080),
        )


def load_config() -> BridgeConfig:
    """
    Load configuration from YAML file.

    Returns:
        BridgeConfig: The loaded configuration or default if file doesn't exist
    """
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        return BridgeConfig.from_dict(data)
    return BridgeConfig()


def save_config(config: BridgeConfig) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: The configuration to save
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)


# =============================================================================
# Server State Management
# =============================================================================

@dataclass
class ServerState:
    """Runtime state of a running server."""
    name: str
    pid: int
    command: str
    port: Optional[int]
    transport: str
    started_at: str
    status: str = "running"
    health: str = "unknown"
    last_health_check: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerState":
        """Create from dictionary."""
        return cls(**data)


def load_state() -> Dict[str, ServerState]:
    """
    Load running server state from JSON file.

    Returns:
        Dict[str, ServerState]: Map of server names to their states
    """
    state_path = get_state_path()
    if state_path.exists():
        with open(state_path, "r") as f:
            data = json.load(f)
        return {name: ServerState.from_dict(s) for name, s in data.items()}
    return {}


def save_state(state: Dict[str, ServerState]) -> None:
    """
    Save running server state to JSON file.

    Args:
        state: Map of server names to their states
    """
    state_path = get_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump({name: s.to_dict() for name, s in state.items()}, f, indent=2)


def cleanup_dead_servers() -> Dict[str, ServerState]:
    """
    Clean up state entries for servers that are no longer running.

    Returns:
        Dict[str, ServerState]: Cleaned state with only running servers
    """
    state = load_state()
    cleaned = {}
    for name, server_state in state.items():
        if is_process_running(server_state.pid):
            cleaned[name] = server_state
    save_state(cleaned)
    return cleaned


def is_process_running(pid: int) -> bool:
    """
    Check if a process with the given PID is running.

    Args:
        pid: Process ID to check

    Returns:
        bool: True if process is running, False otherwise
    """
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# =============================================================================
# JSON-RPC Protocol
# =============================================================================

class JsonRpcError(Exception):
    """Exception for JSON-RPC errors."""
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC Error {code}: {message}")


def make_jsonrpc_request(method: str, params: Any = None, id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a JSON-RPC 2.0 request.

    Args:
        method: The method name to call
        params: Optional parameters for the method
        id: Optional request ID (generated if not provided)

    Returns:
        Dict: JSON-RPC request object
    """
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "id": id or str(uuid.uuid4()),
    }
    if params is not None:
        request["params"] = params
    return request


def parse_jsonrpc_response(response: Dict[str, Any]) -> Any:
    """
    Parse a JSON-RPC 2.0 response.

    Args:
        response: The response object to parse

    Returns:
        Any: The result from the response

    Raises:
        JsonRpcError: If the response contains an error
    """
    if "error" in response:
        error = response["error"]
        raise JsonRpcError(
            error.get("code", -1),
            error.get("message", "Unknown error"),
            error.get("data"),
        )
    return response.get("result")


# =============================================================================
# MCP Client
# =============================================================================

class MCPClient:
    """
    Client for communicating with MCP servers.

    Supports both stdio and HTTP/SSE transports as defined in the MCP specification.
    """

    def __init__(
        self,
        transport: str = "stdio",
        command: Optional[str] = None,
        host: str = "localhost",
        port: Optional[int] = None,
        timeout: int = 30,
    ):
        """
        Initialize MCP client.

        Args:
            transport: Transport type ("stdio", "http", "websocket")
            command: Command to run for stdio transport
            host: Host for HTTP/WebSocket transport
            port: Port for HTTP/WebSocket transport
            timeout: Request timeout in seconds
        """
        self.transport = transport
        self.command = command
        self.host = host
        self.port = port
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._initialized = False
        self._capabilities: Dict[str, Any] = {}

    async def connect(self) -> None:
        """
        Establish connection to the MCP server.

        Raises:
            ConnectionError: If connection fails
        """
        if self.transport == "stdio":
            await self._connect_stdio()
        elif self.transport == "http":
            await self._connect_http()
        elif self.transport == "websocket":
            await self._connect_websocket()
        else:
            raise ValueError(f"Unsupported transport: {self.transport}")

        # Initialize the connection
        await self._initialize()

    async def _connect_stdio(self) -> None:
        """Connect via stdio transport."""
        if not self.command:
            raise ValueError("Command required for stdio transport")

        # Parse command - handle both direct commands and complex commands
        if " " in self.command:
            # Complex command, use shell
            self._process = subprocess.Popen(
                self.command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            # Simple command
            self._process = subprocess.Popen(
                [self.command],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

    async def _connect_http(self) -> None:
        """Connect via HTTP transport."""
        self._http_client = httpx.AsyncClient(
            base_url=f"http://{self.host}:{self.port}",
            timeout=self.timeout,
        )

    async def _connect_websocket(self) -> None:
        """Connect via WebSocket transport."""
        uri = f"ws://{self.host}:{self.port}"
        self._ws = await websockets.connect(uri)

    async def _initialize(self) -> None:
        """
        Initialize the MCP session.

        Sends the initialize request and negotiates capabilities.
        """
        response = await self.request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
            },
            "clientInfo": {
                "name": "mcp_bridge",
                "version": __version__,
            },
        })
        self._capabilities = response.get("capabilities", {})
        self._initialized = True

        # Send initialized notification
        await self.notify("notifications/initialized")

    async def request(self, method: str, params: Any = None) -> Any:
        """
        Send a JSON-RPC request and wait for response.

        Args:
            method: Method name to call
            params: Optional parameters

        Returns:
            Any: The result from the response
        """
        request = make_jsonrpc_request(method, params)
        response = await self._send_receive(request)
        return parse_jsonrpc_response(response)

    async def notify(self, method: str, params: Any = None) -> None:
        """
        Send a JSON-RPC notification (no response expected).

        Args:
            method: Method name
            params: Optional parameters
        """
        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            notification["params"] = params
        await self._send(notification)

    async def _send_receive(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request and receive response.

        Args:
            request: JSON-RPC request object

        Returns:
            Dict: JSON-RPC response object
        """
        if self.transport == "stdio":
            return await self._stdio_send_receive(request)
        elif self.transport == "http":
            return await self._http_send_receive(request)
        elif self.transport == "websocket":
            return await self._ws_send_receive(request)
        raise ValueError(f"Unsupported transport: {self.transport}")

    async def _send(self, message: Dict[str, Any]) -> None:
        """Send a message without waiting for response."""
        if self.transport == "stdio":
            await self._stdio_send(message)
        elif self.transport == "http":
            await self._http_send(message)
        elif self.transport == "websocket":
            await self._ws_send(message)

    async def _stdio_send_receive(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send/receive via stdio."""
        if not self._process:
            raise ConnectionError("Process not started")

        # Write request
        message = json.dumps(request) + "\n"
        self._process.stdin.write(message.encode())
        self._process.stdin.flush()

        # Read response
        line = self._process.stdout.readline()
        if not line:
            raise ConnectionError("No response from server")
        return json.loads(line.decode())

    async def _stdio_send(self, message: Dict[str, Any]) -> None:
        """Send via stdio without waiting."""
        if not self._process:
            raise ConnectionError("Process not started")
        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode())
        self._process.stdin.flush()

    async def _http_send_receive(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send/receive via HTTP."""
        if not self._http_client:
            raise ConnectionError("HTTP client not initialized")
        response = await self._http_client.post("/", json=request)
        response.raise_for_status()
        return response.json()

    async def _http_send(self, message: Dict[str, Any]) -> None:
        """Send via HTTP without waiting."""
        if not self._http_client:
            raise ConnectionError("HTTP client not initialized")
        await self._http_client.post("/", json=message)

    async def _ws_send_receive(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send/receive via WebSocket."""
        if not self._ws:
            raise ConnectionError("WebSocket not connected")
        await self._ws.send(json.dumps(request))
        response = await self._ws.recv()
        return json.loads(response)

    async def _ws_send(self, message: Dict[str, Any]) -> None:
        """Send via WebSocket without waiting."""
        if not self._ws:
            raise ConnectionError("WebSocket not connected")
        await self._ws.send(json.dumps(message))

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the server.

        Returns:
            List[Dict]: List of tool definitions
        """
        response = await self.request("tools/list")
        return response.get("tools", [])

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the server.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Any: Tool result
        """
        response = await self.request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        return response

    async def list_resources(self) -> List[Dict[str, Any]]:
        """
        List available resources from the server.

        Returns:
            List[Dict]: List of resource definitions
        """
        response = await self.request("resources/list")
        return response.get("resources", [])

    async def list_prompts(self) -> List[Dict[str, Any]]:
        """
        List available prompts from the server.

        Returns:
            List[Dict]: List of prompt definitions
        """
        response = await self.request("prompts/list")
        return response.get("prompts", [])

    async def ping(self) -> bool:
        """
        Send a ping to check server health.

        Returns:
            bool: True if server responds
        """
        try:
            await self.request("ping")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the connection."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        if self._ws:
            await self._ws.close()
            self._ws = None

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Get server capabilities."""
        return self._capabilities


# =============================================================================
# MCP Proxy Server
# =============================================================================

class MCPProxy:
    """
    Proxy server that aggregates multiple MCP servers.

    Allows clients to connect to a single endpoint and access tools
    from multiple backend MCP servers.
    """

    def __init__(self, port: int = 8080):
        """
        Initialize the proxy server.

        Args:
            port: Port to listen on
        """
        self.port = port
        self.backends: Dict[str, MCPClient] = {}
        self._running = False

    def add_backend(self, name: str, client: MCPClient) -> None:
        """
        Add a backend MCP server.

        Args:
            name: Name for the backend
            client: Connected MCP client
        """
        self.backends[name] = client

    async def list_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List tools from all backends.

        Returns:
            Dict: Map of backend names to their tools
        """
        tools = {}
        for name, client in self.backends.items():
            try:
                tools[name] = await client.list_tools()
            except Exception as e:
                tools[name] = [{"error": str(e)}]
        return tools

    async def call_tool(
        self,
        backend: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """
        Call a tool on a specific backend.

        Args:
            backend: Backend name
            tool_name: Tool to call
            arguments: Tool arguments

        Returns:
            Any: Tool result
        """
        if backend not in self.backends:
            raise ValueError(f"Unknown backend: {backend}")
        return await self.backends[backend].call_tool(tool_name, arguments)

    async def start(self) -> None:
        """Start the proxy server."""
        self._running = True
        print(f"Proxy server starting on port {self.port}")

        # Simple HTTP server for the proxy
        async def handle_request(reader, writer):
            """Handle incoming HTTP request."""
            data = await reader.read(4096)
            if not data:
                return

            # Parse HTTP request (simplified)
            lines = data.decode().split("\r\n")
            if not lines:
                return

            request_line = lines[0]
            method, path, _ = request_line.split(" ")

            # Find body
            body = ""
            for i, line in enumerate(lines):
                if line == "":
                    body = "\r\n".join(lines[i + 1:])
                    break

            # Process JSON-RPC request
            response_body = await self._handle_jsonrpc(body)

            # Send response
            response = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(response_body)}\r\n"
                "\r\n"
                f"{response_body}"
            )
            writer.write(response.encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(
            handle_request,
            "localhost",
            self.port,
        )

        async with server:
            await server.serve_forever()

    async def _handle_jsonrpc(self, body: str) -> str:
        """
        Handle a JSON-RPC request.

        Args:
            body: Request body

        Returns:
            str: JSON response
        """
        try:
            request = json.loads(body)
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")

            if method == "tools/list":
                result = await self.list_all_tools()
            elif method == "tools/call":
                backend = params.get("backend")
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                result = await self.call_tool(backend, tool_name, arguments)
            else:
                result = {"error": f"Unknown method: {method}"}

            return json.dumps({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            })
        except Exception as e:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32000,
                    "message": str(e),
                },
            })

    async def stop(self) -> None:
        """Stop the proxy server."""
        self._running = False
        for client in self.backends.values():
            await client.close()


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_start(args: argparse.Namespace) -> int:
    """
    Start an MCP server.

    Args:
        args: Parsed command line arguments

    Returns:
        int: Exit code
    """
    command = args.command
    name = args.name or f"server_{int(time.time())}"
    port = args.port
    transport = args.transport or "stdio"

    # Check if server with this name already exists
    state = cleanup_dead_servers()
    if name in state:
        print(f"Error: Server '{name}' is already running (PID: {state[name].pid})")
        return ExitCode.INVALID_ARGS

    print(f"Starting server '{name}'...")
    print(f"  Command: {command}")
    print(f"  Transport: {transport}")
    if port:
        print(f"  Port: {port}")

    try:
        # Start the process
        if " " in command:
            process = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Detach from terminal
            )
        else:
            process = subprocess.Popen(
                [command],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

        # Give it a moment to start
        time.sleep(0.5)

        # Check if it's still running
        if process.poll() is not None:
            stderr = process.stderr.read().decode() if process.stderr else ""
            print(f"Error: Server failed to start")
            if stderr:
                print(f"  stderr: {stderr}")
            return ExitCode.SERVER_ERROR

        # Save state
        server_state = ServerState(
            name=name,
            pid=process.pid,
            command=command,
            port=port,
            transport=transport,
            started_at=datetime.now().isoformat(),
        )
        state[name] = server_state
        save_state(state)

        print(f"Server '{name}' started successfully (PID: {process.pid})")
        return ExitCode.SUCCESS

    except FileNotFoundError:
        print(f"Error: Command not found: {command}")
        return ExitCode.SERVER_ERROR
    except Exception as e:
        print(f"Error starting server: {e}")
        return ExitCode.SERVER_ERROR


def cmd_stop(args: argparse.Namespace) -> int:
    """
    Stop an MCP server.

    Args:
        args: Parsed command line arguments

    Returns:
        int: Exit code
    """
    name = args.name

    state = cleanup_dead_servers()
    if name not in state:
        print(f"Error: Server '{name}' not found or not running")
        return ExitCode.INVALID_ARGS

    server_state = state[name]
    pid = server_state.pid

    print(f"Stopping server '{name}' (PID: {pid})...")

    try:
        # Send SIGTERM
        os.kill(pid, signal.SIGTERM)

        # Wait for graceful shutdown
        for _ in range(10):
            time.sleep(0.5)
            if not is_process_running(pid):
                break

        # Force kill if still running
        if is_process_running(pid):
            print("  Server not responding, sending SIGKILL...")
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)

        # Update state
        del state[name]
        save_state(state)

        print(f"Server '{name}' stopped")
        return ExitCode.SUCCESS

    except ProcessLookupError:
        # Process already dead
        del state[name]
        save_state(state)
        print(f"Server '{name}' was already stopped")
        return ExitCode.SUCCESS
    except Exception as e:
        print(f"Error stopping server: {e}")
        return ExitCode.SERVER_ERROR


def cmd_list(args: argparse.Namespace) -> int:
    """
    List running MCP servers.

    Args:
        args: Parsed command line arguments

    Returns:
        int: Exit code
    """
    state = cleanup_dead_servers()

    if not state:
        print("No running servers")
        return ExitCode.SUCCESS

    print("Running MCP servers:")
    print("-" * 80)
    print(f"{'Name':<15} {'PID':<8} {'Transport':<10} {'Port':<8} {'Started':<20} {'Status':<10}")
    print("-" * 80)

    for name, server_state in state.items():
        port_str = str(server_state.port) if server_state.port else "-"
        started = server_state.started_at[:19] if server_state.started_at else "-"
        print(f"{name:<15} {server_state.pid:<8} {server_state.transport:<10} {port_str:<8} {started:<20} {server_state.status:<10}")

    print("-" * 80)
    print(f"Total: {len(state)} server(s)")

    return ExitCode.SUCCESS


def cmd_list_tools(args: argparse.Namespace) -> int:
    """
    List tools from an MCP server.

    Args:
        args: Parsed command line arguments

    Returns:
        int: Exit code
    """
    server_name = args.server

    state = cleanup_dead_servers()
    if server_name not in state:
        print(f"Error: Server '{server_name}' not found or not running")
        return ExitCode.INVALID_ARGS

    server_state = state[server_name]

    print(f"Listing tools from server '{server_name}'...")

    try:
        # Create client and connect
        async def list_tools():
            client = MCPClient(
                transport=server_state.transport,
                command=server_state.command,
                port=server_state.port,
            )
            await client.connect()
            tools = await client.list_tools()
            await client.close()
            return tools

        tools = asyncio.run(list_tools())

        if not tools:
            print("No tools available")
            return ExitCode.SUCCESS

        print(f"\nTools from '{server_name}':")
        print("-" * 60)

        for tool in tools:
            name = tool.get("name", "unnamed")
            description = tool.get("description", "No description")
            print(f"\n  {name}")
            print(f"    Description: {description}")

            input_schema = tool.get("inputSchema", {})
            if input_schema:
                properties = input_schema.get("properties", {})
                required = input_schema.get("required", [])

                if properties:
                    print("    Parameters:")
                    for param_name, param_info in properties.items():
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "")
                        req = "*" if param_name in required else ""
                        print(f"      - {param_name}{req} ({param_type}): {param_desc}")

        print("-" * 60)
        print(f"Total: {len(tools)} tool(s)")

        return ExitCode.SUCCESS

    except ConnectionError as e:
        print(f"Connection error: {e}")
        return ExitCode.CONNECTION_ERROR
    except Exception as e:
        print(f"Error listing tools: {e}")
        return ExitCode.SERVER_ERROR


def cmd_call(args: argparse.Namespace) -> int:
    """
    Call a tool on an MCP server.

    Args:
        args: Parsed command line arguments

    Returns:
        int: Exit code
    """
    server_name = args.server
    tool_name = args.tool
    arguments_json = args.args or "{}"

    state = cleanup_dead_servers()
    if server_name not in state:
        print(f"Error: Server '{server_name}' not found or not running")
        return ExitCode.INVALID_ARGS

    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON arguments: {e}")
        return ExitCode.INVALID_ARGS

    server_state = state[server_name]

    print(f"Calling tool '{tool_name}' on server '{server_name}'...")
    print(f"Arguments: {json.dumps(arguments, indent=2)}")

    try:
        async def call_tool():
            client = MCPClient(
                transport=server_state.transport,
                command=server_state.command,
                port=server_state.port,
            )
            await client.connect()
            result = await client.call_tool(tool_name, arguments)
            await client.close()
            return result

        result = asyncio.run(call_tool())

        print("\nResult:")
        print(json.dumps(result, indent=2))

        return ExitCode.SUCCESS

    except ConnectionError as e:
        print(f"Connection error: {e}")
        return ExitCode.CONNECTION_ERROR
    except JsonRpcError as e:
        print(f"Tool call error: {e}")
        return ExitCode.SERVER_ERROR
    except Exception as e:
        print(f"Error calling tool: {e}")
        return ExitCode.SERVER_ERROR


def cmd_health(args: argparse.Namespace) -> int:
    """
    Check health of MCP servers.

    Args:
        args: Parsed command line arguments

    Returns:
        int: Exit code
    """
    state = cleanup_dead_servers()

    if args.all:
        servers_to_check = list(state.keys())
    elif args.server:
        if args.server not in state:
            print(f"Error: Server '{args.server}' not found or not running")
            return ExitCode.INVALID_ARGS
        servers_to_check = [args.server]
    else:
        servers_to_check = list(state.keys())

    if not servers_to_check:
        print("No servers to check")
        return ExitCode.SUCCESS

    print("Health check results:")
    print("-" * 60)

    all_healthy = True

    for name in servers_to_check:
        server_state = state[name]

        # Check if process is running
        if not is_process_running(server_state.pid):
            print(f"  {name}: DEAD (process not running)")
            all_healthy = False
            continue

        # Try to ping
        try:
            async def check_health():
                client = MCPClient(
                    transport=server_state.transport,
                    command=server_state.command,
                    port=server_state.port,
                )
                await client.connect()
                healthy = await client.ping()
                await client.close()
                return healthy

            healthy = asyncio.run(check_health())

            if healthy:
                print(f"  {name}: HEALTHY")
            else:
                print(f"  {name}: UNHEALTHY (ping failed)")
                all_healthy = False

        except Exception as e:
            print(f"  {name}: UNHEALTHY ({e})")
            all_healthy = False

    print("-" * 60)

    return ExitCode.SUCCESS if all_healthy else ExitCode.SERVER_ERROR


def cmd_config(args: argparse.Namespace) -> int:
    """
    Manage MCP Bridge configuration.

    Args:
        args: Parsed command line arguments

    Returns:
        int: Exit code
    """
    action = args.action

    if action == "show":
        config = load_config()
        print(f"Configuration file: {get_config_path()}")
        print("-" * 60)
        print(yaml.dump(config.to_dict(), default_flow_style=False))
        return ExitCode.SUCCESS

    elif action == "add":
        name = args.name
        command = args.command
        port = args.port
        transport = args.transport or "stdio"

        if not name or not command:
            print("Error: --name and --command are required for 'add'")
            return ExitCode.INVALID_ARGS

        config = load_config()
        config.servers[name] = ServerConfig(
            name=name,
            command=command,
            port=port,
            transport=transport,
        )
        save_config(config)

        print(f"Added server '{name}' to configuration")
        return ExitCode.SUCCESS

    elif action == "remove":
        name = args.name
        if not name:
            print("Error: --name is required for 'remove'")
            return ExitCode.INVALID_ARGS

        config = load_config()
        if name not in config.servers:
            print(f"Error: Server '{name}' not found in configuration")
            return ExitCode.INVALID_ARGS

        del config.servers[name]
        save_config(config)

        print(f"Removed server '{name}' from configuration")
        return ExitCode.SUCCESS

    elif action == "init":
        config_path = get_config_path()
        if config_path.exists() and not args.force:
            print(f"Error: Configuration already exists at {config_path}")
            print("Use --force to overwrite")
            return ExitCode.INVALID_ARGS

        config = BridgeConfig()
        save_config(config)
        print(f"Initialized configuration at {config_path}")
        return ExitCode.SUCCESS

    else:
        print(f"Error: Unknown config action: {action}")
        return ExitCode.INVALID_ARGS


def cmd_proxy(args: argparse.Namespace) -> int:
    """
    Start the MCP proxy server.

    Args:
        args: Parsed command line arguments

    Returns:
        int: Exit code
    """
    config_file = args.config
    port = args.port or 8080

    print(f"Starting MCP proxy server on port {port}...")

    # Load configuration
    if config_file:
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_file}")
            return ExitCode.INVALID_ARGS

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config = BridgeConfig.from_dict(config_data)
    else:
        config = load_config()

    if not config.servers:
        print("Error: No servers configured")
        print("Add servers with: mcp_bridge config add <name> --command <cmd>")
        return ExitCode.INVALID_ARGS

    async def run_proxy():
        proxy = MCPProxy(port=port)

        # Connect to all configured servers
        for name, server_config in config.servers.items():
            print(f"  Connecting to backend: {name}")
            try:
                client = MCPClient(
                    transport=server_config.transport,
                    command=server_config.command,
                    port=server_config.port,
                )
                await client.connect()
                proxy.add_backend(name, client)
                print(f"    Connected to {name}")
            except Exception as e:
                print(f"    Failed to connect to {name}: {e}")

        if not proxy.backends:
            print("Error: No backends connected")
            return ExitCode.CONNECTION_ERROR

        print(f"\nProxy running with {len(proxy.backends)} backend(s)")
        print("Press Ctrl+C to stop")

        try:
            await proxy.start()
        except KeyboardInterrupt:
            print("\nShutting down proxy...")
            await proxy.stop()

        return ExitCode.SUCCESS

    return asyncio.run(run_proxy())


# =============================================================================
# CLI Argument Parser
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser with all commands and options.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="mcp_bridge",
        description="""
MCP Bridge - A comprehensive CLI tool for managing Model Context Protocol (MCP) servers.

MCP (Model Context Protocol) is an open protocol enabling LLM applications to integrate
with external data sources and tools via JSON-RPC 2.0 messages.

This tool provides a unified interface for starting, stopping, and managing MCP servers,
as well as listing and calling their tools.
        """,
        epilog="""
Exit Codes:
    0 - Success
    1 - Server error (server crashed, failed to start, etc.)
    2 - Connection error (timeout, refused, network issues)
    3 - Invalid arguments (bad CLI args, invalid config, etc.)

Configuration:
    Default config: ~/.mcp_bridge/config.yaml
    Override with: MCP_BRIDGE_CONFIG environment variable

Examples:
    # Start a server
    %(prog)s start ./servers/filesystem-server.js --name fs

    # Start via npx
    %(prog)s start "npx @anthropic/mcp-server-filesystem" --name fs

    # List running servers
    %(prog)s list

    # List tools from a server
    %(prog)s list-tools --server fs

    # Call a tool
    %(prog)s call --server fs --tool read_file --args '{"path": "test.txt"}'

    # Check health
    %(prog)s health --all

    # Start proxy
    %(prog)s proxy --port 8080

For more information, visit: https://modelcontextprotocol.io
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="subcommand",
        metavar="<command>",
    )

    # -------------------------------------------------------------------------
    # start command
    # -------------------------------------------------------------------------
    start_parser = subparsers.add_parser(
        "start",
        help="Start an MCP server",
        description="""
Start an MCP server process. The server can be a local script or an npx command.

The server will run in the background and can be stopped with the 'stop' command.
        """,
        epilog="""
Examples:
    # Start a local JavaScript server
    %(prog)s ./servers/filesystem-server.js --name fs --port 3000

    # Start via npx
    %(prog)s "npx @anthropic/mcp-server-filesystem" --name fs

    # Start with HTTP transport
    %(prog)s ./server.py --name api --transport http --port 8000
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    start_parser.add_argument(
        "command",
        help="Command to start the server (can be a path or shell command)",
    )
    start_parser.add_argument(
        "--name", "-n",
        help="Name for the server (auto-generated if not provided)",
    )
    start_parser.add_argument(
        "--port", "-p",
        type=int,
        help="Port for HTTP/WebSocket transport",
    )
    start_parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "http", "websocket"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    start_parser.set_defaults(func=cmd_start)

    # -------------------------------------------------------------------------
    # stop command
    # -------------------------------------------------------------------------
    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop a running MCP server",
        description="Stop a running MCP server by name.",
        epilog="""
Examples:
    %(prog)s fs
    %(prog)s my-server
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    stop_parser.add_argument(
        "name",
        help="Name of the server to stop",
    )
    stop_parser.set_defaults(func=cmd_stop)

    # -------------------------------------------------------------------------
    # list command
    # -------------------------------------------------------------------------
    list_parser = subparsers.add_parser(
        "list",
        help="List running MCP servers",
        description="List all running MCP servers and their status.",
        epilog="""
Example:
    %(prog)s
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    list_parser.set_defaults(func=cmd_list)

    # -------------------------------------------------------------------------
    # list-tools command
    # -------------------------------------------------------------------------
    list_tools_parser = subparsers.add_parser(
        "list-tools",
        help="List tools from an MCP server",
        description="""
List all tools available from a running MCP server.

Shows tool names, descriptions, and parameter information.
        """,
        epilog="""
Example:
    %(prog)s --server fs
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    list_tools_parser.add_argument(
        "--server", "-s",
        required=True,
        help="Name of the server to query",
    )
    list_tools_parser.set_defaults(func=cmd_list_tools)

    # -------------------------------------------------------------------------
    # call command
    # -------------------------------------------------------------------------
    call_parser = subparsers.add_parser(
        "call",
        help="Call a tool on an MCP server",
        description="""
Call a tool on a running MCP server with specified arguments.

Arguments must be provided as a JSON object.
        """,
        epilog="""
Examples:
    # Call a tool with no arguments
    %(prog)s --server fs --tool list_files

    # Call a tool with arguments
    %(prog)s --server fs --tool read_file --args '{"path": "test.txt"}'

    # Complex arguments
    %(prog)s --server db --tool query --args '{"sql": "SELECT * FROM users", "limit": 10}'
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    call_parser.add_argument(
        "--server", "-s",
        required=True,
        help="Name of the server to call",
    )
    call_parser.add_argument(
        "--tool", "-t",
        required=True,
        help="Name of the tool to call",
    )
    call_parser.add_argument(
        "--args", "-a",
        default="{}",
        help="JSON object of arguments (default: {})",
    )
    call_parser.set_defaults(func=cmd_call)

    # -------------------------------------------------------------------------
    # health command
    # -------------------------------------------------------------------------
    health_parser = subparsers.add_parser(
        "health",
        help="Check health of MCP servers",
        description="Check the health status of running MCP servers.",
        epilog="""
Examples:
    # Check all servers
    %(prog)s --all

    # Check specific server
    %(prog)s --server fs
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    health_parser.add_argument(
        "--all",
        action="store_true",
        help="Check health of all running servers",
    )
    health_parser.add_argument(
        "--server", "-s",
        help="Check health of a specific server",
    )
    health_parser.set_defaults(func=cmd_health)

    # -------------------------------------------------------------------------
    # config command
    # -------------------------------------------------------------------------
    config_parser = subparsers.add_parser(
        "config",
        help="Manage MCP Bridge configuration",
        description="""
Manage MCP Bridge configuration file.

The configuration file stores server definitions and settings.
Default location: ~/.mcp_bridge/config.yaml
        """,
        epilog="""
Examples:
    # Show current configuration
    %(prog)s show

    # Initialize new configuration
    %(prog)s init

    # Add a server
    %(prog)s add fs --command "node server.js" --port 3000

    # Remove a server
    %(prog)s remove fs
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config_parser.add_argument(
        "action",
        choices=["show", "add", "remove", "init"],
        help="Configuration action",
    )
    config_parser.add_argument(
        "name",
        nargs="?",
        help="Server name (for add/remove)",
    )
    config_parser.add_argument(
        "--command", "-c",
        help="Command to start the server",
    )
    config_parser.add_argument(
        "--port", "-p",
        type=int,
        help="Port for HTTP/WebSocket transport",
    )
    config_parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "http", "websocket"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    config_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force overwrite existing configuration",
    )
    config_parser.set_defaults(func=cmd_config)

    # -------------------------------------------------------------------------
    # proxy command
    # -------------------------------------------------------------------------
    proxy_parser = subparsers.add_parser(
        "proxy",
        help="Start MCP proxy server",
        description="""
Start a proxy server that aggregates multiple MCP servers.

The proxy exposes a single endpoint that can route tool calls to
multiple backend MCP servers.
        """,
        epilog="""
Examples:
    # Start proxy with default config
    %(prog)s --port 8080

    # Start proxy with custom config file
    %(prog)s --config servers.yaml --port 8080
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    proxy_parser.add_argument(
        "--config", "-c",
        help="Path to proxy configuration file",
    )
    proxy_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    proxy_parser.set_defaults(func=cmd_proxy)

    return parser


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """
    Main entry point for the MCP Bridge CLI.

    Returns:
        int: Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        return ExitCode.SUCCESS

    if hasattr(args, "func"):
        try:
            return args.func(args)
        except KeyboardInterrupt:
            print("\nInterrupted")
            return ExitCode.SUCCESS
        except Exception as e:
            if args.verbose:
                import traceback
                traceback.print_exc()
            else:
                print(f"Error: {e}")
            return ExitCode.SERVER_ERROR

    parser.print_help()
    return ExitCode.SUCCESS


if __name__ == "__main__":
    sys.exit(main())
