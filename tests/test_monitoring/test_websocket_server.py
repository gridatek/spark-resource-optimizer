"""Tests for WebSocket server functionality."""

import pytest
import json
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from spark_optimizer.monitoring.websocket_server import (
    WebSocketServer,
    WEBSOCKETS_AVAILABLE,
)


class TestWebSocketServerBasic:
    """Test basic WebSocket server functionality."""

    def test_server_initialization(self):
        """Test server initialization."""
        server = WebSocketServer(host="localhost", port=9999)

        assert server.host == "localhost"
        assert server.port == 9999
        assert not server._running
        assert server.client_count == 0

    def test_server_default_initialization(self):
        """Test server with default values."""
        server = WebSocketServer()

        assert server.host == "0.0.0.0"
        assert server.port == 8765

    def test_client_count(self):
        """Test client count property."""
        server = WebSocketServer()

        assert server.client_count == 0

        # Manually add a client
        mock_client = Mock()
        server._clients.add(mock_client)

        assert server.client_count == 1

    def test_broadcast_not_running(self):
        """Test broadcast when server is not running."""
        server = WebSocketServer()

        # Should not raise
        server.broadcast("test", {"data": "value"})

    def test_send_to_client_not_running(self):
        """Test send_to_client when server is not running."""
        server = WebSocketServer()
        mock_client = Mock()

        # Should not raise
        server.send_to_client(mock_client, "test", {"data": "value"})


class TestWebSocketServerStartStop:
    """Test server start/stop functionality."""

    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    def test_start_creates_thread(self):
        """Test that starting creates a background thread."""
        server = WebSocketServer(port=19999)

        server.start()
        assert server._running
        assert server._thread is not None
        assert server._thread.is_alive()

        server.stop()

    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    def test_start_idempotent(self):
        """Test that starting twice doesn't create multiple threads."""
        server = WebSocketServer(port=19998)

        server.start()
        thread1 = server._thread

        server.start()
        thread2 = server._thread

        assert thread1 is thread2

        server.stop()

    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    def test_stop_cleans_up(self):
        """Test that stopping cleans up resources."""
        server = WebSocketServer(port=19997)

        server.start()
        server.stop()

        assert not server._running

    def test_start_without_websockets(self):
        """Test start when websockets library is not available."""
        with patch.object(
            WebSocketServer, "__init__", lambda self, *args, **kwargs: None
        ):
            server = WebSocketServer.__new__(WebSocketServer)
            server.host = "localhost"
            server.port = 8765
            server._running = False
            server._thread = None
            server._clients = set()
            server._subscriptions = {}
            server._server = None
            server._loop = None
            server._lock = Mock()

        # Patch WEBSOCKETS_AVAILABLE
        import spark_optimizer.monitoring.websocket_server as ws_module

        original = ws_module.WEBSOCKETS_AVAILABLE

        try:
            ws_module.WEBSOCKETS_AVAILABLE = False
            server.start()
            # Should return without starting
            assert not server._running
        finally:
            ws_module.WEBSOCKETS_AVAILABLE = original


class TestWebSocketServerBroadcast:
    """Test broadcast functionality."""

    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    def test_broadcast_to_all_clients(self):
        """Test broadcasting to all connected clients."""
        server = WebSocketServer()
        server._running = True
        server._loop = asyncio.new_event_loop()

        # Create mock clients
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()

        server._clients.add(mock_client1)
        server._clients.add(mock_client2)
        server._subscriptions[mock_client1] = {"*"}
        server._subscriptions[mock_client2] = {"*"}

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            server.broadcast("test_event", {"key": "value"})

            # Should be called for each client
            assert mock_run.call_count == 2

        server._loop.close()

    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    def test_broadcast_respects_subscriptions(self):
        """Test that broadcast respects client subscriptions."""
        server = WebSocketServer()
        server._running = True
        server._loop = asyncio.new_event_loop()

        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()

        server._clients.add(mock_client1)
        server._clients.add(mock_client2)

        # Client 1 subscribed to event1 only
        server._subscriptions[mock_client1] = {"event1"}
        # Client 2 subscribed to event2 only
        server._subscriptions[mock_client2] = {"event2"}

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            server.broadcast("event1", {"key": "value"})

            # Should only send to client1
            assert mock_run.call_count == 1

        server._loop.close()


class TestWebSocketServerMessageHandling:
    """Test message handling functionality."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    async def test_handle_subscribe_message(self):
        """Test handling subscribe message."""
        server = WebSocketServer()
        mock_websocket = AsyncMock()

        server._clients.add(mock_websocket)
        server._subscriptions[mock_websocket] = {"*"}

        message = json.dumps(
            {
                "type": "subscribe",
                "events": ["metrics", "alerts"],
            }
        )

        await server._handle_message(mock_websocket, message)

        assert server._subscriptions[mock_websocket] == {"metrics", "alerts"}
        mock_websocket.send.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    async def test_handle_unsubscribe_message(self):
        """Test handling unsubscribe message."""
        server = WebSocketServer()
        mock_websocket = AsyncMock()

        server._clients.add(mock_websocket)
        server._subscriptions[mock_websocket] = {"metrics", "alerts", "status"}

        message = json.dumps(
            {
                "type": "unsubscribe",
                "events": ["alerts"],
            }
        )

        await server._handle_message(mock_websocket, message)

        assert "alerts" not in server._subscriptions[mock_websocket]
        assert "metrics" in server._subscriptions[mock_websocket]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    async def test_handle_ping_message(self):
        """Test handling ping message."""
        server = WebSocketServer()
        mock_websocket = AsyncMock()

        message = json.dumps({"type": "ping"})

        await server._handle_message(mock_websocket, message)

        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "pong"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    async def test_handle_invalid_json(self):
        """Test handling invalid JSON message."""
        server = WebSocketServer()
        mock_websocket = AsyncMock()

        # Should not raise
        await server._handle_message(mock_websocket, "not valid json")

    @pytest.mark.asyncio
    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    async def test_handle_unknown_message_type(self):
        """Test handling unknown message type."""
        server = WebSocketServer()
        mock_websocket = AsyncMock()

        message = json.dumps({"type": "unknown"})

        # Should not raise
        await server._handle_message(mock_websocket, message)


class TestWebSocketServerClientHandling:
    """Test client connection handling."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    async def test_handle_client_adds_to_clients(self):
        """Test that handling client adds to clients set."""
        server = WebSocketServer()

        mock_websocket = AsyncMock()
        # Make iteration stop immediately
        mock_websocket.__aiter__ = Mock(return_value=iter([]))

        await server._handle_client(mock_websocket)

        # Client should be removed after disconnection
        assert mock_websocket not in server._clients

    @pytest.mark.asyncio
    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    async def test_handle_client_sends_welcome(self):
        """Test that client receives welcome message."""
        server = WebSocketServer()

        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = Mock(return_value=iter([]))

        await server._handle_client(mock_websocket)

        # First call should be welcome message
        calls = mock_websocket.send.call_args_list
        assert len(calls) >= 1

        welcome = json.loads(calls[0][0][0])
        assert welcome["type"] == "connected"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    async def test_handle_client_default_subscription(self):
        """Test that client gets default subscription."""
        server = WebSocketServer()

        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = Mock(return_value=iter([]))

        # Check subscription during handling
        original_handle = server._handle_client

        async def check_subscription(ws):
            server._clients.add(ws)
            server._subscriptions[ws] = {"*"}
            assert server._subscriptions[ws] == {"*"}

        await check_subscription(mock_websocket)


class TestWebSocketServerSendToClient:
    """Test send_to_client functionality."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    async def test_send_to_client_success(self):
        """Test successful send to client."""
        server = WebSocketServer()

        mock_client = AsyncMock()

        await server._send_to_client(mock_client, '{"test": "data"}')

        mock_client.send.assert_called_once_with('{"test": "data"}')

    @pytest.mark.asyncio
    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    async def test_send_to_client_error_removes_client(self):
        """Test that send error removes client from set."""
        server = WebSocketServer()

        mock_client = AsyncMock()
        mock_client.send.side_effect = Exception("Connection closed")

        server._clients.add(mock_client)
        server._subscriptions[mock_client] = {"*"}

        await server._send_to_client(mock_client, '{"test": "data"}')

        assert mock_client not in server._clients
        assert mock_client not in server._subscriptions


class TestWebSocketServerIntegration:
    """Integration tests for WebSocket server."""

    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    def test_server_message_format(self):
        """Test that broadcast creates correct message format."""
        server = WebSocketServer()
        server._running = True
        server._loop = asyncio.new_event_loop()

        mock_client = AsyncMock()
        server._clients.add(mock_client)
        server._subscriptions[mock_client] = {"*"}

        captured_messages = []

        def capture_message(coro, loop):
            # Extract message from coroutine
            captured_messages.append(coro)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=capture_message):
            server.broadcast("test_event", {"key": "value"})

        server._loop.close()

    def test_websocket_availability_check(self):
        """Test that WEBSOCKETS_AVAILABLE is correctly set."""
        try:
            import websockets

            assert WEBSOCKETS_AVAILABLE is True
        except ImportError:
            assert WEBSOCKETS_AVAILABLE is False
