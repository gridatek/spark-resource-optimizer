"""WebSocket server for real-time monitoring updates."""

from typing import Dict, Set, Optional, Any
import threading
import json
import logging
import asyncio

logger = logging.getLogger(__name__)

# Try to import websockets library
try:
    import websockets
    from websockets.server import serve

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore


class WebSocketServer:
    """WebSocket server for streaming real-time updates to clients.

    Provides real-time push of:
    - Application status changes
    - Metric updates
    - Alerts and notifications
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = 8765
    ):  # Changed from 0.0.0.0 for security
        """Initialize the WebSocket server.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        if not WEBSOCKETS_AVAILABLE:
            logger.warning(
                "websockets library not available. "
                "Install with: pip install websockets"
            )

        self.host = host
        self.port = port
        self._clients: Set[Any] = set()
        self._subscriptions: Dict[Any, Set[str]] = {}
        self._running = False
        self._server: Optional[Any] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error(
                "Cannot start WebSocket server: websockets library not available"
            )
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False

        if self._loop and self._server:
            # Schedule server close on the event loop
            asyncio.run_coroutine_threadsafe(self._close_server(), self._loop)

        if self._thread:
            self._thread.join(timeout=5)

        logger.info("WebSocket server stopped")

    def broadcast(self, event_type: str, data: Dict) -> None:
        """Broadcast an event to all connected clients.

        Args:
            event_type: Type of event
            data: Event data
        """
        if not self._running or not self._loop:
            return

        message = json.dumps({"type": event_type, "data": data})

        with self._lock:
            clients_to_notify = list(self._clients)

        for client in clients_to_notify:
            # Check if client is subscribed to this event type
            subscriptions = self._subscriptions.get(client, set())
            if not subscriptions or event_type in subscriptions or "*" in subscriptions:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._send_to_client(client, message), self._loop
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")

    def send_to_client(self, client: Any, event_type: str, data: Dict) -> None:
        """Send an event to a specific client.

        Args:
            client: WebSocket client connection
            event_type: Type of event
            data: Event data
        """
        if not self._running or not self._loop:
            return

        message = json.dumps({"type": event_type, "data": data})

        try:
            asyncio.run_coroutine_threadsafe(
                self._send_to_client(client, message), self._loop
            )
        except Exception as e:
            logger.error(f"Error sending to client: {e}")

    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        with self._lock:
            return len(self._clients)

    def _run_server(self) -> None:
        """Run the WebSocket server event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        """Start serving WebSocket connections."""
        if not WEBSOCKETS_AVAILABLE:
            return

        async with serve(self._handle_client, self.host, self.port) as server:
            self._server = server
            # Keep running until stopped
            while self._running:
                await asyncio.sleep(0.1)

    async def _close_server(self) -> None:
        """Close the server gracefully."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(self, websocket: Any) -> None:
        """Handle a client connection.

        Args:
            websocket: WebSocket connection
        """
        with self._lock:
            self._clients.add(websocket)
            self._subscriptions[websocket] = {"*"}  # Subscribe to all by default

        logger.info(f"Client connected. Total clients: {len(self._clients)}")

        try:
            # Send welcome message
            await websocket.send(
                json.dumps(
                    {
                        "type": "connected",
                        "data": {"message": "Connected to Spark Resource Optimizer"},
                    }
                )
            )

            # Handle incoming messages
            async for message in websocket:
                await self._handle_message(websocket, message)

        except Exception as e:
            logger.debug(f"Client connection closed: {e}")
        finally:
            with self._lock:
                self._clients.discard(websocket)
                self._subscriptions.pop(websocket, None)

            logger.info(f"Client disconnected. Total clients: {len(self._clients)}")

    async def _handle_message(self, websocket: Any, message: str) -> None:
        """Handle an incoming message from a client.

        Args:
            websocket: WebSocket connection
            message: Message received from client
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "subscribe":
                # Subscribe to specific event types
                events = data.get("events", [])
                with self._lock:
                    self._subscriptions[websocket] = set(events)
                await websocket.send(
                    json.dumps({"type": "subscribed", "data": {"events": events}})
                )

            elif msg_type == "unsubscribe":
                # Unsubscribe from specific event types
                events = data.get("events", [])
                with self._lock:
                    current = self._subscriptions.get(websocket, set())
                    for event in events:
                        current.discard(event)
                    self._subscriptions[websocket] = current

            elif msg_type == "ping":
                # Respond to ping
                await websocket.send(json.dumps({"type": "pong"}))

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _send_to_client(self, client: Any, message: str) -> None:
        """Send a message to a client.

        Args:
            client: WebSocket connection
            message: Message to send
        """
        try:
            await client.send(message)
        except Exception as e:
            logger.debug(f"Error sending to client: {e}")
            with self._lock:
                self._clients.discard(client)
                self._subscriptions.pop(client, None)
