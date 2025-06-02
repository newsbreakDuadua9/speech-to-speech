import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ConnectionReceiver:
    """Read fixed-size audio chunks from an existing socket."""

    def __init__(self, stop_event, queue_out, should_listen, conn, chunk_size: int = 1024):
        self.stop_event = stop_event
        self.queue_out = queue_out
        self.should_listen = should_listen
        self.conn = conn
        self.chunk_size = chunk_size

    def receive_full_chunk(self) -> Optional[bytes]:
        data = b""
        while len(data) < self.chunk_size:
            packet = self.conn.recv(self.chunk_size - len(data))
            if not packet:
                return None
            data += packet
        return data

    def run(self):
        self.should_listen.set()
        while not self.stop_event.is_set():
            audio_chunk = self.receive_full_chunk()
            if audio_chunk is None:
                self.stop_event.set()
                self.queue_out.put(b"END")
                break
            if self.should_listen.is_set():
                self.queue_out.put(audio_chunk)
        logger.info("Receiver closed")


class ConnectionSender:
    """Send audio chunks from a queue over a socket."""

    def __init__(self, stop_event, queue_in, conn):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.conn = conn

    def run(self):
        while not self.stop_event.is_set():
            audio_chunk = self.queue_in.get()
            try:
                self.conn.sendall(audio_chunk)
            except Exception:
                self.stop_event.set()
                break
            if isinstance(audio_chunk, bytes) and audio_chunk == b"END":
                break
        logger.info("Sender closed")
