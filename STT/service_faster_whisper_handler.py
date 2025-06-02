import queue
import logging
from typing import Any
from rich.console import Console

from baseHandler import BaseHandler
from services.stt_service import WhisperService

logger = logging.getLogger(__name__)
console = Console()


class ServiceFasterWhisperSTTHandler(BaseHandler):
    """STT handler using :class:`WhisperService` for shared inference."""

    def setup(
        self,
        model_name: str = "tiny.en",
        device: str = "auto",
        compute_type: str = "auto",
        gen_kwargs: dict | None = None,
    ) -> None:
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.service = WhisperService(model_name, device, compute_type, gen_kwargs or {})

    def process(self, audio: Any):
        self.service.submit(audio, self.queue.put)
        text = self.queue.get()
        if text:
            console.print(f"[yellow]USER: {text}")
            yield text
