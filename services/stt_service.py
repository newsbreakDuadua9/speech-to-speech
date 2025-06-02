import threading
import queue
import logging
from typing import Callable, Any, Dict, Tuple

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperService:
    """Thread safe service wrapper around :class:`WhisperModel`.

    A single model instance is shared across all ``WhisperService`` instances
    created with the same configuration. Requests are processed asynchronously
    by a dedicated worker thread.
    """

    _models: Dict[Tuple[str, str, str], WhisperModel] = {}
    _lock = threading.Lock()

    def __init__(self,
                 model_name: str = "tiny.en",
                 device: str = "auto",
                 compute_type: str = "auto",
                 gen_kwargs: Dict[str, Any] | None = None) -> None:
        self.config = (model_name, device, compute_type)
        self.gen_kwargs = gen_kwargs or {}
        with self._lock:
            if self.config not in self._models:
                logger.info(f"Loading Whisper model {self.config}")
                self._models[self.config] = WhisperModel(model_name,
                                                         device=device,
                                                         compute_type=compute_type)
            self.model = self._models[self.config]

        self._queue: "queue.Queue[Tuple[Any, Callable[[str], None]]]" = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, audio: Any, callback: Callable[[str], None]) -> None:
        """Submit ``audio`` for transcription.

        ``callback`` is called with the resulting text once inference is done.
        """
        self._queue.put((audio, callback))

    def close(self) -> None:
        self._stop.set()
        self._worker.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                audio, cb = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                segments, _ = self.model.transcribe(audio, **self.gen_kwargs)
                text = " ".join(s.text for s in segments).strip()
                cb(text)
            except Exception:  # pragma: no cover - logging only
                logger.exception("WhisperService callback failed")
            finally:
                self._queue.task_done()
