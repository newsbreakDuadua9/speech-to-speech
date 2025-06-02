import threading
import queue
import logging
from typing import Callable, Any, Dict, Tuple

logger = logging.getLogger(__name__)


class TTSService:
    """Generic thread safe service for Text To Speech models."""

    _models: Dict[Tuple[str, Tuple[Any, ...], Tuple[Tuple[str, Any], ...]], Any] = {}
    _lock = threading.Lock()

    def __init__(self,
                 model_key: str,
                 loader: Callable[[], Any],
                 generate_fn: Callable[[Any, str], Any]):
        """Create or reuse a model instance.

        Parameters
        ----------
        model_key:
            Unique key identifying the model configuration.
        loader:
            Function used to instantiate the model if not already loaded.
        generate_fn:
            Function taking ``(model, text)`` and returning audio data.
        """
        self.generate_fn = generate_fn
        with self._lock:
            if model_key not in self._models:
                logger.info(f"Loading TTS model {model_key}")
                self._models[model_key] = loader()
            self.model = self._models[model_key]

        self._queue: "queue.Queue[Tuple[str, Callable[[Any], None]]]" = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, text: str, callback: Callable[[Any], None]) -> None:
        """Submit ``text`` for synthesis.

        ``callback`` is called with the generated audio when ready.
        """
        self._queue.put((text, callback))

    def close(self) -> None:
        self._stop.set()
        self._worker.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text, cb = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                audio = self.generate_fn(self.model, text)
                cb(audio)
            except Exception:  # pragma: no cover - logging only
                logger.exception("TTSService callback failed")
            finally:
                self._queue.task_done()
