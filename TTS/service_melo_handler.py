import queue
import logging
import librosa
import numpy as np
from typing import Any, Tuple
from rich.console import Console

from melo.api import TTS
from baseHandler import BaseHandler
from services.tts_service import TTSService

logger = logging.getLogger(__name__)
console = Console()

WHISPER_LANGUAGE_TO_MELO_LANGUAGE = {
    "en": "EN",
    "fr": "FR",
    "es": "ES",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}

WHISPER_LANGUAGE_TO_MELO_SPEAKER = {
    "en": "EN-BR",
    "fr": "FR",
    "es": "ES",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}


class ServiceMeloTTSHandler(BaseHandler):
    """TTS handler using :class:`TTSService` for shared Melo models."""

    def setup(
        self,
        should_listen,
        device: str = "mps",
        language: str = "en",
        speaker_to_id: str = "en",
        blocksize: int = 512,
    ) -> None:
        self.should_listen = should_listen
        self.device = device
        self.blocksize = blocksize
        self.language = language
        self.speaker_to_id = speaker_to_id
        self.service = self._create_service(language, speaker_to_id)
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def _create_service(self, language: str, speaker_to_id: str) -> TTSService:
        def loader() -> Tuple[TTS, int]:
            model = TTS(
                language=WHISPER_LANGUAGE_TO_MELO_LANGUAGE[language],
                device=self.device,
            )
            spk = model.hps.data.spk2id[WHISPER_LANGUAGE_TO_MELO_SPEAKER[speaker_to_id]]
            return model, spk

        def generate(model_tuple: Tuple[TTS, int], text: str) -> np.ndarray:
            model, spk = model_tuple
            audio = model.tts_to_file(text, spk, quiet=True)
            audio = librosa.resample(audio, orig_sr=44100, target_sr=16000)
            return (audio * 32768).astype(np.int16)

        return TTSService(("melo", language, self.device), loader, generate)

    def process(self, llm_sentence: Any):
        language_code = None
        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence
        if language_code and language_code != self.language:
            self.language = language_code
            self.service = self._create_service(language_code, language_code)
        self.service.submit(str(llm_sentence), self.queue.put)
        audio = self.queue.get()
        if len(audio) == 0:
            self.should_listen.set()
            return
        for i in range(0, len(audio), self.blocksize):
            chunk = audio[i : i + self.blocksize]
            if len(chunk) < self.blocksize:
                chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
            yield chunk
        self.should_listen.set()
