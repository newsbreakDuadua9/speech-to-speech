import socket
import threading
import logging
from typing import Tuple

from VAD.vad_handler import VADHandler
from AEC.livekit_aec_handler import LivekitAecHandler
from resample_handler import ResampleHandler

from connections.stream_connection import ConnectionReceiver, ConnectionSender
from utils.thread_manager import ThreadManager
from STT.service_faster_whisper_handler import ServiceFasterWhisperSTTHandler
from TTS.service_melo_handler import ServiceMeloTTSHandler
from s2s_pipeline import (
    parse_arguments,
    prepare_all_args,
    get_llm_handler,
    rename_args,
    optimal_mac_settings,
    check_mac_settings,
    overwrite_device_argument,
    initialize_queues_and_events,
)

logger = logging.getLogger(__name__)


class ClientHandler(threading.Thread):
    def __init__(self, conn: socket.socket, args: Tuple):
        super().__init__(daemon=True)
        (
            module_kwargs,
            socket_receiver_kwargs,
            socket_sender_kwargs,
            vad_handler_kwargs,
            whisper_stt_handler_kwargs,
            paraformer_stt_handler_kwargs,
            faster_whisper_stt_handler_kwargs,
            language_model_handler_kwargs,
            open_api_language_model_handler_kwargs,
            mlx_language_model_handler_kwargs,
            parler_tts_handler_kwargs,
            melo_tts_handler_kwargs,
            chat_tts_handler_kwargs,
            facebook_mms_tts_handler_kwargs,
        ) = args
        self.conn = conn
        self.kwargs = dict(
            module_kwargs=module_kwargs,
            vad_handler_kwargs=vad_handler_kwargs,
            faster_whisper_stt_handler_kwargs=faster_whisper_stt_handler_kwargs,
            language_model_handler_kwargs=language_model_handler_kwargs,
            open_api_language_model_handler_kwargs=open_api_language_model_handler_kwargs,
            mlx_language_model_handler_kwargs=mlx_language_model_handler_kwargs,
            melo_tts_handler_kwargs=melo_tts_handler_kwargs,
        )

    def run(self):
        q = initialize_queues_and_events()
        stop_event = q["stop_event"]
        should_listen = q["should_listen"]
        interrupt_event = q["interrupt_event"]

        receiver = ConnectionReceiver(
            stop_event,
            q["recv_audio_chunks_queue"],
            should_listen,
            self.conn,
            chunk_size=1024,
        )
        sender = ConnectionSender(stop_event, q["send_audio_chunks_queue"], self.conn)
        aec = LivekitAecHandler(
            stop_event,
            queue_in=q["recv_audio_chunks_queue"],
            queue_out=q["aec_to_vad_queue"],
        )
        vad = VADHandler(
            stop_event,
            queue_in=q["aec_to_vad_queue"],
            queue_out=q["spoken_prompt_queue"],
            interrupt_event=interrupt_event,
            setup_args=(should_listen,),
            setup_kwargs=vars(self.kwargs["vad_handler_kwargs"]),
        )
        stt = ServiceFasterWhisperSTTHandler(
            stop_event,
            queue_in=q["spoken_prompt_queue"],
            queue_out=q["text_prompt_queue"],
            setup_kwargs=vars(self.kwargs["faster_whisper_stt_handler_kwargs"]),
        )
        lm = get_llm_handler(
            self.kwargs["module_kwargs"],
            stop_event,
            q["text_prompt_queue"],
            q["lm_response_queue"],
            interrupt_event,
            self.kwargs["language_model_handler_kwargs"],
            self.kwargs["open_api_language_model_handler_kwargs"],
            self.kwargs["mlx_language_model_handler_kwargs"],
        )
        tts = ServiceMeloTTSHandler(
            stop_event,
            queue_in=q["lm_response_queue"],
            queue_out=q["send_audio_chunks_queue"],
            interrupt_event=interrupt_event,
            setup_args=(should_listen,),
            setup_kwargs=vars(self.kwargs["melo_tts_handler_kwargs"]),
        )
        manager = ThreadManager([receiver, sender, aec, vad, stt, lm, tts])
        manager.start()
        for t in manager.threads:
            t.join()
        self.conn.close()


def main():
    args = parse_arguments()
    prepare_all_args(*args)
    module_kwargs = args[0]
    host = "0.0.0.0"
    port = 23456
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen()
    logger.info("Multi-user server listening on %s:%s", host, port)
    try:
        while True:
            conn, _ = server.accept()
            logger.info("New client connected")
            ClientHandler(conn, args).start()
    finally:
        server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
