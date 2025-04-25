import os
import asyncio
import pyaudio #type: ignore
import logging
import time
import numpy as np #type: ignore
from signal import SIGINT, SIGTERM
from dotenv import load_dotenv #type: ignore

from deepgram import ( #type: ignore
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeechRecognizer:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.is_finals = []
        self.deepgram_client = None
        self.dg_connection = None
        self.microphone = None
        self.is_recording = False
        self.transcription_queue = None
        self.silence_threshold = 300
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        self.api_timing_stats = {
            "total_calls": 0,
            "total_time": 0,
            "average_time": 0,
            "max_time": 0,
            "interim_results": 0,
            "final_results": 0,
        }
        self.current_utterance_start = None
        self.keep_alive_task = None

        if not self.api_key:
            logger.error("DEEPGRAM_API_KEY not found in environment variables")
            raise ValueError("DEEPGRAM_API_KEY environment variable is required")

    async def setup(self, transcription_queue):
        self.transcription_queue = transcription_queue
        config = DeepgramClientOptions(options={"keepalive": "true"})
        self.deepgram_client = DeepgramClient(self.api_key, config)
        logger.info("Speech to text setup complete with queue")

    async def calibrate_silence_threshold(self):
        print("Calibrating microphone for ambient noise...")
        print("Please remain silent...")

        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4096,
        )

        ambient_data = []
        for i in range(0, int(16000 / 4096 * 2)):
            data = stream.read(4096)
            ambient_data.append(np.frombuffer(data, dtype=np.int16))

        stream.stop_stream()
        stream.close()

        ambient_noise = np.mean(np.abs(np.concatenate(ambient_data)))
        self.silence_threshold = ambient_noise * 2.5

        print(f"Ambient noise level: {ambient_noise}")
        print(f"Silence threshold set to: {self.silence_threshold}")
        
        return self.silence_threshold

    async def start_recognition(self):
        try:
            loop = asyncio.get_event_loop()

            self.dg_connection = self.deepgram_client.listen.asyncwebsocket.v("1")
            
            self.dg_connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
            self.dg_connection.on(LiveTranscriptionEvents.Metadata, self._on_metadata)
            self.dg_connection.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
            self.dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
            self.dg_connection.on(LiveTranscriptionEvents.Close, self._on_close)
            self.dg_connection.on(LiveTranscriptionEvents.Error, self._on_error)

            options = LiveOptions(
                model="nova-3",
                language="en-US",
                smart_format=True,
                encoding="linear16",
                channels=1,
                sample_rate=16000,
                interim_results=True,
                utterance_end_ms="1000",
                vad_events=True,
                endpointing=300,
            )

            addons = {
                "no_delay": "true"
            }

            logger.info("Starting transcription connection to Deepgram")
            if await self.dg_connection.start(options, addons=addons) is False:
                logger.error("Failed to connect to Deepgram")
                return False

            self.is_recording = True
            self.keep_alive_task = asyncio.create_task(self._keep_connection_alive())

            self.microphone = Microphone(self.dg_connection.send)
            self.microphone.start()
            logger.info("Microphone started. Begin speaking...")
            
            return True
            
        except Exception as e:
            logger.error(f"Could not start transcription: {e}")
            return False

    async def _keep_connection_alive(self):
        while self.is_recording:
            try:
                await asyncio.sleep(5)
                if self.dg_connection and self.is_recording:
                    logger.debug("Connection monitoring: still recording")
            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")
                await asyncio.sleep(1)

    async def _on_open(self, *args, **kwargs):
        logger.info("Connected to Deepgram")

    async def _on_message(self, self_obj, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if len(sentence) == 0:
            return
            
        if result.is_final:
            end_time = time.time()
            self.api_timing_stats["final_results"] += 1
            
            if self.current_utterance_start is not None:
                elapsed_time = end_time - self.current_utterance_start
                self.api_timing_stats["total_calls"] += 1
                self.api_timing_stats["total_time"] += elapsed_time
                self.api_timing_stats["average_time"] = self.api_timing_stats["total_time"] / self.api_timing_stats["total_calls"]
                self.api_timing_stats["max_time"] = max(self.api_timing_stats["max_time"], elapsed_time)
                
                logger.info(f"Deepgram transcription timing: {elapsed_time:.2f}s (avg: {self.api_timing_stats['average_time']:.2f}s)")
            
            logger.info(f"Transcribed: {sentence}")
            await self.transcription_queue.put(sentence)
            self.current_utterance_start = None
        else:
            if self.current_utterance_start is None:
                self.current_utterance_start = time.time()
            self.api_timing_stats["interim_results"] += 1
            logger.debug(f"Interim: {sentence}")

    async def _on_metadata(self, *args, **kwargs):
        logger.info("Received metadata")

    async def _on_speech_started(self, *args, **kwargs):
        logger.info("Deepgram detected speech start")
        if self.current_utterance_start is None:
            self.current_utterance_start = time.time()

    async def _on_utterance_end(self, *args, **kwargs):
        logger.info("Deepgram detected speech end")
        await self.transcription_queue.put("__UTTERANCE_END__")
        self.current_utterance_start = None

    async def _on_close(self, *args, **kwargs):
        logger.info("Deepgram connection closed")

    async def _on_error(self, self_obj, error, **kwargs):
        logger.error(f"Deepgram error: {error}")

    def get_timing_stats(self):
        return self.api_timing_stats

    async def stop_recognition(self):
        logger.info("Stopping transcription")
        self.is_recording = False
        
        if hasattr(self, 'keep_alive_task') and self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        if self.microphone:
            self.microphone.finish()
            logger.info("Recording stopped")
        
        if self.dg_connection:
            await self.dg_connection.finish()
            logger.info("Closed Deepgram connection")
            logger.info(f"Final Deepgram API timing stats: {self.api_timing_stats}")