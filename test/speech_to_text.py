import os
import asyncio
import websockets
import json
import pyaudio  # type: ignore
import logging
import numpy as np  # type: ignore
import time

logger = logging.getLogger(__name__)

AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4096


class SpeechRecognizer:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.deepgram_ws = None
        self.is_recording = False
        self.audio_queue = asyncio.Queue()
        self.transcription_queue = None
        self.silence_threshold = 300
        self.silence_duration = 0
        self.max_silence_frames = 15
        self.speaking_detected = False
        self.current_speech_segment = []
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

        if not self.deepgram_api_key:
            logger.error("Deepgram API key not found in environment variables")

    async def setup(self, transcription_queue):
        self.transcription_queue = transcription_queue
        logger.info("Speech recognizer setup complete")

    async def calibrate_silence_threshold(self):
        print("Calibrating microphone for ambient noise...")
        print("Please remain silent...")

        stream = self.audio.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        ambient_data = []
        for i in range(0, int(RATE / CHUNK * 2)):
            data = stream.read(CHUNK)
            ambient_data.append(np.frombuffer(data, dtype=np.int16))

        stream.stop_stream()
        stream.close()

        ambient_noise = np.mean(np.abs(np.concatenate(ambient_data)))

        self.silence_threshold = ambient_noise * 2.5

        print(f"Ambient noise level: {ambient_noise}")
        print(f"Silence threshold set to: {self.silence_threshold}")

    async def connect_to_deepgram(self):
        """Connect to Deepgram's streaming API"""
        deepgram_url = f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate={RATE}&channels={CHANNELS}&model=nova-2&punctuate=true&vad_events=true&interim_results=false"

        headers = {"Authorization": f"Token {self.deepgram_api_key}"}

        try:
            self.deepgram_ws = await websockets.connect(
                deepgram_url, extra_headers=headers
            )
            logger.info("Connected to Deepgram")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            return False

    async def start_recognition(self):
        if not await self.connect_to_deepgram():
            logger.error(
                "Failed to start speech recognition: couldn't connect to Deepgram"
            )
            return

        self.is_recording = True

        def audio_callback(in_data, frame_count, time_info, status):
            if self.is_recording:
                self.audio_queue.put_nowait(in_data)

                audio_data = np.frombuffer(in_data, dtype=np.int16)
                volume_norm = np.abs(audio_data).mean()

                if volume_norm > self.silence_threshold:
                    self.silence_duration = 0
                    if not self.speaking_detected:
                        self.speaking_detected = True
                        self.current_speech_segment = []
                        logger.info("Speech started")
                    self.current_speech_segment.append(in_data)
                else:
                    if self.speaking_detected:
                        self.current_speech_segment.append(in_data)
                        self.silence_duration += 1

                        if self.silence_duration >= self.max_silence_frames:
                            self.speaking_detected = False
                            logger.info("Speech ended - processing segment")
                            self.silence_duration = 0
                            self.current_speech_segment = []

            return (in_data, pyaudio.paContinue)

        self.stream = self.audio.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=audio_callback,
        )

        logger.info("Started recording audio")
        self.stream.start_stream()

        await self._transcribe_audio()

    async def _transcribe_audio(self):
        receive_task = asyncio.create_task(self._receive_transcription())
        try:
            while self.is_recording:
                if not self.audio_queue.empty():
                    chunk = await self.audio_queue.get()
                    await self.deepgram_ws.send(chunk)
                    await asyncio.sleep(0.01)
                else:
                    await asyncio.sleep(0.01)

            if self.deepgram_ws:
                await self.deepgram_ws.send(json.dumps({"type": "CloseStream"}))
                receive_task.cancel()

        except Exception as e:
            logger.error(f"Error in transcription stream: {e}")
            if receive_task:
                receive_task.cancel()

    async def _receive_transcription(self):
        try:
            async for msg in self.deepgram_ws:
                result = json.loads(msg)
                if "type" in result and result["type"] == "VADEvent":
                    if result["event"] == "speech_start":
                        logger.info("Deepgram detected speech start")
                    elif result["event"] == "speech_end":
                        logger.info("Deepgram detected speech end")

                elif "channel" in result:
                    transcript = result["channel"]["alternatives"][0].get(
                        "transcript", ""
                    )
                    if transcript.strip():
                        logger.info(f"Transcribed: {transcript}")

                        await self.transcription_queue.put(transcript)

        except Exception as e:
            logger.error(f"Error receiving transcription: {e}")

    async def stop_recognition(self):
        """Stop recording and recognizing speech"""
        self.is_recording = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            logger.info("Recording stopped")

        if self.deepgram_ws:
            try:
                await self.deepgram_ws.send(json.dumps({"type": "CloseStream"}))
                await asyncio.sleep(0.5)
                await self.deepgram_ws.close()
                logger.info("Closed Deepgram connection")
            except Exception as e:
                logger.error(f"Error closing Deepgram connection: {e}")
