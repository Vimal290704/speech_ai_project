# text_to_speech.py
import os
import asyncio
import requests  # type: ignore
import tempfile
import threading
import time
import logging
from pydub import AudioSegment  # type:ignore
from pydub.playback import play  # type:ignore

logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    def __init__(self):
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.response_queue = None
        self.is_running = True
        self.voice_id = (
            "21m00Tcm4TlvDq8ikWAM"  # We need to change the model after testing
        )

        if not self.elevenlabs_api_key:
            logger.error("ElevenLabs API key not found in environment variables")

    def setup(self, response_queue):
        self.response_queue = response_queue
        logger.info("Speech synthesizer setup complete")

    async def generate_speech(self):
        """Generate speech from text in the queue"""
        while self.is_running:
            try:
                response_text = await self.response_queue.get()
                await self._synthesize_speech(response_text)
                self.response_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating speech: {e}")
                await asyncio.sleep(0.1)

    async def _synthesize_speech(self, text):
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"

            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key,
            }

            data = {
                "text": text,
                "model_id": "eleven_turbo_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            }
            with requests.post(
                url, json=data, headers=headers, stream=True
            ) as response:
                if response.status_code == 200:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".mp3"
                    ) as temp_file:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                temp_file.write(chunk)
                        temp_file_path = temp_file.name
                    audio = AudioSegment.from_mp3(temp_file_path)
                    logger.info(f"Playing audio response")
                    threading.Thread(target=play, args=(audio,)).start()

                    def cleanup_temp_file():
                        time.sleep(audio.duration_seconds + 1)
                        os.unlink(temp_file_path)

                    threading.Thread(target=cleanup_temp_file).start()
                else:
                    logger.error(
                        f"Error with ElevenLabs API: {response.status_code}, {response.text}"
                    )

        except Exception as e:
            logger.error(f"Error generating speech: {e}")

    def set_voice(self, voice_id):
        self.voice_id = voice_id
        logger.info(f"Voice changed to {voice_id}")

    def stop(self):
        self.is_running = False
        logger.info("Speech synthesizer stopped")
