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
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice ID
        self.api_timing_stats = {
            "total_calls": 0,
            "total_time": 0,
            "average_time": 0,
            "max_time": 0,
            "download_time": 0,
            "playback_time": 0,
        }

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
            start_time = time.time()
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
                    download_start = time.time()
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".mp3"
                    ) as temp_file:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                temp_file.write(chunk)
                        temp_file_path = temp_file.name
                    
                    download_time = time.time() - download_start
                    self.api_timing_stats["download_time"] += download_time
                    
                    audio_load_start = time.time()
                    audio = AudioSegment.from_mp3(temp_file_path)
                    audio_load_time = time.time() - audio_load_start
                    
                    # Calculate API response time (not including playback)
                    elapsed_time = time.time() - start_time
                    self.api_timing_stats["total_calls"] += 1
                    self.api_timing_stats["total_time"] += elapsed_time
                    self.api_timing_stats["average_time"] = self.api_timing_stats["total_time"] / self.api_timing_stats["total_calls"]
                    self.api_timing_stats["max_time"] = max(self.api_timing_stats["max_time"], elapsed_time)
                    
                    logger.info(f"ElevenLabs API timing: {elapsed_time:.2f}s (avg: {self.api_timing_stats['average_time']:.2f}s, max: {self.api_timing_stats['max_time']:.2f}s)")
                    logger.info(f"Download time: {download_time:.2f}s, Audio load time: {audio_load_time:.2f}s")
                    
                    logger.info(f"Playing audio response (duration: {audio.duration_seconds:.2f}s)")
                    threading.Thread(target=play, args=(audio,)).start()
                    
                    # Add to playback time stats
                    self.api_timing_stats["playback_time"] += audio.duration_seconds

                    def cleanup_temp_file():
                        time.sleep(audio.duration_seconds + 1)
                        try:
                            os.unlink(temp_file_path)
                        except Exception as e:
                            logger.error(f"Error cleaning up temp file: {e}")

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

    def get_timing_stats(self):
        return self.api_timing_stats

    def stop(self):
        self.is_running = False
        logger.info("Speech synthesizer stopped")
        logger.info(f"Final ElevenLabs API timing stats: {self.api_timing_stats}")