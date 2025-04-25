import os
import asyncio
import time
import google.generativeai as genai  # type: ignore
import logging

logger = logging.getLogger(__name__)


class AIProcessor:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.client = None
        self.model = None
        self.context = []
        self.transcription_queue = None
        self.response_queue = None
        self.is_running = True
        self.api_timing_stats = {
            "total_calls": 0,
            "total_time": 0,
            "average_time": 0,
            "max_time": 0,
        }

        if not self.google_api_key:
            logger.error("Google API key not found in environment variables")

    def setup(self, transcription_queue, response_queue):
        self.transcription_queue = transcription_queue
        self.response_queue = response_queue
        genai.configure(api_key=self.google_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        logger.info("AI processor setup complete")

    async def process_transcriptions(self):
        while self.is_running:
            try:
                transcript = await self.transcription_queue.get()
                
                # Skip processing if it's an utterance marker
                if transcript == "__UTTERANCE_END__":
                    self.transcription_queue.task_done()
                    continue
                    
                await self._process_with_gemini(transcript)
                self.transcription_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing transcription: {e}")
                await asyncio.sleep(0.1)

    async def _process_with_gemini(self, text):
        try:
            logger.info(f"Processing with Gemini: {text}")
            start_time = time.time()
            
            gemini_messages = []
            for msg in self.context:
                role = "user" if msg["role"] == "Teacher" else "model"
                gemini_messages.append({"role": role, "parts": [msg["content"]]})
            
            gemini_messages.append({"role": "user", "parts": [text]})
            
            self.context.append({"role": "Teacher", "content": text})
            if len(self.context) > 10:
                self.context = self.context[-10:]
            
            chat = self.model.start_chat(history=gemini_messages[:-1])
            response = await asyncio.to_thread(chat.send_message, text)
            
            response_text = response.text
            
            # Calculate timing
            elapsed_time = time.time() - start_time
            self.api_timing_stats["total_calls"] += 1
            self.api_timing_stats["total_time"] += elapsed_time
            self.api_timing_stats["average_time"] = self.api_timing_stats["total_time"] / self.api_timing_stats["total_calls"]
            self.api_timing_stats["max_time"] = max(self.api_timing_stats["max_time"], elapsed_time)
            
            logger.info(f"Gemini response: {response_text}")
            logger.info(f"Gemini API timing: {elapsed_time:.2f} seconds (avg: {self.api_timing_stats['average_time']:.2f}s, max: {self.api_timing_stats['max_time']:.2f}s)")
            
            self.context.append({"role": "assistant", "content": response_text})
            await self.response_queue.put(response_text)

        except Exception as e:
            logger.error(f"Error processing with Gemini: {e}")

    def get_timing_stats(self):
        return self.api_timing_stats

    def stop(self):
        self.is_running = False
        logger.info("AI processor stopped")
        logger.info(f"Final API timing stats: {self.api_timing_stats}")