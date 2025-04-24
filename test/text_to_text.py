import os
import asyncio
import anthropic  # type: ignore
import logging

logger = logging.getLogger(__name__)


class AIProcessor:
    def __init__(self):
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        self.context = []
        self.transcription_queue = None
        self.response_queue = None
        self.is_running = True

        if not self.anthropic_api_key:
            logger.error("Anthropic API key not found in environment variables")

    def setup(self, transcription_queue, response_queue):
        self.transcription_queue = transcription_queue
        self.response_queue = response_queue
        self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        logger.info("AI processor setup complete")

    async def process_transcriptions(self):
        while self.is_running:
            try:
                transcript = await self.transcription_queue.get()
                await self._process_with_claude(transcript)
                self.transcription_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing transcription: {e}")
                await asyncio.sleep(0.1)

    async def _process_with_claude(self, text):
        try:
            logger.info(f"Processing with Claude: {text}")
            self.context.append({"role": "Teacher", "content": text})
            if len(self.context) > 10:
                self.context = self.context[-10:]

            response_text = ""
            with self.client.messages.stream(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1024,
                messages=self.context,
            ) as stream:
                for text in stream.text_stream:
                    response_text += text
                    print(text, end="", flush=True)

            logger.info(f"Claude response: {response_text}")
            self.context.append({"role": "assistant", "content": response_text})
            await self.response_queue.put(response_text)

        except Exception as e:
            logger.error(f"Error processing with Claude: {e}")

    def stop(self):
        self.is_running = False
        logger.info("AI processor stopped")
