import os
import asyncio
import logging
import signal
from dotenv import load_dotenv
from speech_to_text import SpeechRecognizer
from text_to_text import AIProcessor  # type: ignore
from text_to_speech import SpeechSynthesizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()

is_running = True


def signal_handler(sig, frame):
    global is_running
    logger.info("Shutdown signal received")
    is_running = False


class VoiceAISystem:
    def __init__(self):
        # First we create all system components
        self.speech_recognizer = SpeechRecognizer()
        self.ai_processor = AIProcessor()
        self.speech_synthesizer = SpeechSynthesizer()
        # Intitialize queues for transcriptions and responses
        self.transcription_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()

    async def setup(self):
        await self.speech_recognizer.setup(self.transcription_queue)
        self.ai_processor.setup(self.transcription_queue, self.response_queue)
        self.speech_synthesizer.setup(self.response_queue)

    async def start(self):
        await self.speech_recognizer.calibrate_silence_threshold()
        speech_task = asyncio.create_task(self.speech_recognizer.start_recognition())
        processor_task = asyncio.create_task(self.ai_processor.process_transcriptions())
        synthesizer_task = asyncio.create_task(
            self.speech_synthesizer.generate_speech()
        )
        while is_running:
            await asyncio.sleep(0.1)
        await self.stop()
        speech_task.cancel()
        processor_task.cancel()
        synthesizer_task.cancel()

        try:
            await speech_task
        except asyncio.CancelledError:
            pass
        try:
            await processor_task
        except asyncio.CancelledError:
            pass
        try:
            await synthesizer_task
        except asyncio.CancelledError:
            pass

    async def stop(self):
        await self.speech_recognizer.stop_recognition()
        self.ai_processor.stop()
        self.speech_synthesizer.stop()


async def main():
    signal.signal(signal.SIGINT, signal_handler)
    print("=== Modular Voice AI System ===")
    print("This system will record continuously and process speech as you talk")
    print("Press Enter to start, and Ctrl+C to stop at any time")

    input()
    system = VoiceAISystem()
    await system.setup()
    await system.start()
    print("System stopped")


if __name__ == "__main__":
    asyncio.run(main())
