import os
import asyncio
import logging
import signal
import time
from dotenv import load_dotenv

from speech_to_text import SpeechRecognizer
from text_to_text import AIProcessor
from text_to_speech import SpeechSynthesizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()

is_running = True
session_start_time = None


def signal_handler(sig, frame):
    global is_running
    logger.info("Shutdown signal received")
    is_running = False


class VoiceAISystem:
    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.ai_processor = AIProcessor()
        self.speech_synthesizer = SpeechSynthesizer()
        self.transcription_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()

    async def setup(self):
        await self.speech_recognizer.setup(self.transcription_queue)
        self.ai_processor.setup(self.transcription_queue, self.response_queue)
        self.speech_synthesizer.setup(self.response_queue)

    async def start(self):
        global session_start_time
        session_start_time = time.time()
        
        await self.speech_recognizer.calibrate_silence_threshold()
        
        try:
            if not await self.speech_recognizer.start_recognition():
                logger.error("Failed to start speech recognition")
                return
                
            processor_task = asyncio.create_task(self.ai_processor.process_transcriptions())
            synthesizer_task = asyncio.create_task(self.speech_synthesizer.generate_speech())
            
            while is_running:
                await asyncio.sleep(0.1)
                if not self.speech_recognizer.is_recording:
                    logger.warning("Speech recognition stopped unexpectedly, attempting to restart...")
                    if not await self.speech_recognizer.start_recognition():
                       logger.error("Failed to restart speech recognition")
                       break
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            await self.stop()
            
            for task in [t for t in asyncio.all_tasks() 
                        if t is not asyncio.current_task() and not t.done()]:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Task {task.get_name()} could not be canceled gracefully")
                except Exception as e:
                    logger.error(f"Error canceling task: {e}")
                
            self._report_timing_stats()

    def _report_timing_stats(self):
        global session_start_time
        total_session_time = time.time() - session_start_time
        
        print("\n===== API PERFORMANCE REPORT =====")
        print(f"Total session time: {total_session_time:.2f} seconds")
        
        dg_stats = self.speech_recognizer.get_timing_stats()
        print("\n--- Deepgram Speech-to-Text API ---")
        print(f"Total transcription calls: {dg_stats['total_calls']}")
        print(f"Average transcription time: {dg_stats['average_time']:.2f} seconds")
        print(f"Maximum transcription time: {dg_stats['max_time']:.2f} seconds")
        print(f"Interim results: {dg_stats['interim_results']}")
        print(f"Final results: {dg_stats['final_results']}")
        
        ai_stats = self.ai_processor.get_timing_stats()
        print("\n--- Google Gemini Text-to-Text API ---")
        print(f"Total API calls: {ai_stats['total_calls']}")
        print(f"Average response time: {ai_stats['average_time']:.2f} seconds")
        print(f"Maximum response time: {ai_stats['max_time']:.2f} seconds")
        
        tts_stats = self.speech_synthesizer.get_timing_stats()
        print("\n--- ElevenLabs Text-to-Speech API ---")
        print(f"Total API calls: {tts_stats['total_calls']}")
        print(f"Average response time: {tts_stats['average_time']:.2f} seconds")
        print(f"Maximum response time: {tts_stats['max_time']:.2f} seconds")
        print(f"Total download time: {tts_stats['download_time']:.2f} seconds")
        print(f"Total playback time: {tts_stats['playback_time']:.2f} seconds")
        
        print("\n=================================")

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