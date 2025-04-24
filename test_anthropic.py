import os
from dotenv import load_dotenv  # type: ignore
from deepgram.utils import verboselogs
from datetime import datetime
import httpx

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)

load_dotenv()

AUDIO_FILE = "./English-recording.wav"

deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")


def main():
    try:
        config: DeepgramClientOptions = DeepgramClientOptions(
            verbose=verboselogs.SPAM,
        )
        deepgram: DeepgramClient = DeepgramClient(deepgram_api_key, config)

        with open(AUDIO_FILE, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {"buffer": buffer_data}

        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
            utterances=True,
            punctuate=True,
            diarize=True,
        )

        before = datetime.now()

        response = deepgram.listen.rest.v("1").transcribe_file(
            payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
        )

        after = datetime.now()

        print(response.to_json(indent=4))
        print("")
        difference = after - before
        print(f"time: {difference.seconds}")
    except Exception as error:
        print(f"Exception: {error}")


if __name__ == "__main__":
    main()
