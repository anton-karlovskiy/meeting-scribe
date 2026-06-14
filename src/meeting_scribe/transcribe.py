from openai import OpenAI


def transcribe_audio(audio_path: str) -> str:
    client = OpenAI()
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )
    return result
