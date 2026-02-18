"""Quick test: record 3 seconds of audio and transcribe via the LFM audio server."""

import base64
import io
import time

import numpy as np
import pyaudio
import soundfile as sf
from openai import OpenAI

SERVER_URL = "http://127.0.0.1:8142/v1"
SAMPLE_RATE = 16000
RECORD_SECONDS = 4


def record_audio(seconds=RECORD_SECONDS):
    print(f"Recording {seconds}s of audio... Speak now!")
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=1024,
    )
    frames = []
    for _ in range(0, int(SAMPLE_RATE / 1024 * seconds)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    pa.terminate()

    audio = np.frombuffer(b"".join(frames), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    print(f"Recorded {len(audio)/SAMPLE_RATE:.1f}s")
    return buf.read()


def transcribe(wav_bytes):
    client = OpenAI(base_url=SERVER_URL, api_key="dummy")
    encoded = base64.b64encode(wav_bytes).decode("utf-8")

    messages = [
        {"role": "system", "content": "Perform ASR."},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": encoded, "format": "wav"},
                }
            ],
        },
    ]

    print("Transcribing...")
    t0 = time.time()
    stream = client.chat.completions.create(
        model="",
        messages=messages,
        stream=True,
        max_tokens=512,
    )

    text_parts = []
    for chunk in stream:
        if chunk.choices[0].finish_reason == "stop":
            break
        delta = chunk.choices[0].delta
        if delta.content:
            text_parts.append(delta.content)
            print(delta.content, end="", flush=True)

    elapsed = time.time() - t0
    text = "".join(text_parts).strip()
    print(f"\n\nTranscription: {text}")
    print(f"Time: {elapsed:.2f}s")
    return text


if __name__ == "__main__":
    import httpx

    # Check server health (no /health endpoint; probe /v1/chat/completions)
    try:
        r = httpx.post(
            "http://127.0.0.1:8142/v1/chat/completions",
            json={"model": "", "messages": [], "max_tokens": 1},
            timeout=3,
        )
        print(f"Server status: {r.status_code} (400 = alive, no /health endpoint)")
    except Exception as e:
        print(f"Server not reachable: {e}")
        print("Start the server first with start.bat or manually.")
        exit(1)

    wav = record_audio()
    transcribe(wav)
