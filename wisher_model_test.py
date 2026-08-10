#### pip install faster-whisper sounddevice scipy ollama

import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from ollama import chat

# -----------------------------
# 1. Load Whisper Tiny
# -----------------------------
stt_model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

# -----------------------------
# 2. Record microphone
# -----------------------------
SAMPLE_RATE = 16000
RECORD_SECONDS = 5

print("🎤 Speak now...")

audio = sd.rec(
    int(RECORD_SECONDS * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)

sd.wait()

write("input.wav", SAMPLE_RATE, audio)

print("✅ Recording finished")


# -----------------------------
# 3. Speech → Text
# -----------------------------
segments, info = stt_model.transcribe(
    "input.wav",
    language="en"
)

text = " ".join(segment.text for segment in segments)

print("\nYou:", text)


# -----------------------------
# 4. Send text to Ollama
# -----------------------------
response = chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful desktop voice assistant. Answer concisely."
        },
        {
            "role": "user",
            "content": text
        }
    ]
)

answer = response.message.content

print("\n🤖 Assistant:", answer)
