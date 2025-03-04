import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tts import TextToSpeechService

console = Console()
stt = whisper.load_model("small")  # Updated model for better accuracy
tts = TextToSpeechService()

template = """
You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less 
than 20 words.

The conversation transcript is as follows:
{history}

And here is the user's follow-up: {input}

Your response:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:", return_messages=True),
    llm=Ollama(),
)


def record_audio(stop_event, data_queue):
    """Captures audio from the microphone and adds it to a queue for processing."""
    def callback(indata, frames, time, status):
        if status:
            console.print(f"[red]Audio error: {status}")
        data_queue.put(bytes(indata))

    try:
        with sd.RawInputStream(
            samplerate=16000, dtype="int16", channels=1, callback=callback, blocksize=4096
        ):
            while not stop_event.is_set():
                time.sleep(0.1)
    except Exception as e:
        console.print(f"[red]Microphone error: {e}")


def transcribe(audio_np: np.ndarray) -> str:
    """Transcribes the audio data using Whisper."""
    result = stt.transcribe(audio_np, fp16=True)  # Use GPU acceleration if available
    return result["text"].strip()


def get_llm_response(text: str) -> str:
    """Generates a response using the latest Ollama model."""
    response = chain.predict(input=text)
    return response.replace("Assistant:", "").strip()


def play_audio(sample_rate, audio_array):
    """Plays the generated audio response."""
    sd.play(audio_array, sample_rate)
    sd.wait()


if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input("Press Enter to start recording, then press Enter again to stop.")

            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status("Generating response...", spinner="earth"):
                    response = get_llm_response(text)
                    sample_rate, audio_array = tts.long_form_synthesize(response)

                console.print(f"[cyan]Assistant: {response}")
                play_audio(sample_rate, audio_array)
            else:
                console.print("[red]No audio recorded. Please check your microphone settings.")

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")
    console.print("[blue]Session ended.")
