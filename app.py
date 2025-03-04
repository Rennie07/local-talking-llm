import os
import time
import threading
import tempfile
import numpy as np
import whisper
import sounddevice as sd
import streamlit as st
from queue import Queue
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import FastEmbedEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from tts import TextToSpeechService

# Initialize models
console = st.empty()
stt = whisper.load_model("small")
tts = TextToSpeechService()

# LangChain prompt
template = """
You are a helpful and friendly AI assistant. Answer the question using the provided PDF context.

PDF Context:
{context}

User Question:
{input}

Your Response:
"""
PROMPT = PromptTemplate(input_variables=["context", "input"], template=template)

# Conversation memory
memory = ConversationBufferMemory(ai_prefix="Assistant:", return_messages=True)

# LLM
llm = Ollama()

# PDF Processing
vector_store = None

def ingest_pdf(pdf_file_path: str):
    """Loads a PDF, splits text into chunks, and stores embeddings in a vector database."""
    global vector_store
    docs = PyPDFLoader(file_path=pdf_file_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100).split_documents(docs)
    chunks = filter_complex_metadata(chunks)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=FastEmbedEmbeddings(),
        persist_directory="chroma_db"
    )

def ask_pdf(query: str) -> str:
    """Retrieves relevant context from the vector store and generates an answer."""
    global vector_store
    if not vector_store:
        return "Please upload a PDF document first."

    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 10, "score_threshold": 0.0})

    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)

def record_audio(stop_event, data_queue):
    """Captures audio from the microphone and adds it to a queue for processing."""
    def callback(indata, frames, time, status):
        if status:
            console.warning(f"Audio error: {status}")
        data_queue.put(bytes(indata))

    try:
        with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback, blocksize=4096):
            while not stop_event.is_set():
                time.sleep(0.1)
    except Exception as e:
        console.warning(f"Microphone error: {e}")

def transcribe(audio_np: np.ndarray) -> str:
    """Transcribes the audio data using Whisper."""
    result = stt.transcribe(audio_np, fp16=True)
    return result["text"].strip()

def play_audio(sample_rate, audio_array):
    """Plays the generated audio response."""
    sd.play(audio_array, sample_rate)
    sd.wait()

def main():
    global vector_store

    st.title("PDF Voice Assistant")

    # PDF Upload
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(uploaded_file.getbuffer())
            file_path = tf.name
        ingest_pdf(file_path)
        os.remove(file_path)
        st.success("PDF uploaded and processed!")

    # Voice Input
    if st.button("Start Recording"):
        data_queue = Queue()
        stop_event = threading.Event()
        recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
        recording_thread.start()

        st.write("Recording... Press 'Stop Recording' to finish.")
        if st.button("Stop Recording"):
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if audio_np.size > 0:
                st.write("Transcribing...")
                text = transcribe(audio_np)
                st.write(f"**You:** {text}")

                st.write("Generating response...")
                response = ask_pdf(text)

                st.write(f"**Assistant:** {response}")
                sample_rate, audio_array = tts.long_form_synthesize(response)
                play_audio(sample_rate, audio_array)
            else:
                st.warning("No audio recorded. Please check your microphone settings.")

if __name__ == "__main__":
    main()
