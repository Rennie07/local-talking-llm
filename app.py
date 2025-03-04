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
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from tts import TextToSpeechService

# Initialize models
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
    global vector_store
    print("[DEBUG] PDF Uploaded. Processing...")
    docs = PyPDFLoader(file_path=pdf_file_path).load()
    print("[DEBUG] Loading PDF...")
    print(f"[DEBUG] PDF contains {len(docs)} pages.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100).split_documents(docs)
    print(f"[DEBUG] PDF split into {len(chunks)} chunks.")
    chunks = filter_complex_metadata(chunks)
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=FastEmbedEmbeddings(),
        persist_directory="chroma_db"
    )
    print("[DEBUG] PDF successfully processed into vector store.")

def ask_pdf(query: str) -> str:
    global vector_store
    if not vector_store:
        return "Please upload a PDF document first."

    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 10, "score_threshold": 0.0})
    
    print("[DEBUG] Retrieving relevant documents...")
    context = retriever.get_relevant_documents(query)
    print(f"[DEBUG] Retrieved {len(context)} relevant document chunks.")
    context_text = "\n".join([doc.page_content for doc in context])

    formatted_prompt = PROMPT.format(context=context_text, input=query)

    print("[DEBUG] Sending prompt to LLM...")
    response = llm(formatted_prompt)
    print(f"[DEBUG] LLM Response: {response}")

    return response

def play_audio(sample_rate, audio_array):
    if audio_array is None or len(audio_array) == 0:
        print("[ERROR] Generated audio is empty. Check TTS output.")
        return

    try:
        print(f"[DEBUG] Playing audio... Sample Rate: {sample_rate}, Audio Length: {len(audio_array)} samples")
        sd.play(audio_array, sample_rate)
        sd.wait()
        print("[DEBUG] Finished playing audio.")
    except Exception as e:
        print(f"[ERROR] Error playing audio: {e}")

def main():
    global vector_store

    st.title("PDF Voice Assistant")

    if "audio_error" not in st.session_state:
        st.session_state["audio_error"] = None

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(uploaded_file.getbuffer())
            file_path = tf.name
        ingest_pdf(file_path)
        os.remove(file_path)
        st.success("PDF uploaded and processed!")

    if st.session_state["audio_error"]:
        st.warning(st.session_state["audio_error"])

    user_input_type = st.radio("Choose input method:", ("Voice", "Text"))

    if user_input_type == "Text":
        user_query = st.text_input("Ask a question:")
        if user_query:
            print(f"[DEBUG] User input (text): {user_query}")
            st.write(f"**You:** {user_query}")
            st.write("Generating response...")
            response = ask_pdf(user_query)
            st.write(f"**Assistant:** {response}")

            print("[DEBUG] Sending response to TTS for synthesis...")
            sample_rate, audio_array = tts.long_form_synthesize(response)
            print(f"[DEBUG] TTS synthesis completed. Sample Rate: {sample_rate}, Audio Length: {len(audio_array)}")

            print("[DEBUG] Calling play_audio function...")
            play_audio(sample_rate, audio_array)
            print("[DEBUG] Finished processing audio.")

if __name__ == "__main__":
    main()
