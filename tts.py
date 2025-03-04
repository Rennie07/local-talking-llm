import nltk
import torch
import warnings
import numpy as np
from transformers import AutoProcessor, BarkModel
import time

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)

nltk.download("punkt", quiet=True)  # Ensure nltk sentence tokenizer is available


class TextToSpeechService:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initializes the TTS model using Suno's Bark."""
        print("[DEBUG] Initializing TextToSpeechService...")
        self.device = device
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        print("[DEBUG] Bark Processor loaded.")
        self.model = BarkModel.from_pretrained("suno/bark")
        print("[DEBUG] Bark Model loaded.")
        self.model.to(self.device)
        print(f"[DEBUG] Model moved to device: {self.device}")

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        """Converts text to speech using Bark."""
        print(f"[DEBUG] Synthesizing text: {text[:50]}...")  # Only print a part of the text for readability
        start_time = time.time()  # Start a timer to track how long synthesis takes

        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_array = self.model.generate(**inputs, pad_token_id=10000)

        end_time = time.time()  # End the timer
        print(f"[DEBUG] Synthesis completed in {end_time - start_time:.2f} seconds.")
        
        return self.model.generation_config.sample_rate, audio_array.cpu().numpy().squeeze()

    def long_form_synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        """Handles long-form text-to-speech by splitting sentences."""
        print(f"[DEBUG] Long-form synthesis starting for text of length {len(text)}...")
        start_time = time.time()  # Start a timer to track how long the entire synthesis process takes

        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.model.generation_config.sample_rate))

        print(f"[DEBUG] Number of sentences to process: {len(sentences)}")

        for i, sent in enumerate(sentences):
            print(f"[DEBUG] Processing sentence {i + 1}/{len(sentences)}: {sent[:50]}...")  # Show a snippet of each sentence
            sentence_start_time = time.time()  # Track time for each sentence

            sample_rate, audio_array = self.synthesize(sent, voice_preset)

            sentence_end_time = time.time()
            print(f"[DEBUG] Sentence {i + 1} synthesized in {sentence_end_time - sentence_start_time:.2f} seconds.")

            pieces += [audio_array, silence.copy()]

        end_time = time.time()
        print(f"[DEBUG] Long-form synthesis completed in {end_time - start_time:.2f} seconds.")

        return self.model.generation_config.sample_rate, np.concatenate(pieces)
