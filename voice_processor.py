import pyaudio
import wave
from io import BytesIO
import numpy as np
import requests
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceProcessor:
    def __init__(self):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.SILENCE_THRESHOLD = 500
        self.SILENCE_DURATION = 1.0  # seconds
        self.PRE_SPEECH_BUFFER_DURATION = 0.5  # seconds
        self.audio = pyaudio.PyAudio()
        
        # Update API endpoint to use Whisper model directly
        self.API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
            "Content-Type": "audio/webm"  # Update content type to match frontend
        }

    def is_silence(self, data):
        """Detect if the provided audio data is silence."""
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data**2))
        return rms < self.SILENCE_THRESHOLD

    def record_audio(self):
        """Record audio with silence detection."""
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("Recording...")
        frames = []
        silent_chunks = 0
        
        while True:
            data = stream.read(self.CHUNK)
            frames.append(data)
            
            if self.is_silence(data):
                silent_chunks += 1
            else:
                silent_chunks = 0
                
            if silent_chunks > int(self.RATE / self.CHUNK * self.SILENCE_DURATION):
                break
        
        print("Finished recording")
        stream.stop_stream()
        stream.close()
        
        # Convert frames to BytesIO object
        audio_bytes = BytesIO()
        with wave.open(audio_bytes, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
        
        audio_bytes.seek(0)
        return audio_bytes

    def transcribe_audio(self, audio_data):
        """Transcribe audio using Hugging Face Whisper API."""
        try:
            # Handle both bytes and BytesIO objects
            if isinstance(audio_data, bytes):
                data = audio_data
            elif isinstance(audio_data, BytesIO):
                data = audio_data.read()
            else:
                raise ValueError("Invalid audio data format")

            logger.info("Sending audio data to Hugging Face API")
            response = requests.post(
                self.API_URL,
                headers=self.headers,
                data=data
            )
            
            if not response.ok:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return {"error": f"API Error: {response.status_code}"}

            try:
                result = response.json()
                logger.info(f"Transcription result: {result}")
                
                if isinstance(result, dict):
                    if 'error' in result:
                        return {"error": str(result['error'])}
                    if 'text' in result:
                        return {"text": result['text']}
                return {"text": str(result)}
                
            except requests.exceptions.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)} - Response: {response.text}")
                return {"error": "Invalid response from API"}
            
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            return {"error": str(e)}

    def __del__(self):
        """Cleanup PyAudio."""
        self.audio.terminate()
