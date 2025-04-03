from voice_processor import VoiceProcessor
import os
from dotenv import load_dotenv

load_dotenv()

class VoiceRecognizer:
    def __init__(self):
        self.processor = VoiceProcessor()

    def transcribe_audio(self, audio_data):
        """Transcribe audio data using the voice processor."""
        result = self.processor.transcribe_audio(audio_data)
        
        if 'error' in result:
            return {"error": result['error']}
            
        # Extract text from Whisper response
        text = result.get('text', '')
        return {"text": text}
