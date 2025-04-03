# server.py
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from chatbot import ChatBot
import os
from kokoro import KPipeline
import soundfile as sf
import io
import tempfile
import numpy as np
from voice_recognition import VoiceRecognizer

app = Flask(__name__, static_folder='.')
CORS(app)

# Initialize chatbot and TTS pipeline
chatbot = ChatBot()
tts_pipeline = KPipeline(lang_code='a')  # American English

# Initialize voice recognizer
voice_recognizer = VoiceRecognizer()

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/api/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Empty query received'}), 400
            
        response = chatbot.get_response(query)
        
        # Generate speech from response
        audio_chunks = []
        # Split response into sentences or smaller chunks if needed
        text_chunks = [s.strip() for s in response.split('.') if s.strip()]
        
        for chunk in text_chunks:
            for _, _, audio in tts_pipeline(chunk + '.', voice='af_bella', speed=1):
                if (audio is not None):
                    audio_chunks.append(audio)
        
        # Concatenate all audio chunks
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            
            # Save complete audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, full_audio, 24000)
                temp_path = temp_file.name
                
            return jsonify({
                'answer': response,
                'audio_url': f'/api/audio/{os.path.basename(temp_path)}'
            })
        else:
            raise Exception("Failed to generate audio")
            
    except Exception as e:
        app.logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': 'Failed to process request'}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            app.logger.error("No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['audio']
        if not audio_file.filename:
            app.logger.error("Empty audio filename")
            return jsonify({'error': 'Invalid audio file'}), 400
            
        audio_data = audio_file.read()
        if not audio_data:
            app.logger.error("Empty audio data")
            return jsonify({'error': 'Empty audio file'}), 400
            
        app.logger.info(f"Processing audio file: {audio_file.filename} ({len(audio_data)} bytes)")
        
        # Get transcription from Hugging Face
        result = voice_recognizer.transcribe_audio(audio_data)
        app.logger.info(f"Transcription result: {result}")
        
        if 'error' in result:
            app.logger.error(f"Transcription error: {result['error']}")
            return jsonify({'error': result['error']}), 500
            
        if 'text' not in result or not result['text']:
            app.logger.error("No text in transcription result")
            return jsonify({'error': 'No text transcribed'}), 500
            
        return jsonify({'text': result['text']})
        
    except Exception as e:
        app.logger.error(f"Error transcribing audio: {str(e)}")
        return jsonify({'error': 'Failed to process audio'}), 500

@app.route('/api/audio/<filename>')
def get_audio(filename):
    try:
        return send_file(
            os.path.join(tempfile.gettempdir(), filename),
            mimetype='audio/wav'
        )
    except Exception as e:
        return jsonify({'error': 'Audio file not found'}), 404

@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)