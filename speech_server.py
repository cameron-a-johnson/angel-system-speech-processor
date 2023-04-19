from flask import Flask, render_template, request, redirect, jsonify
from flask_cors import CORS, cross_origin
import json
import librosa
import os
from requests import Response
import whisper

from speech_recognizer import SpeechRecognizer
from voice_activity_detector import VoiceActivityDetector

DEBUG_MODE = True

app = Flask(__name__)
cors = CORS(app)

sr = SpeechRecognizer()
vad = VoiceActivityDetector()

@app.route("/asr", methods=["GET", "POST"])
def apply_asr():

    files = request.files
    audio_file = files.get('audio_data')
    if not audio_file:
        msg = "ASR POST Request received, but missing 'audio_data' field."
        print(msg)
        return msg

    print("ASR POST Request received")        
    audio_file_path = 'test.wav'
    audio_file.save(audio_file_path)
    print(f"Saved 'audio_data' at {audio_file_path}")

    transcript = ""
    transcript = sr.convert_speech_to_text(audio_file_path)
    print(f"Transcript:\n\n{transcript}")
    resp = jsonify({"text": transcript})
    if not DEBUG_MODE:
        os.remove(audio_file_path)
    return resp

@app.route("/vad", methods=["GET", "POST"])
def apply_vad():
    files = request.files
    audio_file = files.get('audio_data')
    if not audio_file:
        msg = "VAD POST Request received, but missing 'audio_data' field."
        print(msg)
        return msg

    print("VAD POST Request received")        
    audio_file_path = 'test.wav'
    audio_file.save(audio_file_path)
    print(f"Saved 'audio_data' at {audio_file_path}")

    voice_segments = vad.get_segments(audio_file_path)
    for segment in voice_segments:
        print(f"Voice segment from {segment[0], segment[1]}")

    resp = json.dumps({"segments": voice_segments})
    if not DEBUG_MODE:
        os.remove(audio_file_path)
    return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=55667)
