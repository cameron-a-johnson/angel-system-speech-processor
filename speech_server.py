from flask import Flask, render_template, request, redirect, jsonify
from flask_cors import CORS, cross_origin
import json
import librosa
import os
from requests import Response
from typing import List
import whisper

import denoising
from speech_recognizer import SpeechRecognizer
from voice_activity_detector import VoiceActivityDetector

DEBUG_MODE = True

app = Flask(__name__)
cors = CORS(app)

sr = SpeechRecognizer()
vad = VoiceActivityDetector()

def apply_denoising(filename: str, transformation: str):
    if transformation ==  denoising.Reductions.mfcc_up.name:
        denoising.reduce_noise_mfcc_up_file(filename, filename)
    elif transformation == denoising.Reductions.mfcc_down.name:
        denoising.reduce_noise_mfcc_down_file(filename, filename)
    elif transformation == denoising.Reductions.mfcc_median.name:
        denoising.reduce_noise_mfcc_median_file(filename, filename)
    elif transformation == denoising.Reductions.centroid_mb.name:
        denoising.reduce_noise_centroid_mb_file(filename, filename)
    elif transformation == denoising.Reductions.centroid_s.name:
        denoising.reduce_noise_centroid_s_file(filename, filename)
    elif transformation == denoising.Reductions.power.name:
        denoising.reduce_noise_power_file(filename, filename)
    else:
        print(f"Unsupported preprocessing provided {transformation}.")

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

    if 'preprocessing' in request.args.keys():
        print("Applying denoising...")
        apply_denoising(audio_file_path, request.args['preprocessing'])
        print(f"Applied {request.args['preprocessing']} to {audio_file_path}")

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

    if 'preprocessing' in request.args.keys():
        print("Applying denoising...")
        apply_denoising(audio_file_path, request.args['preprocessing'])
        print(f"Applied {request.args['preprocessing']} to {audio_file_path}")

    voice_segments = vad.get_segments(audio_file_path)
    for segment in voice_segments:
        print(f"Voice segment from {segment[0], segment[1]}")

    resp = json.dumps({"segments": voice_segments})
    if not DEBUG_MODE:
        os.remove(audio_file_path)
    return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=55667)
