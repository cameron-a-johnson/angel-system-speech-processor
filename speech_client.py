import argparse
import json
import numpy as np
import pyaudio
import requests
import struct
import time
from typing import *
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 4

ASR_URL = 'http://communication.cs.columbia.edu:8058/asr'
VAD_URL = 'http://communication.cs.columbia.edu:8058/vad'
WAVE_OUTPUT_FILENAME = "output.wav"

def read_audio_stream():
    p = pyaudio.PyAudio()

    # _ = input("Press enter to continue.")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("* recording")

    frames = []
    intframes = []
    continue_listening = True
    found_speech = False
    time_of_last_speech = time.time()
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        # intdata = np.array(struct.unpack(str(CHUNK) + 'h', data))
        # intdata = np.abs(intdata / 32767)
        # intdata = np.array([(intdata / np.max(np.abs(intdata))) * 32767], np.int16)
        # is_speech = np.any(intdata > 0.1)
        # intframes.extend(intdata)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    # print(intframes)
    # print(max(intframes))
    # print(min(intframes))
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return WAVE_OUTPUT_FILENAME

def apply_asr(file: str):
    with open(file, 'rb') as f:
        print(f"Querying {ASR_URL}")
        x = requests.post(ASR_URL, files={'audio_data': f})
    return x.text

def apply_vad(file: str):
    with open(file, 'rb') as f:
        print(f"Querying {VAD_URL}")
        x = requests.post(VAD_URL, files={'audio_data': f})
    return json.loads(x.content)['segments']

def _print_voice_segments(segments: List[Tuple[float]]):
    if not segments:
        print("No voice segments found.")
    for segment in segments:
        print(f"Voice segment from {segment[0]} to {segment[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file",
        help="Process the provided wav file")
    parser.add_argument('-a', '--asr',
                        action='store_true')
    parser.add_argument('--vd',
                        action='store_true')
    args = parser.parse_args()

    if args.asr:
        if args.file:
            print(apply_asr(args.file))
        else:
            filename = read_audio_stream()
            print(apply_asr(filename))

    voice_segments = None
    if args.vd:
        if args.file:
            voice_segments = apply_vad(args.file)
        else:
            filename = read_audio_stream()
            voice_segments = apply_vad(filename)
        _print_voice_segments(voice_segments)

