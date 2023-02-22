import argparse
import numpy as np
import pyaudio
import requests
import struct
import time
import wave

FORMAT = pyaudio.paInt16
CHUNK = 1024
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 4
URL = 'http://communication.cs.columbia.edu:8058/home'
WAV_OUTPUT_FILENAME = "output.wav"


def read_audio_stream(wav_output_filename: str = WAV_OUTPUT_FILENAME):
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
    wf = wave.open(wav_output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # print(frames[1])
    with open(wav_output_filename, 'rb') as f:
        x = requests.post(URL, files={'audio_data': f})
    # packet = {'audio_data' : frames}
    # x = requests.post(url, json=packet)
    # print(x.text)
    return x.text

def send_audio_file(wav_file: str):
    with open(wav_file, 'rb') as f:
        resp = requests.post(URL, files={'audio_data': f})
    return resp.text

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process input arguments for ASR server requesting.')
    parser.add_argument('-f', '--wav_file', type=str,
                        help='Send a specified .wav file to the ASR server')
    args = parser.parse_args()

    if args.wav_file:
        print(print(send_audio_file(args.wav_file)))
    else:
        print(read_audio_stream())

