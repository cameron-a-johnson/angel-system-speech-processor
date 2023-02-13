import pyaudio
import wave
import struct
import numpy as np
import time

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 4
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

    import requests

    url = 'http://communication.cs.columbia.edu:8058/home'
    # print(frames[1])
    with open(WAVE_OUTPUT_FILENAME, 'rb') as f:
        x = requests.post(url, files={'audio_data': f})
    # packet = {'audio_data' : frames}
    # x = requests.post(url, json=packet)
    # print(x.text)
    return x.text

if __name__ == "__main__":
    print(read_audio_stream())

