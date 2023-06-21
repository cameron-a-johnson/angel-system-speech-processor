import subprocess
import sys
import time
import whisper
from termcolor import colored

class SpeechRecognizer:

    def __init__(self):
        self.model = whisper.load_model("medium")

    def convert_speech_to_text(self, audio_file_path, debug_mode=False):

        start_time = time.time()
        audio_input = whisper.load_audio(audio_file_path)
        if debug_mode:
            print(f"Processing for audio file {audio_file_path}")

        audio_input = whisper.pad_or_trim(audio_input)
        mel = whisper.log_mel_spectrogram(audio_input).to(self.model.device)

        options = whisper.DecodingOptions(language="en")
        result = whisper.decode(self.model, mel, options)
        colored_text = colored(result.text, "red")
        if debug_mode:
            print(f"Transcript: {colored_text}")

        print("Decoding took --- %s seconds ---" % (time.time() - start_time))
        return result.text
