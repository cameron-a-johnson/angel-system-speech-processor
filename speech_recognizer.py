import subprocess
import sys
import time
import whisper

#PATH="/home/smg2280/Sayali-Workspace/espnet/egs2/aesrc2020/asr1/"
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
        text = result.text
        if debug_mode:
            print(f"Transcript: {text}")

        # #audio_input = str(sys.argv[1])
        # audio_input = audio_file_path
        
        # process = subprocess.Popen(['bash',PATH+'run_single_audio.sh', audio_input])
        # process.wait()
        # with open(PATH+"exp/espnet/brianyan918_aesrc2020_asr_conformer/decode_asr_asr_model_valid.acc.ave/convert_me/text","r") as f:
        #     text = " ".join(f.read().split(" ")[1:])
        #     print("Transcript:", text)

        print("Decoding took --- %s seconds ---" % (time.time() - start_time))
        return text
