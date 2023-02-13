from flask import Flask, render_template, request, redirect, jsonify
#from test_single_audio import SpeechRecognizer
from speech_recognizer import SpeechRecognizer

from flask_cors import CORS, cross_origin
import librosa
import whisper
from requests import Response

app = Flask(__name__)
cors = CORS(app)

sr = SpeechRecognizer()
#model = whisper.load_model("medium")

@app.route("/home", methods=["GET", "POST"])
def home():
    print("Request received")
   
    if request.method == "POST":
        print("Here")
        files = request.files
        print(files)
        audio_file = files.get('audio_data')
        print(audio_file)
        audio_file.save(str(audio_file))
        # with open(str(audio_file), 'wb') as f:
            # audio_file.save(str(audio_file))
            # f.write(audio_file)
        print(str(audio_file))
        # file = files.get('audio_data')
        # file.save('./convert_me.wav')

        print("FORM DATA RECEIVED")

        transcript = ""

        # filename = 'convert_me.wav'
        # file.save(filename)

        # file.seek(0)

        # if "audio_file" not in request.files:
        #     return redirect(request.url)

        #= request.files["audio_file"]

        # blob = request.data
        # if blob:
        #     print("Blob data received!")

        # with open('./convert_me.wav', 'wb') as f:
        #     f.write(file.read())

        # audio_file = "./convert_me.wav"
        # print('Audio data saved!')
        # if file.filename == "":
        #     return redirect(request.url)

        if audio_file:
            transcript = sr.convert_speech_to_text(str(audio_file))
        return jsonify({"text":transcript})
        # return transcript #{"text": transcript}
    # else:
        # print("Not a post")
        # return "OK"

if __name__ == "__main__":
    print("Derek: This is a test")
    app.run(host='0.0.0.0',port=8058)
