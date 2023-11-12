# Check out https://arxiv.org/abs/2104.04045
# Also check out https://huggingface.co/philschmid/pyannote-segmentation
# 1. visit hf.co/pyannote/segmentation and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained voice activity detection pipeline

import os
import sys
import time

from pyannote.audio import Pipeline

HUGGING_FACE_TOKEN =  os.getenv("HUGGING_FACE_ACCESS_TOKEN")

class VoiceActivityDetector:

    def __init__(self, auth_token=HUGGING_FACE_TOKEN):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=auth_token)

    def get_segments(self, audio_file_path, debug_mode=False): 
        '''
        Returns a list of voice segments (by start/end-times) in an audio file.
        '''
        
        start_time = time.time()
        output = self.pipeline(audio_file_path)
        if debug_mode:
            print("Processing for audio file", audio_file_path)
        
        print("--- %s seconds taken ---" % (time.time() - start_time))
        return list(map(lambda speech: [speech.start, speech.end],
            output.get_timeline().support()))
