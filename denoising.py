import argparse
from enum import Enum
import librosa
import math
import numpy as np
from pysndfx import AudioEffectsChain
import python_speech_features
import scipy as sp
import soundfile
from tqdm import tqdm


class Reductions(Enum):
    mfcc_up = 0
    mfcc_down = 1
    mfcc_median = 2
    centroid_mb = 3
    centroid_s = 4
    power = 5

def read_file(file_name):
    y, sr = librosa.load(file_name)
    return y, sr

def reduce_noise_power(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    threshold_h = round(np.median(cent))*1.5
    threshold_l = round(np.median(cent))*0.1
    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.8).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)#.limiter(gain=6.0)
    y_clean = less_noise(y)
    return y_clean

def reduce_noise_centroid_s(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    threshold_h = np.max(cent)
    threshold_l = np.min(cent)
    less_noise = AudioEffectsChain().lowshelf(gain=-12.0, frequency=threshold_l, slope=0.5).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5).limiter(gain=6.0)
    y_cleaned = less_noise(y)
    return y_cleaned

def reduce_noise_centroid_mb(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    threshold_h = np.max(cent)
    threshold_l = np.min(cent)
    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.5).highshelf(gain=-30.0, frequency=threshold_h, slope=0.5).limiter(gain=10.0)
    # less_noise = AudioEffectsChain().lowpass(frequency=threshold_h).highpass(frequency=threshold_l)
    y_cleaned = less_noise(y)
    cent_cleaned = librosa.feature.spectral_centroid(y=y_cleaned, sr=sr)
    columns, rows = cent_cleaned.shape
    boost_h = math.floor(rows/3*2)
    boost_l = math.floor(rows/6)
    boost = math.floor(rows/3)
    # boost_bass = AudioEffectsChain().lowshelf(gain=20.0, frequency=boost, slope=0.8)
    boost_bass = AudioEffectsChain().lowshelf(gain=16.0, frequency=boost_h, slope=0.5)#.lowshelf(gain=-20.0, frequency=boost_l, slope=0.8)
    y_clean_boosted = boost_bass(y_cleaned)
    return y_clean_boosted

def reduce_noise_mfcc_down(y, sr):
    hop_length = 512
    ## librosa
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    # librosa.mel_to_hz(mfcc)
    ## mfcc
    mfcc = python_speech_features.base.mfcc(y)
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)
    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = AudioEffectsChain().highshelf(frequency=min_hz*(-1)*1.2, gain=-12.0, slope=0.6).limiter(gain=8.0)
    y_speach_boosted = speech_booster(y)

    return (y_speach_boosted)

def reduce_noise_mfcc_up(y, sr):
    hop_length = 512

    ## librosa
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    # librosa.mel_to_hz(mfcc)

    ## mfcc
    mfcc = python_speech_features.base.mfcc(y)
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = AudioEffectsChain().lowshelf(frequency=min_hz*(-1), gain=12.0, slope=0.5)#.highshelf(frequency=min_hz*(-1)*1.2, gain=-12.0, slope=0.5)#.limiter(gain=8.0)
    y_speach_boosted = speech_booster(y)

    return (y_speach_boosted)

def reduce_noise_median(y, sr):
    y = sp.signal.medfilt(y,3)
    return (y)


def trim_silence(y):
    y_trimmed, index = librosa.effects.trim(y, top_db=20, frame_length=2, hop_length=500)
    trimmed_length = librosa.get_duration(y) - librosa.get_duration(y_trimmed)
    return y_trimmed, trimmed_length


def enhance(y):
    apply_audio_effects = AudioEffectsChain().lowshelf(gain=10.0, frequency=260, slope=0.1).reverb(reverberance=25, hf_damping=5, room_scale=5, stereo_depth=50, pre_delay=20, wet_gain=0, wet_only=False)#.normalize()
    y_enhanced = apply_audio_effects(y)
    return y_enhanced

# Original script artifact:
# def output_file(destination, filename, y, sr, ext=""):
#     destination = destination + filename.split("/")[-1][:-4] + ext + '.wav'
#     # librosa.output.write_wav(destination, y, sr)
#     soundfile.write(destination, y, sr, subtype='PCM_16')

def output_file(file_name, y, sr):
    soundfile.write(file_name, y, sr, subtype='PCM_16')

def reduce_noise_power_file(in_file, out_file, is_trim_silence = False):
    y, sr = read_file(in_file)
    denoised_y = reduce_noise_power(y, sr)
    if is_trim_silence:
        denoised_y, time_trimmed = trim_silence(denoised_y)
        print(f"Trimmed audio by {time_trimmed} seconds")
    output_file(out_file, denoised_y, sr)

def reduce_noise_centroid_s_file(in_file, out_file, is_trim_silence = False):
    y, sr = read_file(in_file)
    denoised_y = reduce_noise_centroid_s(y, sr)
    if is_trim_silence:
        denoised_y, time_trimmed = trim_silence(denoised_y)
        print(f"Trimmed audio by {time_trimmed} seconds")
    output_file(out_file, denoised_y, sr)

def reduce_noise_centroid_mb_file(in_file, out_file, is_trim_silence = False):
    y, sr = read_file(in_file)
    denoised_y = reduce_noise_centroid_mb(y, sr)
    if is_trim_silence:
        denoised_y, time_trimmed = trim_silence(denoised_y)
        print(f"Trimmed audio by {time_trimmed} seconds")
    output_file(out_file, denoised_y, sr)

def reduce_noise_mfcc_up_file(in_file, out_file, is_trim_silence = False):
    y, sr = read_file(in_file)
    denoised_y = reduce_noise_mfcc_up(y, sr)
    if is_trim_silence:
        denoised_y, time_trimmed = trim_silence(denoised_y)
        print(f"Trimmed audio by {time_trimmed} seconds")
    output_file(out_file, denoised_y, sr)

def reduce_noise_mfcc_down_file(in_file, out_file, is_trim_silence = False):
    y, sr = read_file(in_file)
    denoised_y = reduce_noise_mfcc_down(y, sr)
    if is_trim_silence:
        denoised_y, time_trimmed = trim_silence(denoised_y)
        print(f"Trimmed audio by {time_trimmed} seconds")
    output_file(out_file, denoised_y, sr)

def reduce_noise_mfcc_median_file(in_file, out_file, is_trim_silence = False):
    y, sr = read_file(in_file)
    denoised_y = reduce_noise_median(y, sr)
    if is_trim_silence:
        denoised_y, time_trimmed = trim_silence(denoised_y)
        print(f"Trimmed audio by {time_trimmed} seconds")
    output_file(out_file, denoised_y, sr)
