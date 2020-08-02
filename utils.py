import os
import librosa
import numpy as np

train_dir = os.path.join(os.curdir, "train")
data_dir = os.path.join(os.curdir, "data")

def signal2data(signal):
    data = librosa.stft(signal)
    # data = librosa.feature.melspectrogram(signal)
    # data = librosa.feature.melspectrogram(signal, sr=44100, n_fft=2048, hop_length=512, power=2.0)
    # data = librosa.feature.mfcc(signal)
    return data[:800]

def fit_size(signal):
    signal = signal[:54200]
    if len(signal) > 44200:
        signal = signal[len(signal) - 44100:]
    else:
        signal = np.concatenate((signal, np.zeros(44100 - len(signal))))
    return signal
