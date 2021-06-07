import os
import sys
import numpy as np
import sounddevice as sd
import librosa
from keras import models

def fit_size(signal):
    signal = signal[:54200]
    if len(signal) > 44200:
        signal = signal[len(signal) - 44200:]
    else:
        signal = np.concatenate((signal, np.zeros(44200 - len(signal))))
    return signal

sample_rate = 44100
duration = 0.0625
chunk = int(duration * sample_rate)


if __name__ == "__main__":
    model = models.load_model("model.h5")

    if len(sys.argv) == 2:
        print("start offline")
        path = sys.argv[1]
        signal, _ = librosa.load(path, sr=sample_rate)
        signal = fit_size(signal)
        data = librosa.stft(signal)
        result = model.predict(np.array([data]))

        if result[0][0] > 0.99:
            print("YES")
        elif result[0][0] < 0.01:
            print("NO")
        else:
            print("MUTE")

    else:
        print("start online")
        signal = np.zeros((chunk * 16,), dtype=np.float32)
        stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.float32)
        stream.start()

        while True:
            Y = 0
            N = 0
            for _ in range(16):
                mic = stream.read(chunk)[0].reshape(chunk)
                sd.wait()
                signal = np.concatenate((mic, signal[chunk:]))
                data = librosa.stft(signal)
                data = np.array([data])
                data = data[..., None]
                result = model.predict(data)
                if(result[0][0] >= 0.9):
                    Y = Y + 1
                if(result[0][1] >= 0.9):
                    N = N + 1
            if Y >= 4 and N >= 4:
                if Y > N:
                    print("Yes")
                else:
                    print("No")
            elif Y >= 4:
                print("Yes")
            elif N >= 4:
                print("No")
            else:
                print("Mute")
