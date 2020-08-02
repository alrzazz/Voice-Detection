import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.models import Sequential
from utils import train_dir, data_dir, signal2data, fit_size
import warnings
import sys
from keras.layers.cudnn_recurrent import CuDNNLSTM

epochs=100
batch_size=128

if __name__ == "__main__":

    sound_files = os.listdir(train_dir)
    input_data = []
    output_data = [] 

    for sound_name in sound_files:
        path = os.path.join(train_dir, sound_name)
        try:
            signal, sample_rate = librosa.load(path, sr=44100)
            signal = fit_size(signal)
            data = signal2data(signal)
            input_data.append(data)
            if 'yes' in sound_name:
                output_data.append([1])
            elif 'no' in sound_name:
                output_data.append([0])
        except :
            pass

    input_data = np.array(input_data)
    output_data = np.array(output_data)

    indexes = np.array(range(input_data.shape[0]))
    np.random.shuffle(indexes)

    X_train = np.array(input_data[indexes])
    Y_train = np.array(output_data[indexes])

    np.random.shuffle(indexes)

    X_test = np.array(input_data[indexes])
    Y_test = np.array(output_data[indexes])

    model = Sequential()
    model.add(Conv1D(128, 3, input_shape=(800, 87), activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.01))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.01))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_test, Y_test))

    model.save(os.path.join(data_dir, "model.h5"))

    loss = history.history["val_loss"]
    accuracy = history.history["val_accuracy"]
    plt.figure(figsize=(35, 10), dpi=300)

    plt.subplot(121)
    plt.title("LOSS", fontsize=40)
    plt.plot(loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.subplot(122)
    plt.title("ACCURACY", fontsize=40)
    plt.plot(accuracy)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(data_dir, "model.png"), format="png", dpi=300)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    