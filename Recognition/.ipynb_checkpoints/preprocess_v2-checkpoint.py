import librosa
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
def get_labels(path=None, label=0):
    labels = os.listdir(path)
    label_indices = np.full_like(np.arange(0, len(labels)), label)
    return labels, label_indices, to_categorical(label_indices)

def wav2mfcc(file_path, max_pad_len=15):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::1] 
    i=0
    wav_length=5000
    if len(wave) > wav_length:

        i=np.argmax(wave)
        if i > (wav_length):
            wave = wave[i-int(wav_length/15):i+int((wav_length/15)*14)]
        else:
            wave = wave[0:i]


    mfcc = librosa.feature.mfcc(y=wave, sr=40000 )
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width < 0:
        pad_width = 0
        mfcc = mfcc[:,:15]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc


def save_data_to_array(path=None, label=0, max_pad_len=11):
    labels, _, _ = get_labels(path, label=label)
    mfcc_vectors = [] # Init mfcc vectors
    label_class = path.split('/')[-1] 
    for label in labels:
        wavfile = path + '/' + label
        try:
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        except:
            print(f'ERROR, Corrupt file found: {wavfile}')
    
    np.save(label_class + '.npy', mfcc_vectors)

def get_train_test(npy_filepath=[], split_ratio=0.6, random_state=42):
    # Load npy files and merge them into a single array
    X = np.load(npy_filepath[0])
    y = np.zeros(X.shape[0])
    for i, npy_file in enumerate(npy_filepath[1:], 1):
        x = np.load(npy_file)
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)

def prepare_dataset(path, label=0):
    labels, _, _ = get_labels(path=None, label=label)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path  + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            # Downsampling
            wave = wave[::1]
            mfcc = librosa.feature.mfcc(wave, sr=40000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data

def load_dataset(path):
    data = prepare_dataset(path=None)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]
