Speech Recognition Using MFCC Features
This project preprocesses audio data, extracting MFCC (Mel-frequency cepstral coefficients) features from WAV files for speech recognition tasks. It utilizes librosa for audio processing and TensorFlow for building machine learning models.

Table of Contents
1.Project Overview
2.Dependencies
3.Installation
4.Usage
5.Preprocessing
6.Training
7.Dataset

Project Overview
The goal of this project is to process audio files, extract MFCC features, and prepare the dataset for training machine learning models. The system works with WAV files and converts them into features that can be used for tasks like speech recognition.

Dependencies
Python 3.9
pip install tqdm==4.65.0
pip install librosa
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install keras==2.15.0
pip install tensorflow==2.15.0

Installation
git clone https://github.com/yoyoandop/Coin_sound_recognition.git
cd your-repo

Install the required packages:
conda create --name python3.9
conda create --name python3.9 python=3.9
conda activate python3.9
conda install pip

pip install tqdm==4.65.0
pip install librosa
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install keras==2.15.0
pip install tensorflow==2.15.0

pip list

python Recognition.py

Usage
Extracting MFCC Features
To extract MFCC features from WAV files:
from preprocess_v2 import wav2mfcc
mfcc = wav2mfcc("path_to_wav_file", max_pad_len=15)


Preprocessing and Saving Data
You can preprocess the audio data and save the MFCC feature arrays as .npy files:
from preprocess_v2 import save_data_to_array
save_data_to_array(path='path_to_audio_files', label=0, max_pad_len=11)

Training and Testing
Split your dataset into training and testing sets:
from preprocess_v2 import get_train_test
X_train, X_test, y_train, y_test = get_train_test(['label_class_0.npy', 'label_class_1.npy'], split_ratio=0.6)

Preprocessing
The main preprocessing script is located in preprocess_v2.py. This script handles:
Loading WAV files
Extracting MFCC features
Handling corrupted files
Saving processed data into .npy arrays

Dataset
in data flie have 1dollarmoney file 10dollarmoney file 50dollarmoney file 
