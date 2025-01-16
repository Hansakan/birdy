# data_preparation.py
import os
import numpy as np
import librosa

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def prepare_data():
    data_dir = 'bird_sounds'
    features = []
    labels = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_dir, file_name)
            feature = extract_features(file_path)
            label = file_name.split('_')[0]  # Assuming file names are like 'birdname_001.wav'
            features.append(feature)
            labels.append(label)
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    X, y = prepare_data()
    np.save('features.npy', X)
    np.save('labels.npy', y)
