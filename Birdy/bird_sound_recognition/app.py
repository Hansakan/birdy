from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import librosa
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('bird_sound_model.h5')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder.npy', allow_pickle=True)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    try:
        features = extract_features(file_path)
        features = features[np.newaxis, ..., np.newaxis]
        predictions = model.predict(features)
        predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])[0]
        os.remove(file_path)
        return jsonify({'bird': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
