import numpy as np
import librosa
from tensorflow.keras.models import load_model

model = load_model("modelsinuse/cnn_model_final.h5") 

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        features = np.mean(mfccs, axis=1)
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std

def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is not None:
        features = standardize(features.reshape(1, -1))
        predictions = model.predict(features)
        pred_class = np.argmax(predictions, axis=1)[0]
        predicted_label = np.argmax(predictions[0]) 
        confidence = predictions[0][pred_class]

        return predicted_label
    return 2