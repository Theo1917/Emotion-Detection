import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Change this line to load the .joblib file
model = load_model("modelsinuse/dense_nn.h5")  # Update the file extension

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        combined_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
        
        features = np.mean(combined_features, axis=1)
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is not None:
        features = features.reshape(1, -1)
        predictions = model.predict(features)
        predicted_label = np.argmax(predictions[0]) 
        return predicted_label
    return 2