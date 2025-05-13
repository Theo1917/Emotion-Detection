from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2

model = load_model('modelsinuse/CNN_model.h5')

def preprocess_face(face):
    if len(face.shape) > 2 and face.shape[2] > 1:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    face_resized = cv2.resize(face, (48, 48))
    
    face_normalized = face_resized / 255.0
    
    face_cnn_input = face_normalized.reshape(1, 48, 48, 1)
    
    face_tensor = tf.convert_to_tensor(face_cnn_input, dtype=tf.float32)
    
    return face_tensor

def find_emotion(face_roi):
    processed_face = preprocess_face(face_roi)
            
            
    # Predict emotion with explicit batch dimension
    prediction = model(processed_face, training=False)
            
    # Get the class with highest probability
    emotion_label = np.argmax(prediction[0])
            
    return emotion_label