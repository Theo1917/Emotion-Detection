from flask import Flask,render_template,Response,jsonify
import cv2
import numpy as np
import threading
import pyaudio
import wave
import os
import audio_emotion_finder
import video_emotion_finder
import time

app = Flask(__name__)

audio_emotion_Latest = 'Neutral'
video_emotion_Latest = 'Neutral'
emotions = ["Surprised", "Happy", "Neutral", "Sad", "Angry"]

def record_audio(duration=5, sample_rate=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
    frames = []

    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    filename = f"audio_recording.wav"
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    print(f"Audio saved to {filename}")

    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))
    
    return audio_data, sample_rate, filename

def detect_audio(sample_rate=44100,duration=5):
    global audio_emotion_Latest
    while True:
        print("Recording...")
        audio_data, sample_rate, filename  = record_audio()
        emotion_label = audio_emotion_finder.predict_emotion(filename)
        audio_emotion_Latest = emotions[emotion_label]
        print(f'Predicted emotion: {emotions[emotion_label]}')
        if os.path.exists(filename):
                os.remove(filename)
                print(f"Deleted file: {filename}")

def video_frames():
    global video_emotion_Latest
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        return "Error: Could not open video capture device."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        current_emotion = 'Neutral'
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            
            if face_roi.size > 0:
                try:
                    emotion_label = video_emotion_finder.find_emotion(face_roi)
                    
                    video_emotion = emotions[emotion_label]
                    current_emotion = video_emotion
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, video_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                except Exception as e:
                    print(f"Error in emotion detection: {e}")

        video_emotion_Latest = current_emotion
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Video')
def Video():
    return render_template('Video.html')

@app.route('/Audio')
def Audio():
    return render_template('Audio.html')

@app.route('/VideoAndAudio')
def VideoAndAudio():
    return render_template('VideoAndAudio.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/audio_emotion')
def audio_emotion():
    return jsonify({'audio_emotion':audio_emotion_Latest})

@app.route('/merged_emotion')
def merged_emotion():
    audio_idx = emotions.index(audio_emotion_Latest) if audio_emotion_Latest in emotions else 2
    video_idx = emotions.index(video_emotion_Latest) if video_emotion_Latest in emotions else 2
    
    audio_weight = 0.6
    video_weight = 0.4
    
    avg_idx = int(round(audio_weight * audio_idx + video_weight * video_idx))
    
    avg_idx = max(0, min(avg_idx, len(emotions)-1))
    
    merged_emotion = emotions[avg_idx]
    
    return jsonify({
        'audio_emotion': audio_emotion_Latest,
        'video_emotion': video_emotion_Latest,
        'merged_emotion': merged_emotion,
    })

if __name__ == '__main__':
    audio_thread = threading.Thread(target=detect_audio)
    audio_thread.start()
    app.run(debug=True)
