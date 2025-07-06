import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from utils.audio_processing import extract_features
import tensorflow as tf

# Load ML models (ensure you have trained/saved these)
gender_model = tf.keras.models.load_model('model/gender_model.h5')
age_model = tf.keras.models.load_model('model/age_model.h5')
emotion_model = tf.keras.models.load_model('model/emotion_model.h5')

# Emotion labels (edit as per your model)
EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Angry']

def predict(audio_path):
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=0)
    
    # Gender Prediction
    gender_pred = gender_model.predict(features)
    gender = 'male' if gender_pred[0][0] > 0.5 else 'female'
    
    if gender == 'female':
        return "Upload male voice.", None, None

    # Age Prediction
    age = int(age_model.predict(features)[0][0])
    if age > 60:
        # Emotion Prediction
        emotion_probs = emotion_model.predict(features)
        emotion = EMOTION_LABELS[np.argmax(emotion_probs)]
        return f"Age: {age} (Senior Citizen)", "Senior Citizen", emotion
    else:
        return f"Age: {age}", None, None

def upload_and_predict():
    audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if not audio_path:
        return

    result, senior, emotion = predict(audio_path)
    result_label.config(text=result)
    if senior:
        emotion_label.config(text=f"Detected Emotion: {emotion}")
    else:
        emotion_label.config(text="")

root = tk.Tk()
root.title("Age & Emotion Detection from Voice")

upload_btn = tk.Button(root, text="Upload Voice Note", command=upload_and_predict)
upload_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=('Arial', 14))
result_label.pack(pady=10)

emotion_label = tk.Label(root, text="", font=('Arial', 12))
emotion_label.pack(pady=5)

root.mainloop()