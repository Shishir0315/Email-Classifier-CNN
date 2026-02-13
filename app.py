from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load Model
MODEL_PATH = 'cnn_email_model.h5'
DATA_PATH = 'email_classification_dataset.csv'

model = tf.keras.models.load_model(MODEL_PATH)

# Re-initialize Tokenizer (simulating the one used in training)
df = pd.read_csv(DATA_PATH)
max_words = 10000
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['email'].astype(str).values)

def predict_email(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded, verbose=0)
    label = 'Spam' if prediction[0][0] > 0.5 else 'Ham (Safe)'
    confidence = float(prediction[0][0] if label == 'Spam' else 1 - prediction[0][0])
    return label, round(confidence * 100, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get('text', '')
    if not email_text:
        return jsonify({'error': 'No text provided'}), 400
    
    label, confidence = predict_email(email_text)
    return jsonify({
        'prediction': label,
        'confidence': f"{confidence}%",
        'is_spam': label == 'Spam'
    })

if __name__ == '__main__':
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
