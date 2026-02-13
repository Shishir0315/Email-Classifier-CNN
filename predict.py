import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

# Load the model
model = tf.keras.models.load_model('cnn_email_model.h5')

# Re-initialize tokenizer with same params
data_path = r'c:\Users\student\Desktop\classification dataset\email_classification_dataset.csv'
df = pd.read_csv(data_path)
max_words = 10000
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['email'].astype(str).values)

def predict_email(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded, verbose=0)
    label = 'spam' if prediction[0][0] > 0.5 else 'ham'
    confidence = prediction[0][0] if label == 'spam' else 1 - prediction[0][0]
    return label, confidence

# Test with some examples
test_emails = [
    "Subject: Hello friend. Long time no see. Let's grab a coffee tomorrow?",
    "Subject: URGENT! You won $10,000 lottery award. Click here to claim your prize now!!"
]

for email in test_emails:
    label, conf = predict_email(email)
    print(f"Email: {email[:50]}...")
    print(f"Prediction: {label} (Confidence: {conf:.4f})\n")
