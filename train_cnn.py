import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

# Load dataset
data_path = r'c:\Users\student\Desktop\classification dataset\email_classification_dataset.csv'
df = pd.read_csv(data_path)

# 1. Preprocessing
# Combine Subject and Body if they are separate, but here 'email' column contains everything
X = df['email'].astype(str).values
y = df['label'].values

# Label Encoding
le = LabelEncoder()
y = le.fit_transform(y) # ham -> 0, spam -> 1

# Tokenization
max_words = 10000
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(sequences, maxlen=max_len)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# 2. Build CNN Model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Train Model
epochs = 10
batch_size = 32

history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1
)

# 4. Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# 5. Save Results
model.save('cnn_email_model.h5')
with open('model_results.txt', 'w') as f:
    f.write(f'Model: 1D CNN for Email Classification\n')
    f.write(f'Test Accuracy: {accuracy:.4f}\n')
    f.write(f'Loss: {loss:.4f}\n')

# Plotting metrics
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()
plt.savefig('training_plots.png')

print("Model training complete and saved.")
