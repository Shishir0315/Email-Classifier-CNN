# Email Classification using 1D CNN

This project implements an end-to-end Deep Learning model to classify emails as **Spam** or **Ham**. It uses a 1D Convolutional Neural Network (CNN) architecture to process text sequences.

## 🚀 Features
- **Deep Learning Model**: 1D CNN built with TensorFlow/Keras.
- **Natural Language Processing**: Tokenization and sequence padding for text data.
- **Web Interface**: A modern, responsive Flask-based dashboard for real-time predictions.
- **High Performance**: Significant accuracy on the provided dataset.

## 📂 Project Structure
- `train_cnn.py`: Training script for the model.
- `app.py`: Flask web server for deployment.
- `predict.py`: CLI script for quick predictions.
- `cnn_email_model.h5`: The trained model weights.
- `templates/`: HTML frontend files.

## 🛠️ How to Run

### 1. Train the Model (Optional)
If you want to re-train the model:
```bash
py train_cnn.py
```

### 2. Launch the Web App
To run the interactive dashboard:
```bash
py app.py
```
Then open your browser and go to `http://127.0.0.1:5000`.

### 3. Run a CLI Prediction
```bash
py predict.py
```

## 🧠 Model Architecture
- **Embedding Layer**: Vectorizes text inputs.
- **Conv1D**: Detects local patterns in word sequences.
- **GlobalMaxPooling**: Reduces dimensionality while keeping critical features.
- **Dense/Dropout**: Classifies the features and prevents overfitting.
