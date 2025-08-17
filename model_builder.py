# model_builder.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# --- Configuration ---
DATA_PATH = 'data/resumes.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'resume_classifier.keras')
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer.pickle')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pickle')

# Model Hyperparameters
VOCAB_SIZE = 10000
MAX_LEN = 500
EMBEDDING_DIM = 16
OOV_TOKEN = "<OOV>"
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'

def build_and_train_model():
    """
    Loads resume data, preprocesses it, and trains a classification model.
    """
    # --- 1. Load and Prepare Data ---
    print("Loading and preparing data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please create a 'data/resumes.csv' file with 'resume_text' and 'category' columns.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Check if required columns exist
    if 'resume_text' not in df.columns or 'category' not in df.columns:
        print("Error: CSV must contain 'resume_text' and 'category' columns.")
        return

    # Drop rows with missing values
    df.dropna(subset=['resume_text', 'category'], inplace=True)

    texts = df['resume_text'].values
    labels = df['category'].values

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)

    # --- 2. Tokenize and Pad Text ---
    print("Tokenizing and padding text data...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(X_train)

    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_padded = pad_sequences(train_sequences, maxlen=MAX_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_padded = pad_sequences(test_sequences, maxlen=MAX_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

    # --- 3. Build the Model ---
    print("Building the TensorFlow model...")
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    # --- 4. Train the Model ---
    print("Training the model...")
    history = model.fit(train_padded, y_train,
                        epochs=20,
                        validation_data=(test_padded, y_test),
                        verbose=2)

    # --- 5. Save the Model and Artifacts ---
    print("Saving model and artifacts...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Save the model
    model.save(MODEL_PATH)

    # Save the tokenizer
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the label encoder
    with open(LABEL_ENCODER_PATH, 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Model trained and saved successfully to '{MODEL_DIR}/'")
    print(f"Classes: {label_encoder.classes_}")

if __name__ == '__main__':
    build_and_train_model()