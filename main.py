# main.py

import argparse
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

from resume_parser import parse_resume, extract_text_from_file

# --- Configuration ---
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'resume_classifier.keras')
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer.pickle')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pickle')
MAX_LEN = 500 # Should match the value in model_builder.py

def load_model_and_artifacts():
    """
    Loads the trained model, tokenizer, and label encoder.
    """
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(LABEL_ENCODER_PATH, 'rb') as handle:
            label_encoder = pickle.load(handle)
        return model, tokenizer, label_encoder
    except FileNotFoundError as e:
        print(f"Error loading model artifacts: {e}")
        print("Please run 'python model_builder.py' to train and save the model first.")
        return None, None, None

def classify_resume(text, model, tokenizer, label_encoder):
    """
    Classifies the resume text using the loaded model.
    """
    # Preprocess the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

    # Make prediction
    prediction = model.predict(padded_sequence)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    # Decode the prediction
    predicted_category = label_encoder.inverse_transform([predicted_class_index])[0]
    confidence = prediction[0][predicted_class_index]

    return predicted_category, confidence

def main():
    """
    Main function to run the resume screening engine.
    """
    parser = argparse.ArgumentParser(description="Resume Screening Engine")
    parser.add_argument("--file", type=str, required=True, help="Path to the resume file to be screened.")
    args = parser.parse_args()

    # --- 1. Load Model ---
    print("Loading classification model...")
    model, tokenizer, label_encoder = load_model_and_artifacts()
    if not model:
        return # Exit if model loading failed

    # --- 2. Parse and Classify Resume ---
    print(f"Processing resume: {args.file}...")
    
    # First, just get the text for classification
    resume_text = extract_text_from_file(args.file)
    if not resume_text:
        print("Could not extract text from the resume.")
        return

    # Classify the resume
    category, confidence = classify_resume(resume_text, model, tokenizer, label_encoder)

    # --- 3. Extract Structured Profile ---
    # Now, perform the full parsing to get the structured data
    _, structured_profile = parse_resume(args.file)
    
    if not structured_profile:
        print("Could not generate a structured profile for the resume.")
        return

    # --- 4. Display Results ---
    result = {
        "file_path": args.file,
        "predicted_category": category,
        "confidence_score": float(confidence),
        "extracted_profile": structured_profile
    }

    print("\n--- Screening Result ---")
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()