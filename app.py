import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

# Set page configuration
st.set_page_config(page_title="IMDB Movie Review Classifier", layout="wide")

# Title as requested
st.title("IMDB Movie Review Classifier by Reshma")

# Constants
NUM_WORDS = 10000
MAXLEN = 500
MODEL_DIR = "saved_models"
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_best.h5")

@st.cache_resource
def load_classification_model():
    if os.path.exists(LSTM_MODEL_PATH):
        return load_model(LSTM_MODEL_PATH)
    else:
        st.error(f"Model file not found at {LSTM_MODEL_PATH}. Please run the notebook to train the model first.")
        return None

@st.cache_data
def load_imdb_data():
    # Load data
    word_index = imdb.get_word_index()
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
    return (x_test, y_test), word_index

def decode_review(seq, word_index):
    reverse_word_index = {value+3: key for (key, value) in word_index.items()}
    reverse_word_index[0] = '<PAD>'
    reverse_word_index[1] = '<START>'
    reverse_word_index[2] = '<UNK>'
    reverse_word_index[3] = 'the'
    return ' '.join([reverse_word_index.get(i, '?') for i in seq if i != 0])

# Load resources
with st.spinner("Loading model and data..."):
    model = load_classification_model()
    (x_test, y_test), word_index = load_imdb_data()

if model and len(x_test) > 0:
    st.markdown("### Sample Movie Reviews and Classification Results")
    
    # Select 5 samples
    indices = range(5)
    
    for i in indices:
        with st.container():
            st.markdown(f"#### Review #{i+1}")
            
            # Get the raw text
            raw_seq = x_test[i]
            text = decode_review(raw_seq, word_index)
            
            # Preprocess for model
            padded_seq = pad_sequences([raw_seq], maxlen=MAXLEN, padding='pre', truncating='pre')
            
            # Predict
            prob = model.predict(padded_seq, verbose=0)[0,0]
            pred_label = "Positive" if prob >= 0.5 else "Negative"
            actual_label = "Positive" if y_test[i] == 1 else "Negative"
            
            # Color code the prediction
            color = "green" if pred_label == actual_label else "red"
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.text_area("Review Text", value=text, height=150, disabled=True, key=f"review_{i}")
            
            with col2:
                st.markdown(f"**Actual:** {actual_label}")
                st.markdown(f"**Predicted:** :{color}[{pred_label}]")
                st.markdown(f"**Confidence:** {prob:.4f}")
                
            st.divider()
