import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pickle

# Load the trained model and tokenizer
model = load_model("sentimentanlysis.h5")  # Ensure the model file is in the same directory
with open("tokenizer.pkl", "rb") as handle:  # Replace with the actual tokenizer file
    tokenizer = pickle.load(handle)

# Function to preprocess input text
def preprocess_text(input_text, tokenizer, max_len=50):
    """Preprocess user input for model prediction."""
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    return padded_sequence

# Streamlit app layout
st.title("Sentiment Analysis Prediction")
st.write("Enter a text below to determine if the sentiment is **Positive** or **Negative**.")

# Input text box
user_input = st.text_area("Enter text:", value="", height=150)

# Predict sentiment
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.write("Please enter some text to analyze.")
    else:
        # Preprocess the input
        preprocessed_input = preprocess_text(user_input, tokenizer)

        # Predict sentiment
        prediction = model.predict(preprocessed_input).flatten()[0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"

        # Display the result
        st.write(f"### Prediction: **{sentiment}**")
        st.write(f"Confidence Score: {prediction:.2f}")
