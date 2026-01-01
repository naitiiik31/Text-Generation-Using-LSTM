import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Quote Generator",
    page_icon="📝",
    layout="centered"
)

st.title("📝 Quote Generator using LSTM")
st.write("Generate quotes using a trained LSTM model")

@st.cache_resource
def load_assets():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_assets()

@st.cache_data
def load_data():
    return pd.read_csv("qoute_dataset.csv")

data = load_data()

def generate_text(seed_text, next_words=20):
    output = seed_text
    for _ in range(next_words):
        sequence = tokenizer.texts_to_sequences([output])[0]
        sequence = pad_sequences([sequence], maxlen=max_len, padding="pre")
        prediction = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(prediction)
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output += " " + word
                break
    return output

st.subheader("Enter Starting Text")

seed_text = st.text_input(
    "Type a few words to start the quote:",
    value="life"
)

num_words = st.slider(
    "Number of words to generate",
    min_value=5,
    max_value=50,
    value=20
)

if st.button("Generate Quote"):
    if seed_text.strip() == "":
        st.warning("Please enter some starting text.")
    else:
        with st.spinner("Generating..."):
            result = generate_text(seed_text, num_words)
        st.success("Generated Quote:")
        st.write(result)

