import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

# Load saved models and tools
model = tf.keras.models.load_model('model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as enc:
    label_encoder = pickle.load(enc)

# Streamlit UI
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter the news text:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=500)
        prediction = model.predict(padded)
        label = label_encoder.inverse_transform([np.argmax(prediction)])

        if label[0].lower() == 'fake':
            st.error(f"🚨 This news is likely **FAKE**.")
        else:
            st.success(f"✅ This news is likely **REAL**.")
