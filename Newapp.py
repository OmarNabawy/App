import streamlit as st
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import tensorflow as tf

def load_text_generation_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

text_generation_model = load_text_generation_model('LSTM_model.h5')

# Load the tokenizer used during training
tokenizer_path = 'plaintext_tokenizer.pkl'  # Use the appropriate tokenizer file
with open(tokenizer_path, 'rb') as f:
    plaintext_tokenizer = pickle.load(f)

max_length = 50

def decrypt_text(model, input_text, tokenizer, max_length):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_length, padding="post", truncating="post")

    # Print the input sequence shape for debugging
    print("Input Shape:", input_seq.shape)

    generated_sequence = model.predict(input_seq)
    decrypted_text = tokenizer.sequences_to_texts(generated_sequence.argmax(axis=-1))
    return decrypted_text[0].strip()


def main():
    st.title("Text Decryption App")

    # Upload encrypted text and key through Streamlit
    encrypted_text = st.text_input("Enter the encrypted text:")
    encryption_key = st.text_input("Enter the encryption key:")

    if encrypted_text and encryption_key:
        # Perform text decryption using the model
        decrypted_text = decrypt_text(text_generation_model, encrypted_text, plaintext_tokenizer, max_length)

        # Display the decrypted text
        st.markdown("Decrypted Text:")
        st.text(decrypted_text)

text_generation_model.summary()


if __name__ == "__main__":
    main()
