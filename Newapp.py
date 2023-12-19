import streamlit as st
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf

def load_text_generation_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def load_models():
    # Load the text generation model
    text_generation_model = load_text_generation_model('LSTM_model.h5')

    # Load the tokenizer used during training
    tokenizer_path = 'plaintext_tokenizer.pkl'  # Use the appropriate tokenizer file
    with open(tokenizer_path, 'rb') as f:
        plaintext_tokenizer = pickle.load(f)

    return text_generation_model, plaintext_tokenizer

def decrypt_text(model, input_text, tokenizer, max_length):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_length, padding="post", truncating="post")

    # Print the input sequence shape for debugging
    print("Input Shape:", input_seq.shape)

    # Reshape the input sequence to match the model's input shape
    input_seq = input_seq.reshape((1, input_seq.shape[1], 1))

    # Perform the prediction
    generated_sequence = model.predict(input_seq)

    # Print the shape of the generated sequence for debugging
    print("Generated Sequence Shape:", generated_sequence.shape)

    decrypted_text = tokenizer.sequences_to_texts(generated_sequence.argmax(axis=-1))
    return decrypted_text[0].strip()


def main():
    max_length = 50 
    st.title("Text Decryption App")

    # Load the models
    text_generation_model, plaintext_tokenizer = load_models()

    # Print the model summary
    st.subheader("Model Summary:")
    st.text(text_generation_model.summary())

    # Upload encrypted text and key through Streamlit
    encrypted_text = st.text_input("Enter the encrypted text:")
    encryption_key = st.text_input("Enter the encryption key:")

    if encrypted_text and encryption_key:
        # Perform text decryption using the model
        decrypted_text = decrypt_text(text_generation_model, encrypted_text, plaintext_tokenizer, max_length)

        # Display the decrypted text
        st.subheader("Decrypted Text:")
        st.text(decrypted_text)

if __name__ == "__main__":
    main()
