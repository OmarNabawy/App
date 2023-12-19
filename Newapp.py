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

def sequence_to_text(sequence, tokenizer):
    text = tokenizer.sequences_to_texts([sequence])[0]
    return text.strip()

def decrypt_text(model, input_text, tokenizer, max_length):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_length, padding="post", truncating="post")

    # Perform the prediction
    generated_sequence = model.predict(input_seq)

    # Ensure that generated_sequence is a 1D array
    generated_sequence = np.squeeze(generated_sequence)

    # Reshape the generated sequence to match the expected output
    try:
        generated_sequence = generated_sequence.reshape((generated_sequence.shape[0],))
    except ValueError as e:
        print("Error during reshaping:", e)
        print("Actual Generated Sequence Shape:", generated_sequence.shape)

    # Apply temperature to the generated sequence for sampling
    generated_sequence = np.log(generated_sequence) / 1.0
    probabilities = np.exp(generated_sequence) / np.sum(np.exp(generated_sequence), axis=-1)

    # Sample from the distribution to get the index of the predicted character
    predicted_index = np.random.choice(len(probabilities), p=probabilities)
    
    # Convert the index to text using the tokenizer
    decrypted_text = sequence_to_text([predicted_index], tokenizer)

    return decrypted_text

def main():
    max_length = 40  # Set max_length to 40 to match the model's input shape
    st.title("Text Decryption App")

    # Load the models
    text_generation_model, plaintext_tokenizer = load_models()

    # Print the model summary
    st.subheader("Model Summary:")
    st.text(str(text_generation_model.summary()))

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
