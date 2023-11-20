import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load the dataset
df = pd.read_csv('Training_set_moe.csv')

# Extract peptide sequences from the 'peptide_sequence' column
peptide_sequences = df['peptide_sequence'].tolist()

# Tokenize the sequences
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(peptide_sequences)
total_chars = len(tokenizer.word_index) + 1

# Create input sequences and their corresponding output sequences
sequences = []
for seq in peptide_sequences:
    tokenized_seq = tokenizer.texts_to_sequences([seq])[0]
    for i in range(1, len(tokenized_seq)):
        n_gram_sequence = tokenized_seq[:i+1]
        sequences.append(n_gram_sequence)

max_sequence_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=total_chars)  # One-hot encode the labels

# Build the LSTM model with increased complexity
model = Sequential()
model.add(Embedding(total_chars, 50, input_length=max_sequence_length-1))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(total_chars, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for more epochs
model.fit(X, y, epochs=200, verbose=2)

# Generate novel peptide sequences with higher temperature for more randomness
def generate_sequences(seed_text, next_words, model, max_sequence_length, num_sequences=1, temperature=1.5):
    generated_sequences = []
    for _ in range(num_sequences):
        generated_sequence = seed_text
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([generated_sequence])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)[0]
            predicted_probs = predicted_probs / np.sum(predicted_probs)  # Normalize probabilities to sum to 1
            predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
            output_word = tokenizer.index_word.get(predicted_index, "")
            generated_sequence += " " + output_word
        generated_sequences.append(generated_sequence)
    return generated_sequences

# Example usage with higher temperature for more randomness
seed_sequence = "H"
generated_sequences = generate_sequences(seed_sequence, 8, model, max_sequence_length, num_sequences=5, temperature=1.5)
print("Generated Sequences:", generated_sequences)

