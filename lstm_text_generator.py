import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import string
import requests


print("--- Step 1: Downloading and Preprocessing Dataset ---")

DATASET_URL = "https://www.gutenberg.org/files/100/100-0.txt"
response = requests.get(DATASET_URL)
raw_text = response.text


text_corpus = raw_text[30000:250000]


text_corpus = text_corpus.lower()
text_corpus = text_corpus.translate(str.maketrans('', '', string.punctuation))


unique_chars = sorted(list(set(text_corpus)))
vocab_size = len(unique_chars)
char_to_idx = {ch: idx for idx, ch in enumerate(unique_chars)}
idx_to_char = {idx: ch for idx, ch in enumerate(unique_chars)}

print(f"Total Corpus Slice Volume: {len(text_corpus)} characters")
print(f"Total Vocabulary Token Space (V): {vocab_size}")

MAX_SEQUENCE_LEN = 40
SLIDE_STEP_SIZE = 3
input_windows = []
target_tokens = []

for i in range(0, len(text_corpus) - MAX_SEQUENCE_LEN, SLIDE_STEP_SIZE):
    input_windows.append(text_corpus[i : i + MAX_SEQUENCE_LEN])
    target_tokens.append(text_corpus[i + MAX_SEQUENCE_LEN])

print(f"Total Vector Arrays Extracted: {len(input_windows)}")


X_train = np.zeros((len(input_windows), MAX_SEQUENCE_LEN), dtype=np.int32)
y_train = np.zeros((len(input_windows), vocab_size), dtype=np.bool_)

for i, window in enumerate(input_windows):
    for t, char in enumerate(window):
        X_train[i, t] = char_to_idx[char]
    y_train[i, char_to_idx[target_tokens[i]]] = 1


print("\n--- Step 2: Formulating Stacked LSTM Topology ---")
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=MAX_SEQUENCE_LEN),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(vocab_size, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)
model.summary()


print("\n--- Step 3: Initiating Training Framework ---")

callbacks_list = [
    EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('optimal_lstm_weights.h5', monitor='loss', save_best_only=True)
]


model.fit(
    X_train, y_train, 
    batch_size=128, 
    epochs=15, 
    callbacks=callbacks_list
)


print("\n--- Step 4: Instantiating Text Generation Inferencing Engine ---")

def generate_synthetic_text(seed_phrase, output_len=150, temperature=0.6):
    """
    Generates novel text tokens by computing soft-max probabilities 
    and applying scalable thermal distribution parameters (Temperature).
    """
    processed_seed = seed_phrase.lower().translate(str.maketrans('', '', string.punctuation))
    
    
    if len(processed_seed) < MAX_SEQUENCE_LEN:
        processed_seed = processed_seed.rjust(MAX_SEQUENCE_LEN)
    else:
        processed_seed = processed_seed[-MAX_SEQUENCE_LEN:]
        
    running_context = processed_seed
    output_string = processed_seed
    
    for _ in range(output_len):
        x_input = np.zeros((1, MAX_SEQUENCE_LEN), dtype=np.int32)
        for t, char in enumerate(running_context):
            if char in char_to_idx:
                x_input[0, t] = char_to_idx[char]
                

        raw_predictions = model.predict(x_input, verbose=0)[0]
        
    
        scaled_logits = np.log(raw_predictions + 1e-7) / temperature
        exponentiated_logits = np.exp(scaled_logits)
        probabilities = exponentiated_logits / np.sum(exponentiated_logits)
        
    
        selected_index = np.random.choice(range(vocab_size), p=probabilities)
        predicted_char = idx_to_char[selected_index]
        
        output_string += predicted_char
        running_context = running_context[1:] + predicted_char
        
    return output_string

if __name__ == "__main__":
    
    sample_seed = "shall i compare thee to a summers day"
    generated_passage = generate_synthetic_text(sample_seed, output_len=150, temperature=0.5)
    print(f"\nGenerated Result:\n{generated_passage}")