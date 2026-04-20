from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

class CompatibleEmbedding(tf.keras.layers.Embedding):
    def __init__(self, *args, quantization_config=None, **kwargs):
        # Ignore unsupported legacy key from older saved model configs.
        super().__init__(*args, **kwargs)


class CompatibleDense(tf.keras.layers.Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        # Ignore unsupported legacy key from older saved model configs.
        super().__init__(*args, **kwargs)


# Load model
model = tf.keras.models.load_model(
    "lstm_model.h5",
    custom_objects={
        "Embedding": CompatibleEmbedding,
        "Dense": CompatibleDense,
    },
)

# Load tokenizer
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

max_len = 20
index_to_word = {index: word for word, index in tokenizer.word_index.items()}


def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding="pre")
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
    return index_to_word.get(predicted)


def predict_sequence(seed_text, num_words=10):
    generated_words = []
    current_text = seed_text.strip()

    for _ in range(num_words):
        next_word = predict_next_word(current_text)
        if not next_word:
            break
        generated_words.append(next_word)
        current_text = f"{current_text} {next_word}"

    return generated_words, current_text


@app.get("/predict")
def predict(text: str, words: int = 10):
    words = max(1, min(words, 50))
    generated_words, generated_text = predict_sequence(text, words)
    return {
        "input_text": text,
        "generated_words": generated_words,
        "generated_text": generated_text,
    }