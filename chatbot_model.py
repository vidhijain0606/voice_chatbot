"""
chatbot_model.py
Preprocessing utilities (tokenize/bag-of-words) and the Keras
intent-classification model definition used by train.py.

Tokenization is deliberately a plain regex word-split (no NLTK
lemmatization) so that the exact same logic can be reproduced in
JavaScript for the client-side (TensorFlow.js) inference used by the
deployed static site — see web/app.js's tokenize().
"""
import json
import pickle
import random
import re

import numpy as np

IGNORE_CHARS = {"?", "!", ".", ","}


def tokenize(sentence: str):
    """Lowercase and split into word tokens (letters/apostrophes only)."""
    tokens = re.findall(r"[a-z']+", sentence.lower())
    return [t for t in tokens if t not in IGNORE_CHARS]


def bag_of_words(sentence: str, vocab: list) -> np.ndarray:
    """Convert a sentence into a bag-of-words vector against the given vocab."""
    sentence_words = tokenize(sentence)
    bag = np.zeros(len(vocab), dtype=np.float32)
    for w in sentence_words:
        if w in vocab:
            bag[vocab.index(w)] = 1.0
    return bag


def build_model(input_size: int, output_size: int):
    """Feedforward NN for intent classification: Dense-Dropout-Dense-Dropout-Softmax."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import SGD

    model = Sequential([
        Dense(128, input_shape=(input_size,), activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(output_size, activation="softmax"),
    ])
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model


def load_artifacts(model_dir="model"):
    """Load trained model + vocab + classes + intents for inference."""
    from tensorflow.keras.models import load_model

    model = load_model(f"{model_dir}/chatbot_model.h5")
    words = pickle.load(open(f"{model_dir}/words.pkl", "rb"))
    classes = pickle.load(open(f"{model_dir}/classes.pkl", "rb"))
    intents = json.load(open("data/intents.json"))
    return model, words, classes, intents


def predict_intent(sentence: str, model, words, classes, threshold: float = 0.5):
    """Return (tag, confidence) for the highest scoring intent, or ('fallback', conf) if below threshold."""
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]
    idx = int(np.argmax(res))
    confidence = float(res[idx])
    if confidence < threshold:
        return "fallback", confidence
    return classes[idx], confidence


def get_response(tag: str, intents_data: dict) -> str:
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    for intent in intents_data["intents"]:
        if intent["tag"] == "fallback":
            return random.choice(intent["responses"])
    return "Sorry, I don't have a response for that."
