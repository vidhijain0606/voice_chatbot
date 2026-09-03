"""
train.py
Builds the bag-of-words training set from data/intents.json, trains the
intent-classification neural network, and saves the model + vocab + classes
into model/ for use by app.py.
"""
import json
import pickle
import random

import numpy as np

from chatbot_model import tokenize, build_model, IGNORE_CHARS

random.seed(42)
np.random.seed(42)


def main():
    intents = json.load(open("data/intents.json"))

    words = []
    classes = []
    documents = []  # (tokenized_pattern, tag)

    for intent in intents["intents"]:
        tag = intent["tag"]
        if tag not in classes:
            classes.append(tag)
        for pattern in intent["patterns"]:
            tokens = tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, tag))

    words = sorted(set(w for w in words if w not in IGNORE_CHARS))
    classes = sorted(classes)

    print(f"{len(documents)} training patterns")
    print(f"{len(classes)} intent classes: {classes}")
    print(f"{len(words)} unique vocabulary words")

    # Build training data: bag-of-words input, one-hot intent output.
    train_x, train_y = [], []
    for tokens, tag in documents:
        bag = [1 if w in tokens else 0 for w in words]
        output_row = [0] * len(classes)
        output_row[classes.index(tag)] = 1
        train_x.append(bag)
        train_y.append(output_row)

    train_x = np.array(train_x, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.float32)

    model = build_model(input_size=len(words), output_size=len(classes))
    history = model.fit(train_x, train_y, epochs=200, batch_size=8, verbose=1)

    final_acc = history.history["accuracy"][-1]
    final_loss = history.history["loss"][-1]
    print(f"\nFinal training accuracy: {final_acc:.4f}, loss: {final_loss:.4f}")

    model.save("model/chatbot_model.h5")
    pickle.dump(words, open("model/words.pkl", "wb"))
    pickle.dump(classes, open("model/classes.pkl", "wb"))
    # JSON copies for the browser-side (TensorFlow.js) inference in web/app.js.
    json.dump(words, open("model/words.json", "w"), indent=2)
    json.dump(classes, open("model/classes.json", "w"), indent=2)
    print("Saved model/chatbot_model.h5, model/words.pkl, model/classes.pkl, model/words.json, model/classes.json")

    # Persist final metrics for the report.
    with open("model/train_metrics.json", "w") as f:
        json.dump({
            "final_accuracy": final_acc,
            "final_loss": final_loss,
            "epochs": 200,
            "num_intents": len(classes),
            "num_patterns": len(documents),
            "vocab_size": len(words),
        }, f, indent=2)


if __name__ == "__main__":
    main()
