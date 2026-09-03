"""
export_web_model.py
Exports the trained Keras model's weights as plain JSON so the browser-side
app (web/app.js) can run the forward pass with a tiny hand-written matrix
multiply — no TensorFlow.js / model-conversion tooling required.

Run after train.py. Writes:
  web/model/weights.json  (Dense layer weights/biases + activations)
  web/model/words.json    (vocabulary, copied from model/words.json)
  web/model/classes.json  (intent classes, copied from model/classes.json)
  web/intents.json        (copied from data/intents.json, for responses)
"""
import json
import os
import shutil

from tensorflow.keras.models import load_model

OUT_DIR = "web/model"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = load_model("model/chatbot_model.h5")

    layers = []
    dense_layers = [l for l in model.layers if l.__class__.__name__ == "Dense"]
    for i, layer in enumerate(dense_layers):
        w, b = layer.get_weights()
        activation = layer.get_config()["activation"]  # "relu" or "softmax"
        layers.append({
            "w": w.tolist(),  # shape [in_dim, out_dim]
            "b": b.tolist(),  # shape [out_dim]
            "activation": activation,
        })

    with open(f"{OUT_DIR}/weights.json", "w") as f:
        json.dump({"layers": layers}, f)

    shutil.copy("model/words.json", f"{OUT_DIR}/words.json")
    shutil.copy("model/classes.json", f"{OUT_DIR}/classes.json")
    shutil.copy("data/intents.json", "web/intents.json")

    print(f"Exported {len(layers)} dense layers to {OUT_DIR}/weights.json")
    print("Copied words.json, classes.json to web/model/, intents.json to web/")


if __name__ == "__main__":
    main()
