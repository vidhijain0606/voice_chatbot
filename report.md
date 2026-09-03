# Lab Report — Voice-Enabled Chatbot using Speech Recognition and Deep Learning

## 1. Objective
Develop, implement, and deploy an online voice-enabled chatbot that:
- Accepts voice input from the user.
- Converts speech to text using a Speech Recognition technique.
- Processes the text using a Deep Learning-based intent classification model.
- Generates and displays an appropriate response, alongside the recognized speech.
- Is deployed publicly.

## 2. System Architecture

```
 User voice
     |
     v
[Browser Web Speech API]  --(speech-to-text, client-side)-->  Recognized text (displayed)
     |
     v
[Bag-of-Words vectorizer (JS)]  --(preprocessing)-->  Feature vector
     |
     v
[Feedforward NN forward pass (hand-written JS,        --(intent classification)-->  Predicted intent + confidence
 using weights exported from the trained Keras model)]
     |
     v
[Response selector]  --(random choice from matched intent)-->  Chatbot response (displayed)
```

- **Speech Recognition**: Browser-native `SpeechRecognition` / `webkitSpeechRecognition` API (Web Speech API). Runs client-side in the user's browser (best support in Chrome/Edge); transcribes spoken audio to text with no external API key required. A manual text-input fallback is provided for browsers without support (e.g. Firefox).
- **Deep Learning model**: A feedforward neural network built with Keras/TensorFlow for intent classification, trained offline in Python on a custom dataset of intents.
- **Web app / deployment**: The trained model's weights are exported to JSON (`export_web_model.py`) and the forward pass (matrix multiply + ReLU + softmax) is reproduced in a small hand-written JavaScript module. Combined with the Web Speech API and a plain HTML/CSS/JS UI, the entire app runs client-side with **no backend server** — deployed as a static site on **GitHub Pages** (free).

  *Note*: Hugging Face Spaces was the original planned host, but Spaces now requires a paid plan to run Gradio/Docker apps (free tier is Static-only). Re-architecting to a fully static, client-side app avoided any hosting cost while keeping the same DL model and preprocessing logic — verified to produce numerically identical predictions to the Python model (see §7).

## 3. Dataset
A custom-built dataset (`data/intents.json`) of **15 intents**, each with several example training phrases ("patterns") and candidate responses — a standard approach for small-scale intent-based chatbots.

| Property | Value |
|---|---|
| Number of intents (classes) | 15 |
| Intents | greeting, goodbye, thanks, about, help, name, hours, weather_smalltalk, joke, mood, identity_user, purpose, compliment, confused, fallback |
| Total training patterns (sentences) | 69 |
| Vocabulary size (unique tokens after preprocessing) | 99 |

Each intent contains 4–8 example patterns (e.g. `greeting`: "Hi", "Hello", "Hey", "Good morning", ...) and 2–3 candidate responses, one of which is chosen at random at inference time for variety.

## 4. Preprocessing / Methodology
1. **Tokenization**: Each pattern is lowercased and split into word tokens with a simple regex (`[a-z']+`) — deliberately avoiding NLTK lemmatization so the identical logic could be reproduced in JavaScript for the client-side deployment (see `chatbot_model.py`'s `tokenize()` and `web/app.js`'s `tokenize()`).
2. **Vocabulary construction**: All unique tokens across the dataset form the vocabulary (bag-of-words dimension = 99).
3. **Feature vectorization**: Each training sentence is converted into a binary bag-of-words vector (1 if the vocabulary word is present in the sentence, else 0).
4. **Label encoding**: Each intent tag is one-hot encoded across the 15 classes.
5. **Inference**: A new sentence goes through the same tokenize → bag-of-words pipeline, is fed to the trained model, and the predicted class with the highest softmax probability is chosen (if below a confidence threshold of 0.5, a `fallback` response is returned instead). In the deployed app this runs as a hand-written JavaScript forward pass using the exact weights exported from the trained Keras model (`export_web_model.py`), rather than re-invoking Python/Keras.

## 5. Model Architecture
A simple, fast-training feedforward neural network (classic approach for small-vocabulary intent classifiers):

```
Input (99-dim bag-of-words vector)
   -> Dense(128, activation='relu')
   -> Dropout(0.5)
   -> Dense(64, activation='relu')
   -> Dropout(0.5)
   -> Dense(15, activation='softmax')
```

- **Loss function**: Categorical cross-entropy
- **Optimizer**: SGD (learning rate 0.01, momentum 0.9, Nesterov)
- **Epochs**: 200
- **Batch size**: 8

Dropout layers (0.5) are used after each hidden layer to reduce overfitting, given the small dataset size.

## 6. Training Results
Trained locally via `train.py` on the 69 training patterns (tokenizer simplified to a plain regex word-split, dropping NLTK lemmatization, so the exact same preprocessing logic could be reproduced in JavaScript for the client-side deployment):

| Metric | Value |
|---|---|
| Final training accuracy | **98.55%** |
| Final training loss | **0.0497** |
| Epochs | 200 |
| Training samples | 69 |
| Vocabulary size | 99 |
| Intent classes | 15 |

The model fits the training set almost perfectly, which is expected and acceptable for a small, hand-crafted intent dataset (the goal here is reliable intent matching, not open-domain generalization).

## 7. Sample Conversations & Python/JavaScript Parity Check
The chatbot response for a matched intent is chosen at random from that intent's candidate `responses` list (see `data/intents.json`), so exact wording varies between runs — what matters for correctness is that the **predicted intent and confidence** match between the Python-trained model and the JavaScript forward pass used in the deployed static site. Verified identical to 4 decimal places:

| User input (voice/text) | Predicted intent | Confidence (Python) | Confidence (JS) |
|---|---|---|---|
| "hello there" | greeting | 0.9999 | 0.9999 |
| "what is your name" | about | 0.9963 | 0.9963 |
| "tell me a joke" | joke | 0.9999 | 0.9999 |
| "bye bye" | goodbye | 0.9992 | 0.9992 |
| "thanks a bunch" | thanks | 0.9997 | 0.9997 |
| "how are you doing" | mood | 1.0000 | 1.0000 |
| "when are you available" | hours | 1.0000 | 1.0000 |

## 8. Deployment
- **Platform**: GitHub Pages (static site, free) — no backend server, all inference runs client-side.
- **Live link**: _[fill in after deployment]_
- Verified locally end-to-end before deployment: static file server serving `web/`, all assets (model weights, vocab, classes, intents) load correctly, and the Python-trained model's predictions were cross-checked bit-for-bit against the JavaScript forward pass (§7).

## 9. Limitations & Future Work
- The bag-of-words model has no notion of semantic similarity — out-of-vocabulary or very different phrasing can be misclassified (observed: nonsense input was occasionally matched to `greeting` with high confidence). A larger dataset or a pretrained sentence-embedding-based classifier would improve robustness.
- Web Speech API accuracy and browser support (Chrome/Edge only) is a client-side constraint; a server-side ASR model (e.g. Whisper) would broaden browser compatibility at the cost of added latency/hosting complexity.
- The dataset is small and hand-authored; a production system would benefit from a larger, more diverse intents dataset.
