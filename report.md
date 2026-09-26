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
[Azure AI Speech SDK (JS, browser)]  --(speech-to-text, using a short-lived
     |    ^                            token minted server-side)--> Recognized text (displayed)
     |    | token
     |    |
     |  [Azure Function: /api/get-speech-token]
     |  (Speech resource key kept as a secret Environment Variable
     |   on the Static Web App — never sent to the client)
     v
[Bag-of-Words vectorizer (JS)]  --(preprocessing)-->  Feature vector
     |
     v
[Feedforward NN forward pass (hand-written JS,        --(intent classification)-->  Predicted intent + confidence
 using weights exported from the Azure ML-trained Keras model)]
     |
     v
[Response selector]  --(random choice from matched intent)-->  Chatbot response (displayed)
```

- **Speech Recognition**: **Azure AI Speech** (Cognitive Services Speech resource, F0 free tier), accessed via the Speech SDK for JavaScript in the browser. The browser first calls `/api/get-speech-token` — an Azure Function that exchanges the Speech resource's secret key (stored as an Environment Variable on the Static Web App, never in source control or client code) for a short-lived (10-minute) authorization token — then uses that token to run `recognizeOnceAsync`. A manual text-input fallback is always available.
- **Deep Learning model**: A feedforward neural network built with Keras/TensorFlow for intent classification, trained on **Azure Machine Learning** (workspace `voice-chatbot-ml`, run on a Compute Instance) on a custom dataset of intents.
- **Web app / deployment**: The trained model's weights are exported to JSON (`export_web_model.py`) and the forward pass (matrix multiply + ReLU + softmax) is reproduced in a small hand-written JavaScript module — no TensorFlow.js runtime needed client-side. The static site (`web/`) plus the token-minting Azure Function (`api/`) are deployed together as an **Azure Static Web App** (Free tier), auto-built and deployed via a GitHub Actions workflow on every push to `main`.

  *Note*: The original plan used the browser's built-in Web Speech API and GitHub Pages hosting (after an earlier pivot away from Hugging Face Spaces, which requires a paid plan to run Gradio/Docker apps). The project was further migrated to use Azure services for speech recognition, model training, and hosting end-to-end, using an existing Azure subscription. The client-side intent-classification logic (bag-of-words + hand-written JS forward pass) was kept unchanged through both pivots — verified to produce numerically identical predictions to the Python model (see §7).

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
Trained via `train.py`, executed on an **Azure Machine Learning Compute Instance** (workspace `voice-chatbot-ml`) on the 69 training patterns (tokenizer simplified to a plain regex word-split, dropping NLTK lemmatization, so the exact same preprocessing logic could be reproduced in JavaScript for the client-side deployment):

| Metric | Value |
|---|---|
| Final training accuracy | **98.55%** |
| Final training loss | **0.0616** |
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
- **Platform**: Azure Static Web Apps (Free tier) — hosts the static site (`web/`) and a companion Azure Function (`api/get-speech-token`) together, auto-deployed via a GitHub Actions workflow on every push to `main`. Model inference and intent classification run entirely client-side (no backend needed for that part); the only server-side piece is minting short-lived Azure Speech tokens so the Speech resource key is never exposed to the browser.
- **Training**: Azure Machine Learning (workspace `voice-chatbot-ml`), Compute Instance, `train.py` + `export_web_model.py`.
- **Speech recognition**: Azure AI Speech resource (`voice-chatbot-speech`, F0 free tier, East US region).
- **Live link**: **https://lively-smoke-01c7d2900.3.azurestaticapps.net**
- Verified end-to-end on the live deployment: page loads, model/vocab/classes/intents all fetch correctly, `/api/get-speech-token` returns a valid short-lived token without exposing the Speech key, microphone-based recognition via the Azure Speech SDK correctly transcribes speech and produces a matching intent + response, and the text-input fallback works as well.

## 8a. Smart-Answers Layer & UI Improvements (post-deployment addition)
After the initial deployment, the web app was extended with:
- **A rule-based "smart answers" intercept** in `web/app.js`, checked before
  the NN classifier: live **weather** (via the free, keyless
  [Open-Meteo](https://open-meteo.com) API, using device geolocation or a
  named city), **current time**, **today's date**, and **simple arithmetic**
  (evaluated with a small hand-written parser, no `eval`). These require
  real/computed data the fixed-response NN classifier structurally can't
  produce, so they deliberately sit *in front of*, not inside, the trained
  model — `chatbot_model.py`, `train.py`, and `model/` are unchanged.
- **UI/UX improvements**: live interim captions while speaking (via the
  Speech SDK's `recognizing` event) with a stop/cancel control, a "thinking"
  indicator while a reply is prepared, a clear-chat button, quick
  suggestion chips (Weather / Time / Quick math / Joke / Help), input
  disabled during in-flight requests, autofocus, and a mobile-friendly
  sticky control bar.
- No new Azure resources or secrets were required (Open-Meteo and the
  browser Geolocation API are both keyless, client-side only).

## 8b. Dashboard Redesign (post-deployment addition)
The single-card chat UI was rebuilt into a full dashboard: a collapsible
sidebar with session history (grouped Today / Yesterday / Last 7 Days /
Older, persisted in `localStorage`), header actions for exporting a
transcript and clearing/starting sessions, a settings panel that wires
real microphone and recognition-language choices into the Speech SDK, and
"telemetry cards" for weather and math replies instead of plain sentences.
The underlying speech recognition, NN intent classifier, and smart-answers
layer (§8a) are unchanged — this was a UI/UX layer rebuild, not a model or
architecture change, so no retraining or new Azure resources were needed.

## 9. Limitations & Future Work
- The bag-of-words model has no notion of semantic similarity — out-of-vocabulary or very different phrasing can be misclassified (observed: nonsense input was occasionally matched to `greeting` with high confidence). A larger dataset or a pretrained sentence-embedding-based classifier would improve robustness.
- The Azure Speech SDK for JavaScript requires microphone access and works in all modern browsers, but the free F0 Speech tier caps usage at 5 audio hours/month — sufficient for a lab demo, but a paid tier would be needed for production-scale usage.
- The dataset is small and hand-authored; a production system would benefit from a larger, more diverse intents dataset.
