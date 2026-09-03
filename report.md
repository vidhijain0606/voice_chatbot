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
[Bag-of-Words vectorizer]  --(preprocessing)-->  Feature vector
     |
     v
[Feedforward Neural Network]  --(intent classification)-->  Predicted intent + confidence
     |
     v
[Response selector]  --(random choice from matched intent)-->  Chatbot response (displayed)
```

- **Speech Recognition**: Browser-native `SpeechRecognition` / `webkitSpeechRecognition` API (Web Speech API). Runs client-side in the user's browser (best support in Chrome/Edge); transcribes spoken audio to text with no external API key required. A manual text-input fallback is provided for browsers without support (e.g. Firefox).
- **Deep Learning model**: A feedforward neural network built with Keras/TensorFlow for intent classification, trained on a custom dataset of intents.
- **Web app / deployment**: Gradio `Blocks` app combining the JS-based voice capture with the Python inference backend, deployed on Hugging Face Spaces.

## 3. Dataset
A custom-built dataset (`data/intents.json`) of **15 intents**, each with several example training phrases ("patterns") and candidate responses — a standard approach for small-scale intent-based chatbots.

| Property | Value |
|---|---|
| Number of intents (classes) | 15 |
| Intents | greeting, goodbye, thanks, about, help, name, hours, weather_smalltalk, joke, mood, identity_user, purpose, compliment, confused, fallback |
| Total training patterns (sentences) | 69 |
| Vocabulary size (unique tokens after preprocessing) | 97 |

Each intent contains 4–8 example patterns (e.g. `greeting`: "Hi", "Hello", "Hey", "Good morning", ...) and 2–3 candidate responses, one of which is chosen at random at inference time for variety.

## 4. Preprocessing / Methodology
1. **Tokenization**: Each pattern is lowercased and tokenized using NLTK's `word_tokenize`.
2. **Lemmatization**: Tokens are lemmatized with NLTK's `WordNetLemmatizer` to normalize word forms (e.g. "hours" → "hour").
3. **Vocabulary construction**: All unique lemmatized tokens across the dataset form the vocabulary (bag-of-words dimension = 97).
4. **Feature vectorization**: Each training sentence is converted into a binary bag-of-words vector (1 if the vocabulary word is present in the sentence, else 0).
5. **Label encoding**: Each intent tag is one-hot encoded across the 15 classes.
6. **Inference**: A new sentence goes through the same tokenize → lemmatize → bag-of-words pipeline, is fed to the trained model, and the predicted class with the highest softmax probability is chosen (if below a confidence threshold of 0.5, a `fallback` response is returned instead).

## 5. Model Architecture
A simple, fast-training feedforward neural network (classic approach for small-vocabulary intent classifiers):

```
Input (97-dim bag-of-words vector)
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
Trained locally via `train.py` on the 69 training patterns:

| Metric | Value |
|---|---|
| Final training accuracy | **100%** (1.0000) |
| Final training loss | **0.0256** |
| Epochs | 200 |
| Training samples | 69 |
| Vocabulary size | 97 |
| Intent classes | 15 |

The model fits the training set perfectly, which is expected and acceptable for a small, hand-crafted intent dataset (the goal here is reliable intent matching, not open-domain generalization).

## 7. Sample Conversations (verified end-to-end)

| User input (voice/text) | Predicted intent | Confidence | Chatbot response |
|---|---|---|---|
| "hello there" | greeting | 1.00 | Hello! How can I help you today? |
| "what is your name" | about | 1.00 | I'm an AI assistant trained to understand and respond to your voice and text. |
| "tell me a joke" | joke | 1.00 | Why did the computer go to therapy? It had too many bytes of emotional baggage! |
| "bye bye" | goodbye | 1.00 | Goodbye! Have a great day. |
| "thanks a bunch" | thanks | 1.00 | You're welcome! |

## 8. Deployment
- **Platform**: Hugging Face Spaces (Gradio SDK).
- **Live link**: _[fill in after deployment — see README.md for deployment steps]_
- The app was verified locally end-to-end (server startup, page load, and the full voice/text → intent → response pipeline via the Gradio API) before deployment.

## 9. Limitations & Future Work
- The bag-of-words model has no notion of semantic similarity — out-of-vocabulary or very different phrasing can be misclassified (observed: nonsense input was occasionally matched to `greeting` with high confidence). A larger dataset or a pretrained sentence-embedding-based classifier would improve robustness.
- Web Speech API accuracy and browser support (Chrome/Edge only) is a client-side constraint; a server-side ASR model (e.g. Whisper) would broaden browser compatibility at the cost of added latency/hosting complexity.
- The dataset is small and hand-authored; a production system would benefit from a larger, more diverse intents dataset.
