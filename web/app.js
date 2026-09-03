/**
 * app.js
 * Voice chatbot:
 *  - Azure AI Speech (Speech SDK for JavaScript) for voice -> text. The
 *    browser fetches a short-lived token from /api/get-speech-token (an
 *    Azure Function -- see api/get-speech-token/index.js) so the Speech
 *    resource key never reaches client code.
 *  - A tiny hand-written feedforward neural network forward pass runs the
 *    Keras-trained intent-classification model (trained via an Azure
 *    Machine Learning job) directly from its exported weights
 *    (model/weights.json) -- no ML library/runtime needed client-side.
 *  - Bag-of-words preprocessing mirrors chatbot_model.py's tokenize()
 *    exactly (lowercase, regex word split on [a-z']+) so predictions match
 *    the Python-trained model bit-for-bit.
 */

const CONFIDENCE_THRESHOLD = 0.5;

let weights = null; // { layers: [{ w, b, activation }, ...] }
let vocab = [];
let classes = [];
let intents = null;

const chatWindow = document.getElementById("chat-window");
const textInput = document.getElementById("text-input");
const speakBtn = document.getElementById("speak-btn");
const sendBtn = document.getElementById("send-btn");
const statusEl = document.getElementById("status");

function setStatus(text) {
  statusEl.textContent = text;
}

function addBubble(role, text, meta) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  if (meta) {
    const metaEl = document.createElement("span");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    bubble.appendChild(metaEl);
  }
  chatWindow.appendChild(bubble);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Must match chatbot_model.py's tokenize(): lowercase, split on [a-z']+.
function tokenize(sentence) {
  const matches = sentence.toLowerCase().match(/[a-z']+/g);
  return matches || [];
}

function bagOfWords(sentence) {
  const tokens = tokenize(sentence);
  const bag = new Array(vocab.length).fill(0);
  for (const t of tokens) {
    const idx = vocab.indexOf(t);
    if (idx !== -1) bag[idx] = 1;
  }
  return bag;
}

// --- Minimal dense-layer forward pass (matches Keras Dense: y = xW + b) ---
function denseForward(x, layer) {
  const { w, b, activation } = layer; // w: [inDim][outDim], b: [outDim]
  const outDim = b.length;
  const y = new Array(outDim).fill(0);
  for (let j = 0; j < outDim; j++) {
    let sum = b[j];
    for (let i = 0; i < x.length; i++) {
      sum += x[i] * w[i][j];
    }
    y[j] = sum;
  }
  if (activation === "relu") {
    return y.map((v) => Math.max(0, v));
  }
  if (activation === "softmax") {
    const max = Math.max(...y);
    const exps = y.map((v) => Math.exp(v - max));
    const sumExp = exps.reduce((a, v) => a + v, 0);
    return exps.map((v) => v / sumExp);
  }
  return y;
}

function forward(bag) {
  let x = bag;
  for (const layer of weights.layers) {
    x = denseForward(x, layer);
  }
  return x; // final softmax probabilities
}

function getResponse(tag) {
  const found = intents.intents.find((i) => i.tag === tag);
  const pool = found ? found.responses : intents.intents.find((i) => i.tag === "fallback").responses;
  return pool[Math.floor(Math.random() * pool.length)];
}

function predictIntent(sentence) {
  const bag = bagOfWords(sentence);
  const scores = forward(bag);

  let bestIdx = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[bestIdx]) bestIdx = i;
  }
  const confidence = scores[bestIdx];
  if (confidence < CONFIDENCE_THRESHOLD) {
    return { tag: "fallback", confidence };
  }
  return { tag: classes[bestIdx], confidence };
}

function handleUserMessage(message) {
  const trimmed = message.trim();
  if (!trimmed) return;
  addBubble("user", trimmed);
  textInput.value = "";

  const { tag, confidence } = predictIntent(trimmed);
  const reply = getResponse(tag);
  addBubble("bot", reply, `intent: ${tag}, confidence: ${confidence.toFixed(2)}`);
}

// --- Azure AI Speech (Speech SDK for JavaScript) ---
let cachedToken = null; // { token, region, fetchedAt }
const TOKEN_TTL_MS = 9 * 60 * 1000; // tokens are valid 10 min; refresh a bit early

async function getSpeechToken() {
  if (cachedToken && Date.now() - cachedToken.fetchedAt < TOKEN_TTL_MS) {
    return cachedToken;
  }
  const res = await fetch("/api/get-speech-token");
  if (!res.ok) throw new Error(`Token request failed: ${res.status}`);
  const { token, region } = await res.json();
  cachedToken = { token, region, fetchedAt: Date.now() };
  return cachedToken;
}

async function recognizeSpeechOnce() {
  const { token, region } = await getSpeechToken();
  const speechConfig = SpeechSDK.SpeechConfig.fromAuthorizationToken(token, region);
  speechConfig.speechRecognitionLanguage = "en-US";
  const audioConfig = SpeechSDK.AudioConfig.fromDefaultMicrophoneInput();
  const recognizer = new SpeechSDK.SpeechRecognizer(speechConfig, audioConfig);

  return new Promise((resolve, reject) => {
    recognizer.recognizeOnceAsync(
      (result) => {
        recognizer.close();
        if (result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
          resolve(result.text);
        } else {
          reject(new Error(`No speech recognized (reason: ${result.reason}).`));
        }
      },
      (err) => {
        recognizer.close();
        reject(err);
      }
    );
  });
}

speakBtn.addEventListener("click", async () => {
  speakBtn.disabled = true;
  speakBtn.classList.add("listening");
  setStatus("Listening... speak now.");
  try {
    const transcript = await recognizeSpeechOnce();
    setStatus(`Recognized: "${transcript}"`);
    handleUserMessage(transcript);
  } catch (err) {
    console.error(err);
    setStatus(`Speech recognition error: ${err.message || err}`);
  } finally {
    speakBtn.disabled = false;
    speakBtn.classList.remove("listening");
  }
});

sendBtn.addEventListener("click", () => handleUserMessage(textInput.value));
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") handleUserMessage(textInput.value);
});

// --- Boot: load weights + vocab + classes + intents ---
async function init() {
  try {
    const [weightsData, vocabData, classesData, intentsData] = await Promise.all([
      fetch("model/weights.json").then((r) => r.json()),
      fetch("model/words.json").then((r) => r.json()),
      fetch("model/classes.json").then((r) => r.json()),
      fetch("intents.json").then((r) => r.json()),
    ]);
    weights = weightsData;
    vocab = vocabData;
    classes = classesData;
    intents = intentsData;

    sendBtn.disabled = false;
    setStatus("Model loaded. Click 🎤 Speak or type a message.");
    addBubble("bot", "Hi! I'm ready. Click the mic and talk, or type a message below.");
  } catch (err) {
    console.error(err);
    setStatus("Failed to load the model. Please refresh the page.");
  }
}

sendBtn.disabled = true;
init();
