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
 *  - A small rule-based "smart answers" layer (below) intercepts messages
 *    it can answer with real, live/computed data -- weather, time, date,
 *    simple arithmetic -- before falling through to the NN classifier.
 *    This is deliberately NOT part of the trained model: the NN only ever
 *    picks from a fixed list of canned responses per intent, which can't
 *    express dynamic answers like "it's 4:32 PM" or today's actual weather.
 */

const CONFIDENCE_THRESHOLD = 0.5;

let weights = null; // { layers: [{ w, b, activation }, ...] }
let vocab = [];
let classes = [];
let intents = null;
let busy = false; // true while a reply (incl. any async fetch) is being prepared

const chatWindow = document.getElementById("chat-window");
const textInput = document.getElementById("text-input");
const speakBtn = document.getElementById("speak-btn");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const statusEl = document.getElementById("status");
const chipsEl = document.getElementById("suggestion-chips");

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
  return bubble;
}

function addThinkingBubble() {
  const bubble = document.createElement("div");
  bubble.className = "bubble bot thinking";
  bubble.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  chatWindow.appendChild(bubble);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return bubble;
}

function setBusy(isBusy) {
  busy = isBusy;
  sendBtn.disabled = isBusy;
  speakBtn.disabled = isBusy;
  textInput.disabled = isBusy;
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

// ============================================================================
// Smart-answers intercept layer: weather, time, date, simple math.
// Each handler is tried in order; the first whose test() matches wins and
// its (possibly async) handle() supplies the reply text directly, bypassing
// the NN classifier. Anything that matches nothing falls through to the NN.
// ============================================================================

// --- Time / date (fully local, no network) ---
function isTimeQuery(msg) {
  return /\b(what(?:'s| is) the time|current time|what time is it|tell me the time)\b/i.test(msg);
}
function isDateQuery(msg) {
  return /\b(what(?:'s| is)(?: today's)? date|what day is it|today's date|what is the date)\b/i.test(msg);
}

// --- Simple arithmetic: safe hand-rolled recursive-descent parser (no eval) ---
function tryExtractMathExpression(msg) {
  const stripped = msg
    .replace(/^(?:what is|what's|calculate|compute|solve)\s*/i, "")
    .replace(/[?=]+\s*$/, "")
    .trim();
  if (!stripped || !/^[\d.\s+\-*/()]+$/.test(stripped)) return null;
  if (!/\d/.test(stripped)) return null; // require at least one digit
  return stripped;
}

function evalMathExpression(expr) {
  let i = 0;
  function peek() {
    while (i < expr.length && /\s/.test(expr[i])) i++;
    return expr[i];
  }
  function consume() {
    const c = peek();
    i++;
    return c;
  }
  function parseNumber() {
    while (i < expr.length && /\s/.test(expr[i])) i++;
    const start = i;
    while (i < expr.length && /[\d.]/.test(expr[i])) i++;
    if (i === start) throw new Error("Expected number");
    return parseFloat(expr.slice(start, i));
  }
  function parseFactor() {
    if (peek() === "(") {
      consume();
      const v = parseExpr();
      if (peek() !== ")") throw new Error("Expected )");
      consume();
      return v;
    }
    if (peek() === "-") {
      consume();
      return -parseFactor();
    }
    if (peek() === "+") {
      consume();
      return parseFactor();
    }
    return parseNumber();
  }
  function parseTerm() {
    let v = parseFactor();
    while (peek() === "*" || peek() === "/") {
      const op = consume();
      const rhs = parseFactor();
      v = op === "*" ? v * rhs : v / rhs;
    }
    return v;
  }
  function parseExpr() {
    let v = parseTerm();
    while (peek() === "+" || peek() === "-") {
      const op = consume();
      const rhs = parseTerm();
      v = op === "+" ? v + rhs : v - rhs;
    }
    return v;
  }
  const result = parseExpr();
  while (i < expr.length && /\s/.test(expr[i])) i++;
  if (i < expr.length) throw new Error("Unexpected trailing input");
  return result;
}

// --- Weather (Open-Meteo: free, keyless, CORS-enabled) ---
const WEATHER_CODES = {
  0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
  45: "fog", 48: "depositing rime fog",
  51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
  61: "light rain", 63: "moderate rain", 65: "heavy rain",
  71: "light snow", 73: "moderate snow", 75: "heavy snow",
  80: "light rain showers", 81: "moderate rain showers", 82: "violent rain showers",
  95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
};

function isWeatherQuery(msg) {
  return /\b(weather|temperature|forecast|raining|is it sunny|hot outside|cold outside)\b/i.test(msg);
}

function extractCity(msg) {
  const m = msg.match(/\b(?:weather|temperature|forecast)\b.*?\bin\s+([a-zA-Z\s]+?)\s*[?.!]?$/i);
  return m ? m[1].trim() : null;
}

function getCurrentPosition() {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error("Geolocation not supported"));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
      (err) => reject(err),
      { timeout: 8000 }
    );
  });
}

async function geocodeCity(city) {
  const url = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Geocoding failed: ${res.status}`);
  const data = await res.json();
  if (!data.results || !data.results.length) throw new Error(`Couldn't find "${city}"`);
  const { latitude, longitude, name } = data.results[0];
  return { lat: latitude, lon: longitude, name };
}

async function fetchWeather(lat, lon) {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,weathercode,wind_speed_10m`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Forecast fetch failed: ${res.status}`);
  const data = await res.json();
  const current = data.current;
  if (!current) throw new Error("No current-weather data returned");
  return {
    temp: current.temperature_2m,
    code: current.weathercode,
    wind: current.wind_speed_10m,
  };
}

async function handleWeatherQuery(msg) {
  const city = extractCity(msg);
  try {
    let lat, lon, placeName;
    if (city) {
      const geo = await geocodeCity(city);
      lat = geo.lat;
      lon = geo.lon;
      placeName = geo.name;
    } else {
      const pos = await getCurrentPosition();
      lat = pos.lat;
      lon = pos.lon;
      placeName = "your location";
    }
    const w = await fetchWeather(lat, lon);
    const desc = WEATHER_CODES[w.code] || "unknown conditions";
    return `It's currently ${w.temp}°C and ${desc} in ${placeName} (wind ${w.wind} km/h).`;
  } catch (err) {
    console.error("Weather lookup failed:", err);
    if (!city) {
      return `I couldn't get your location for the weather. Try "weather in <city>" instead, or check your browser's location permission.`;
    }
    // Fall back to the trained NN's canned weather_smalltalk response.
    return getResponse("weather_smalltalk");
  }
}

// Ordered intercepts: first match wins. handle() may be sync or async.
const smartAnswers = [
  { test: isTimeQuery, handle: () => `It's currently ${new Date().toLocaleTimeString()}.` },
  {
    test: isDateQuery,
    handle: () =>
      `Today is ${new Date().toLocaleDateString(undefined, {
        weekday: "long",
        year: "numeric",
        month: "long",
        day: "numeric",
      })}.`,
  },
  { test: isWeatherQuery, handle: handleWeatherQuery },
  {
    test: (msg) => tryExtractMathExpression(msg) !== null,
    handle: (msg) => {
      const expr = tryExtractMathExpression(msg);
      try {
        const result = evalMathExpression(expr);
        if (!Number.isFinite(result)) throw new Error("Non-finite result");
        const rounded = Math.round(result * 1e6) / 1e6;
        return `${expr.trim()} = ${rounded}`;
      } catch (err) {
        return `I couldn't work that out — try a simple expression like "12 * 7".`;
      }
    },
  },
];

async function findSmartAnswer(message) {
  for (const { test, handle } of smartAnswers) {
    if (test(message)) {
      return handle(message);
    }
  }
  return null;
}

async function handleUserMessage(message) {
  if (busy) return;
  const trimmed = message.trim();
  if (!trimmed) return;
  addBubble("user", trimmed);
  textInput.value = "";
  setBusy(true);

  const thinking = addThinkingBubble();
  try {
    const smartReply = await findSmartAnswer(trimmed);
    thinking.remove();
    if (smartReply !== null) {
      addBubble("bot", smartReply);
    } else {
      const { tag, confidence } = predictIntent(trimmed);
      const reply = getResponse(tag);
      addBubble("bot", reply, `intent: ${tag}, confidence: ${confidence.toFixed(2)}`);
    }
  } catch (err) {
    console.error(err);
    thinking.remove();
    addBubble("bot", "Sorry, something went wrong answering that.");
  } finally {
    setBusy(false);
    textInput.focus();
  }
}

// --- Azure AI Speech (Speech SDK for JavaScript) ---
let cachedToken = null; // { token, region, fetchedAt }
const TOKEN_TTL_MS = 9 * 60 * 1000; // tokens are valid 10 min; refresh a bit early
let activeRecognizer = null;

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
  activeRecognizer = recognizer;

  // Live interim captions while the user is still speaking.
  recognizer.recognizing = (_s, e) => {
    if (e.result && e.result.text) {
      setStatus(`Listening: "${e.result.text}"`);
    }
  };

  return new Promise((resolve, reject) => {
    recognizer.recognizeOnceAsync(
      (result) => {
        recognizer.close();
        activeRecognizer = null;
        if (result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
          resolve(result.text);
        } else if (result.reason === SpeechSDK.ResultReason.NoMatch) {
          reject(new Error("No speech recognized."));
        } else {
          reject(new Error("Speech recognition was cancelled."));
        }
      },
      (err) => {
        recognizer.close();
        activeRecognizer = null;
        reject(err);
      }
    );
  });
}

function stopListening() {
  if (activeRecognizer) {
    try {
      activeRecognizer.close();
    } catch (err) {
      console.error(err);
    }
    activeRecognizer = null;
  }
  speakBtn.classList.remove("listening");
  speakBtn.textContent = "🎤 Speak";
  setStatus("Cancelled.");
}

speakBtn.addEventListener("click", async () => {
  if (speakBtn.classList.contains("listening")) {
    stopListening();
    return;
  }
  speakBtn.classList.add("listening");
  speakBtn.textContent = "⏹ Stop";
  setStatus("Listening... speak now.");
  try {
    const transcript = await recognizeSpeechOnce();
    speakBtn.classList.remove("listening");
    speakBtn.textContent = "🎤 Speak";
    setStatus(`Recognized: "${transcript}"`);
    await handleUserMessage(transcript);
  } catch (err) {
    console.error(err);
    setStatus(`Speech recognition error: ${err.message || err}`);
  } finally {
    speakBtn.classList.remove("listening");
    speakBtn.textContent = "🎤 Speak";
  }
});

sendBtn.addEventListener("click", () => handleUserMessage(textInput.value));
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") handleUserMessage(textInput.value);
});

if (clearBtn) {
  clearBtn.addEventListener("click", () => {
    chatWindow.innerHTML = "";
    setStatus("Chat cleared.");
    addBubble("bot", "Hi! I'm ready. Click the mic and talk, or type a message below.");
  });
}

if (chipsEl) {
  chipsEl.addEventListener("click", (e) => {
    const chip = e.target.closest("[data-message]");
    if (!chip) return;
    handleUserMessage(chip.dataset.message);
  });
}

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
    textInput.focus();
  } catch (err) {
    console.error(err);
    setStatus("Failed to load the model. Please refresh the page.");
  }
}

sendBtn.disabled = true;
init();
