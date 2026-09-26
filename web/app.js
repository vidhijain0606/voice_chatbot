/**
 * app.js
 * Voice chatbot dashboard:
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
 *  - A small rule-based "smart answers" layer intercepts messages it can
 *    answer with real, live/computed data -- weather, time, date, simple
 *    arithmetic -- before falling through to the NN classifier. Weather and
 *    math render as "telemetry cards" instead of plain text.
 *  - Session history (sidebar) is kept in localStorage so previous
 *    conversations survive a page reload, grouped by Today / Yesterday /
 *    Last 7 Days / Older.
 */

const CONFIDENCE_THRESHOLD = 0.5;
const HISTORY_KEY = "voicebot_history_v1";
const DAY_MS = 24 * 60 * 60 * 1000;

let weights = null; // { layers: [{ w, b, activation }, ...] }
let vocab = [];
let classes = [];
let intents = null;
let busy = false; // true while a reply (incl. any async fetch) is being prepared

// active session: not persisted until "New session" archives it
let active = { id: "active", title: "New session", createdAt: Date.now(), messages: [] };
let conversations = loadHistory(); // archived sessions, newest first

const sidebar = document.getElementById("sidebar");
const collapseBtn = document.getElementById("collapse-btn");
const newSessionBtn = document.getElementById("new-session-btn");
const historyEl = document.getElementById("history");
const feedInner = document.getElementById("feed-inner");
const systemBanner = document.getElementById("system-banner");
const textInput = document.getElementById("text-input");
const speakBtn = document.getElementById("speak-btn");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const exportBtn = document.getElementById("export-btn");
const statusEl = document.getElementById("status");
const chipsEl = document.getElementById("suggestion-chips");
const brandDot = document.getElementById("brand-dot");
const settingsBtn = document.getElementById("settings-btn");
const settingsPop = document.getElementById("settings-pop");
const micSelect = document.getElementById("mic-select");
const langSelect = document.getElementById("lang-select");

const ICON_SUN =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"></path></svg>';
const ICON_CALC =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="2" width="16" height="20" rx="2"></rect><path d="M8 6h8M8 11h.01M12 11h.01M16 11h.01M8 15h.01M12 15h.01M16 15h.01M8 19h.01M12 19h.01M16 19h.01"></path></svg>';
const ICON_CHAT =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>';
const ICON_CHEVRON_L =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m15 18-6-6 6-6"></path></svg>';
const ICON_CHEVRON_R =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"></path></svg>';

collapseBtn.innerHTML = ICON_CHEVRON_L;

function setStatus(text) {
  statusEl.textContent = text;
}

function fmtTime(d) {
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// ============================================================================
// Persistence (sidebar history)
// ============================================================================

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (err) {
    console.error("Failed to load history:", err);
    return [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(conversations));
  } catch (err) {
    console.error("Failed to save history:", err);
  }
}

function groupLabel(createdAt) {
  const diffDays = Math.floor((Date.now() - createdAt) / DAY_MS);
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays <= 7) return "Last 7 Days";
  return "Older";
}

function relTime(createdAt) {
  const diffH = Math.round((Date.now() - createdAt) / (60 * 60 * 1000));
  if (diffH < 1) return "just now";
  if (diffH < 24) return `${diffH}h ago`;
  return `${Math.round(diffH / 24)}d ago`;
}

function renderHistory() {
  historyEl.innerHTML = "";
  if (!conversations.length) {
    const empty = document.createElement("div");
    empty.className = "hist-empty";
    empty.textContent = "No previous sessions yet.";
    historyEl.appendChild(empty);
    return;
  }
  const groups = { Today: [], Yesterday: [], "Last 7 Days": [], Older: [] };
  conversations.forEach((c) => groups[groupLabel(c.createdAt)].push(c));
  ["Today", "Yesterday", "Last 7 Days", "Older"].forEach((g) => {
    if (!groups[g].length) return;
    const label = document.createElement("div");
    label.className = "group-label";
    label.textContent = g;
    historyEl.appendChild(label);
    groups[g].forEach((c) => {
      const item = document.createElement("div");
      item.className = "hist-item" + (c.id === active.id ? " active" : "");
      item.innerHTML =
        ICON_CHAT +
        `<span class="hist-text"><div class="hist-title">${escapeHtml(c.title)}</div><div class="hist-meta">${relTime(
          c.createdAt
        )}</div></span>`;
      item.addEventListener("click", () => loadConversation(c.id));
      historyEl.appendChild(item);
    });
  });
}

function loadConversation(id) {
  const conv = conversations.find((c) => c.id === id);
  if (!conv) return;
  active = { id: conv.id, title: conv.title, createdAt: conv.createdAt, messages: conv.messages.slice() };
  renderFeed();
}

// ============================================================================
// Chat rendering
// ============================================================================

function weatherIconSvg() {
  return ICON_SUN;
}

function renderMessageNode(m) {
  const row = document.createElement("div");
  row.className = `msg-row ${m.role}`;

  if (m.role === "user") {
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble user";
    bubble.textContent = m.text;
    row.appendChild(bubble);
  } else if (m.weather) {
    const card = document.createElement("div");
    card.className = "msg-bubble bot";
    card.innerHTML =
      `<div class="telemetry-card"><div class="tc-weather">` +
      `<div class="tc-icon">${weatherIconSvg()}</div>` +
      `<div class="tc-main"><div class="tc-temp">${m.weather.temp}&deg;C</div>` +
      `<div class="tc-cond">${escapeHtml(m.weather.cond)} &middot; ${escapeHtml(m.weather.place)}</div></div>` +
      `</div><div class="tc-stats">` +
      `<div><div class="tc-stat-label">wind</div><div class="tc-stat-value">${m.weather.wind} km/h</div></div>` +
      `<div><div class="tc-stat-label">location</div><div class="tc-stat-value">${escapeHtml(m.weather.place)}</div></div>` +
      `<div><div class="tc-stat-label">updated</div><div class="tc-stat-value">${m.time}</div></div>` +
      `</div></div>`;
    row.appendChild(card);
  } else if (m.math) {
    const card = document.createElement("div");
    card.className = "msg-bubble bot";
    card.innerHTML =
      `<div class="telemetry-card"><div class="tc-math">` +
      `<div class="tc-icon">${ICON_CALC}</div>` +
      `<div><div class="tc-math-expr">${escapeHtml(m.math.expr)} =</div>` +
      `<div class="tc-math-result">${escapeHtml(m.math.result)}</div></div>` +
      `</div></div>`;
    row.appendChild(card);
  } else {
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble bot";
    bubble.textContent = m.text;
    row.appendChild(bubble);
  }

  const t = document.createElement("span");
  t.className = "msg-time";
  t.textContent = m.time;
  row.appendChild(t);

  return row;
}

function renderFeed() {
  feedInner.innerHTML = "";
  feedInner.appendChild(systemBanner);
  active.messages.forEach((m) => feedInner.appendChild(renderMessageNode(m)));
  feedInner.appendChild(statusEl);
  scrollFeedToBottom();
  renderHistory();
}

function scrollFeedToBottom() {
  const scroller = feedInner.parentElement;
  scroller.scrollTop = scroller.scrollHeight;
}

function addThinkingRow() {
  const row = document.createElement("div");
  row.className = "msg-row bot";
  row.innerHTML = '<div class="thinking"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>';
  feedInner.insertBefore(row, statusEl);
  scrollFeedToBottom();
  return row;
}

function setBusy(isBusy) {
  busy = isBusy;
  sendBtn.disabled = isBusy;
  speakBtn.disabled = isBusy;
  textInput.disabled = isBusy;
}

// ============================================================================
// NN intent classifier (unchanged model, trained via Azure ML)
// ============================================================================

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
// Each returns a structured reply: { text } or { weather:{...} } or
// { math:{...} }, rendered as a telemetry card for weather/math.
// ============================================================================

function isTimeQuery(msg) {
  return /\b(what(?:'s| is) the time|current time|what time is it|tell me the time)\b/i.test(msg);
}
function isDateQuery(msg) {
  return /\b(what(?:'s| is)(?: today's)? date|what day is it|today's date|what is the date)\b/i.test(msg);
}

// Maps a place name mentioned in the message to a real IANA time zone, so
// "time in Australia" answers with Australia's clock, not the browser's.
// Sorted longest-key-first so "new york" matches before a shorter overlap.
const TIMEZONE_ALIASES = {
  "new zealand": { tz: "Pacific/Auckland", label: "Auckland, New Zealand" },
  "los angeles": { tz: "America/Los_Angeles", label: "Los Angeles" },
  california: { tz: "America/Los_Angeles", label: "California" },
  "new york": { tz: "America/New_York", label: "New York" },
  australia: { tz: "Australia/Sydney", label: "Sydney, Australia" },
  sydney: { tz: "Australia/Sydney", label: "Sydney" },
  melbourne: { tz: "Australia/Melbourne", label: "Melbourne" },
  perth: { tz: "Australia/Perth", label: "Perth" },
  india: { tz: "Asia/Kolkata", label: "India" },
  chennai: { tz: "Asia/Kolkata", label: "Chennai" },
  mumbai: { tz: "Asia/Kolkata", label: "Mumbai" },
  delhi: { tz: "Asia/Kolkata", label: "Delhi" },
  bangalore: { tz: "Asia/Kolkata", label: "Bangalore" },
  london: { tz: "Europe/London", label: "London" },
  uk: { tz: "Europe/London", label: "the UK" },
  england: { tz: "Europe/London", label: "England" },
  japan: { tz: "Asia/Tokyo", label: "Japan" },
  tokyo: { tz: "Asia/Tokyo", label: "Tokyo" },
  dubai: { tz: "Asia/Dubai", label: "Dubai" },
  uae: { tz: "Asia/Dubai", label: "the UAE" },
  singapore: { tz: "Asia/Singapore", label: "Singapore" },
  germany: { tz: "Europe/Berlin", label: "Germany" },
  france: { tz: "Europe/Paris", label: "France" },
  china: { tz: "Asia/Shanghai", label: "China" },
  russia: { tz: "Europe/Moscow", label: "Moscow, Russia" },
  brazil: { tz: "America/Sao_Paulo", label: "Brazil" },
  canada: { tz: "America/Toronto", label: "Toronto, Canada" },
  usa: { tz: "America/New_York", label: "the USA (Eastern)" },
};
const TIMEZONE_KEYS = Object.keys(TIMEZONE_ALIASES).sort((a, b) => b.length - a.length);

function extractTimezone(msg) {
  const lower = msg.toLowerCase();
  for (const key of TIMEZONE_KEYS) {
    if (new RegExp(`\\b${key}\\b`).test(lower)) {
      return TIMEZONE_ALIASES[key];
    }
  }
  return null;
}

function timeInZone(tz) {
  return new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  }).format(new Date());
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

// --- Extended math: square/cube roots, powers, percentages. These use plain
// English patterns (not the digits-only grammar above) and are checked
// first, so questions like "square root of 144" never fall through to the
// NN classifier -- which has no way to compute an answer and would just
// guess the nearest-sounding trained intent. ---
function trySpecialMath(msg) {
  const lower = msg.toLowerCase();
  const num = "(-?\\d+(?:\\.\\d+)?)";
  let m;

  if ((m = lower.match(new RegExp(`square root of\\s*${num}`)))) {
    return { expr: `√${m[1]}`, result: Math.sqrt(parseFloat(m[1])) };
  }
  if ((m = lower.match(new RegExp(`cube root of\\s*${num}`)))) {
    return { expr: `∛${m[1]}`, result: Math.cbrt(parseFloat(m[1])) };
  }
  if ((m = lower.match(new RegExp(`${num}\\s*(?:to the power of|raised to(?: the power of)?)\\s*${num}`)))) {
    return { expr: `${m[1]}^${m[2]}`, result: Math.pow(parseFloat(m[1]), parseFloat(m[2])) };
  }
  if ((m = lower.match(new RegExp(`${num}\\s*squared`)))) {
    const n = parseFloat(m[1]);
    return { expr: `${m[1]}²`, result: n * n };
  }
  if ((m = lower.match(new RegExp(`${num}\\s*cubed`)))) {
    const n = parseFloat(m[1]);
    return { expr: `${m[1]}³`, result: n * n * n };
  }
  if ((m = lower.match(new RegExp(`${num}\\s*%\\s*of\\s*${num}`))) || (m = lower.match(new RegExp(`${num}\\s*percent of\\s*${num}`)))) {
    return { expr: `${m[1]}% of ${m[2]}`, result: (parseFloat(m[1]) / 100) * parseFloat(m[2]) };
  }
  return null;
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
    return { weather: { temp: w.temp, cond: desc, wind: w.wind, place: placeName } };
  } catch (err) {
    console.error("Weather lookup failed:", err);
    if (!city) {
      return {
        text: `I couldn't get your location for the weather. Try "weather in <city>" instead, or check your browser's location permission.`,
      };
    }
    // Fall back to the trained NN's canned weather_smalltalk response.
    return { text: getResponse("weather_smalltalk") };
  }
}

// Ordered intercepts: first match wins. handle() may be sync or async and
// returns a structured reply object.
const smartAnswers = [
  {
    test: isTimeQuery,
    handle: (msg) => {
      const zone = extractTimezone(msg);
      if (zone) {
        return { text: `It's currently ${timeInZone(zone.tz)} in ${zone.label}.` };
      }
      return { text: `It's currently ${new Date().toLocaleTimeString()}.` };
    },
  },
  {
    test: isDateQuery,
    handle: () => ({
      text: `Today is ${new Date().toLocaleDateString(undefined, {
        weekday: "long",
        year: "numeric",
        month: "long",
        day: "numeric",
      })}.`,
    }),
  },
  { test: isWeatherQuery, handle: handleWeatherQuery },
  {
    test: (msg) => trySpecialMath(msg) !== null,
    handle: (msg) => {
      const r = trySpecialMath(msg);
      if (!Number.isFinite(r.result)) {
        return { text: `I couldn't work that out — try something like "square root of 144" or "12 to the power of 2".` };
      }
      const rounded = Math.round(r.result * 1e6) / 1e6;
      return { math: { expr: r.expr, result: String(rounded) } };
    },
  },
  {
    test: (msg) => tryExtractMathExpression(msg) !== null,
    handle: (msg) => {
      const expr = tryExtractMathExpression(msg);
      try {
        const result = evalMathExpression(expr);
        if (!Number.isFinite(result)) throw new Error("Non-finite result");
        const rounded = Math.round(result * 1e6) / 1e6;
        return { math: { expr: expr.trim(), result: String(rounded) } };
      } catch (err) {
        return { text: `I couldn't work that out — try a simple expression like "12 * 7".` };
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

// ============================================================================
// Message handling
// ============================================================================

async function handleUserMessage(message) {
  if (busy) return;
  const trimmed = message.trim();
  if (!trimmed) return;

  const userTime = fmtTime(new Date());
  active.messages.push({ role: "user", text: trimmed, time: userTime });
  renderFeed();
  textInput.value = "";
  setBusy(true);

  const thinking = addThinkingRow();
  try {
    const smart = await findSmartAnswer(trimmed);
    let reply;
    if (smart !== null) {
      reply = smart;
    } else {
      const { tag, confidence } = predictIntent(trimmed);
      reply = { text: getResponse(tag) };
    }
    thinking.remove();
    reply.role = "bot";
    reply.time = fmtTime(new Date());
    active.messages.push(reply);
    renderFeed();
    if (active.id !== "active") persistActiveEdits();
  } catch (err) {
    console.error(err);
    thinking.remove();
    active.messages.push({ role: "bot", text: "Sorry, something went wrong answering that.", time: fmtTime(new Date()) });
    renderFeed();
  } finally {
    setBusy(false);
    textInput.focus();
  }
}

function persistActiveEdits() {
  const idx = conversations.findIndex((c) => c.id === active.id);
  if (idx !== -1) {
    conversations[idx].messages = active.messages.slice();
    saveHistory();
  }
}

// ============================================================================
// Sidebar actions
// ============================================================================

let sidebarCollapsed = false;
collapseBtn.addEventListener("click", () => {
  sidebarCollapsed = !sidebarCollapsed;
  sidebar.classList.toggle("collapsed", sidebarCollapsed);
  collapseBtn.innerHTML = sidebarCollapsed ? ICON_CHEVRON_R : ICON_CHEVRON_L;
});

newSessionBtn.addEventListener("click", () => {
  if (active.messages.length > 0) {
    const firstUser = active.messages.find((m) => m.role === "user");
    const title = firstUser ? firstUser.text.slice(0, 40) : "Conversation";
    if (active.id === "active") {
      conversations.unshift({ id: `c-${Date.now()}`, title, createdAt: active.createdAt, messages: active.messages.slice() });
      saveHistory();
    }
  }
  active = { id: "active", title: "New session", createdAt: Date.now(), messages: [] };
  renderFeed();
  setStatus("new session started.");
  textInput.focus();
});

clearBtn.addEventListener("click", () => {
  active.messages = [];
  renderFeed();
  setStatus("session cleared.");
});

exportBtn.addEventListener("click", () => {
  if (!active.messages.length) {
    setStatus("nothing to export yet.");
    return;
  }
  const lines = active.messages.map((m) => {
    const who = m.role === "user" ? "You" : "Voicebot";
    const content =
      m.text ||
      (m.weather ? `${m.weather.temp}°C, ${m.weather.cond} in ${m.weather.place} (wind ${m.weather.wind} km/h)` : "") ||
      (m.math ? `${m.math.expr} = ${m.math.result}` : "");
    return `[${m.time}] ${who}: ${content}`;
  });
  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "voicebot-transcript.txt";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  setStatus("transcript exported.");
});

// ============================================================================
// Settings: real mic + language selection, wired into the Speech SDK
// ============================================================================

let selectedMicId = "";
let recognitionLanguage = "en-US";

async function populateMicList() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const mics = devices.filter((d) => d.kind === "audioinput");
    micSelect.innerHTML = '<option value="">Default microphone</option>';
    mics.forEach((d, i) => {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Microphone ${i + 1}`;
      micSelect.appendChild(opt);
    });
  } catch (err) {
    console.error("Could not list microphones:", err);
  }
}

micSelect.addEventListener("change", () => {
  selectedMicId = micSelect.value;
});
langSelect.addEventListener("change", () => {
  recognitionLanguage = langSelect.value;
});

settingsBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  const opening = !settingsPop.classList.contains("open");
  settingsPop.classList.toggle("open");
  if (opening) populateMicList();
});
document.addEventListener("click", (e) => {
  if (!settingsPop.contains(e.target) && e.target !== settingsBtn) {
    settingsPop.classList.remove("open");
  }
});

// ============================================================================
// Azure AI Speech (Speech SDK for JavaScript)
// ============================================================================

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
  speechConfig.speechRecognitionLanguage = recognitionLanguage;
  const audioConfig = selectedMicId
    ? SpeechSDK.AudioConfig.fromMicrophoneInput(selectedMicId)
    : SpeechSDK.AudioConfig.fromDefaultMicrophoneInput();
  const recognizer = new SpeechSDK.SpeechRecognizer(speechConfig, audioConfig);
  activeRecognizer = recognizer;

  // Live interim captions while the user is still speaking.
  recognizer.recognizing = (_s, e) => {
    if (e.result && e.result.text) {
      setStatus(`listening: "${e.result.text}"`);
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
  setStatus("cancelled.");
}

speakBtn.addEventListener("click", async () => {
  if (speakBtn.classList.contains("listening")) {
    stopListening();
    return;
  }
  speakBtn.classList.add("listening");
  setStatus("listening — speak now…");
  try {
    const transcript = await recognizeSpeechOnce();
    setStatus(`recognized: "${transcript}"`);
    await handleUserMessage(transcript);
  } catch (err) {
    console.error(err);
    setStatus(`speech error: ${err.message || err}`);
  } finally {
    speakBtn.classList.remove("listening");
  }
});

sendBtn.addEventListener("click", () => handleUserMessage(textInput.value));
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") handleUserMessage(textInput.value);
});

chipsEl.addEventListener("click", (e) => {
  const chip = e.target.closest("[data-message]");
  if (!chip) return;
  handleUserMessage(chip.dataset.message);
});

// ============================================================================
// Boot: load weights + vocab + classes + intents
// ============================================================================

async function init() {
  renderFeed();
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
    setStatus("model loaded — press to talk, or type a message.");
    brandDot.classList.add("live");
    textInput.focus();
  } catch (err) {
    console.error(err);
    setStatus("failed to load the model — please refresh the page.");
  }
}

sendBtn.disabled = true;
init();
