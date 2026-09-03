/**
 * get-speech-token
 *
 * Mints a short-lived (10-minute) Azure AI Speech authorization token using
 * the Speech resource key + region kept as server-side Application Settings
 * (SPEECH_KEY, SPEECH_REGION on the Static Web App) -- the key itself is
 * never sent to the browser. The client (web/app.js) fetches a token from
 * this endpoint before starting speech recognition with the Speech SDK.
 */
const https = require("https");

module.exports = async function (context, req) {
  const key = process.env.SPEECH_KEY;
  const region = process.env.SPEECH_REGION;

  if (!key || !region) {
    context.res = {
      status: 500,
      body: "Speech service is not configured (missing SPEECH_KEY/SPEECH_REGION app settings).",
    };
    return;
  }

  try {
    const token = await issueToken(key, region);
    context.res = {
      status: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token, region }),
    };
  } catch (err) {
    context.log.error("Failed to issue Speech token:", err);
    context.res = { status: 502, body: "Failed to obtain Speech token." };
  }
};

function issueToken(key, region) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: `${region}.api.cognitive.microsoft.com`,
      path: "/sts/v1.0/issueToken",
      method: "POST",
      headers: {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Length": 0,
      },
    };
    const request = https.request(options, (response) => {
      let data = "";
      response.on("data", (chunk) => (data += chunk));
      response.on("end", () => {
        if (response.statusCode === 200) {
          resolve(data);
        } else {
          reject(new Error(`Token endpoint returned ${response.statusCode}: ${data}`));
        }
      });
    });
    request.on("error", reject);
    request.end();
  });
}
