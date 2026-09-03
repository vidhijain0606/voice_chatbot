# Azure Setup Guide — Voice-Enabled Chatbot

Everything below is done in **portal.azure.com** (your browser, where MFA
already works) — Azure CLI sign-in is blocked by your tenant's Conditional
Access policy, so no CLI is used. Do these in order. Nothing here asks for
a paid tier; every resource is created on its free tier.

Repo: https://github.com/vidhijain0606/voice_chatbot

---

## Phase 1 — Resource Group
*One container to hold everything, easy to delete later if needed.*

1. Go to https://portal.azure.com → search **"Resource groups"** (top search bar) → **+ Create**
2. Subscription: your subscription. Resource group name: **`voice-chatbot-rg`**
3. Region: **Central India**
4. **Review + Create** → **Create**

---

## Phase 2 — Speech resource (Speech-to-Text)
*Powers the 🎤 voice input.*

1. Open `voice-chatbot-rg` → **+ Create**
2. Search **"Speech"** → select **Speech** (under Azure AI services) → **Create**
3. Resource group: `voice-chatbot-rg`. Region: **Central India**
4. Name: **`voice-chatbot-speech`**
5. Pricing tier: **Free F0** (5 audio hours/month, enough for a lab demo)
6. **Review + Create** → **Create** → wait for deployment
7. Go to the resource → left nav **"Keys and Endpoint"** → leave this tab open (you'll need **Key 1** and the **Location/Region** value in Phase 4 — don't paste the key into chat, only into the Azure Portal field in Phase 4)

---

## Phase 3 — Azure Machine Learning (training)
*Runs `train.py` on Azure compute instead of your laptop.*

1. In `voice-chatbot-rg` → **+ Create** → search **"Azure Machine Learning"** → **Create**
2. Name: **`voice-chatbot-ml`**, Region: **Central India** → **Review + Create** → **Create**
   (this auto-creates a storage account, key vault, and app insights resource in the group — normal, no cost while idle)
3. Once deployed → **Go to resource** → **Launch studio** (opens `ml.azure.com`)
4. Left nav → **Compute** → **Compute instances** tab → **+ New**
   - Smallest CPU size (e.g. `Standard_DS11_v2`) → **Create** → wait until status = **Running**
5. Click **Terminal** on that compute instance (opens a browser terminal)
6. Run:
   ```bash
   git clone https://github.com/vidhijain0606/voice_chatbot.git
   cd voice_chatbot
   pip install -r requirements.txt
   python train.py
   python export_web_model.py
   ```
7. Push the retrained model back to GitHub from the same terminal (it'll prompt for your GitHub login/PAT — that's between you and GitHub, not me):
   ```bash
   git add model/ web/model/ web/intents.json
   git commit -m "Retrain via Azure ML compute instance"
   git push
   ```
8. **Stop the compute instance**: Compute instances list → select `voice-chatbot-ml`'s instance → **Stop** (avoids ongoing charges — it only bills while running).

---

## Phase 4 — Static Web App (hosting)
*Publishes the site and its API at a public URL.*

1. In `voice-chatbot-rg` → **+ Create** → search **"Static Web App"** → **Create**
2. Name: **`voice-chatbot-swa`**
3. Plan type: **Free**
4. Region for Azure Functions API: closest available option to Central India
5. Deployment source: **GitHub**
   - Sign in / authorize Azure to access your GitHub if prompted
   - Organization: `vidhijain0606`, Repository: `voice_chatbot`, Branch: `main`
6. Build Details:
   - Build Presets: **Custom**
   - App location: `/web`
   - Api location: `/api`
   - Output location: *(leave empty)*
7. **Review + Create** → **Create**
   - This automatically commits a `.github/workflows/azure-static-web-apps-*.yml` file to your repo and triggers a deployment via GitHub Actions.
8. Once deployment finishes → open the resource → left nav **"Configuration"** → **Application settings** → **+ Add**:
   - Name: `SPEECH_KEY`, Value: *(paste Key 1 from Phase 2, step 7)*
   - Name: `SPEECH_REGION`, Value: `centralindia`
   - Click **Save** (top of page)
9. Go to **Overview** tab → copy the **URL** (format `https://<some-name>.azurestaticapps.net`)

---

## Phase 5 — Verify it's live
1. Open the Static Web App URL from Phase 4 in Chrome or Edge.
2. Allow microphone access when prompted.
3. Click **🎤 Speak**, say something like "hello" — you should see the recognized text and a matching chatbot reply.
4. Try the text box too as a fallback check.
5. Send me the URL — I'll do the same check from my end and record it as the final submission link in `report.md`.

---

## Cost check
Everything above uses free tiers **except** the Azure ML compute instance,
which bills per minute *while running* (a few cents for the few minutes
Phase 3 takes) — that's why step 3.8 says to stop it. Speech (F0) and
Static Web Apps (Free) cost nothing at this scale.
