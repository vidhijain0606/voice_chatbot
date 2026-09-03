"""
app.py
Gradio web app: captures voice via the browser's Web Speech API, shows the
recognized text, runs it through the trained intent-classification NN, and
displays the chatbot's response. Falls back to manual text entry for
browsers without Web Speech support (e.g. Firefox).
"""
import gradio as gr

from chatbot_model import load_artifacts, predict_intent, get_response

MODEL, WORDS, CLASSES, INTENTS = load_artifacts()


def respond(message, history):
    history = history or []
    if not message or not message.strip():
        return history, ""
    tag, confidence = predict_intent(message, MODEL, WORDS, CLASSES)
    reply = get_response(tag, INTENTS)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": f"{reply}  \n_(intent: {tag}, confidence: {confidence:.2f})_"},
    ]
    return history, ""


# JS for browser-side speech recognition (Web Speech API). Writes the
# recognized transcript into the hidden textbox, then clicks the submit
# button so it flows through the same respond() pipeline as typed text.
SPEECH_JS = """
() => {
  const textbox = document.querySelector("#voice_textbox textarea");
  const statusEl = document.querySelector("#voice_status");
  if (!("webkitSpeechRecognition" in window) && !("SpeechRecognition" in window)) {
    if (statusEl) statusEl.innerText = "Speech recognition not supported in this browser. Please type your message instead.";
    return;
  }
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  if (statusEl) statusEl.innerText = "Listening... speak now.";

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    if (textbox) {
      textbox.value = transcript;
      textbox.dispatchEvent(new Event("input", { bubbles: true }));
    }
    if (statusEl) statusEl.innerText = "Recognized: \\"" + transcript + "\\"";
    setTimeout(() => {
      const submitBtn = document.querySelector("#submit_btn");
      if (submitBtn) submitBtn.click();
    }, 300);
  };

  recognition.onerror = (event) => {
    if (statusEl) statusEl.innerText = "Speech recognition error: " + event.error;
  };

  recognition.onspeechend = () => {
    recognition.stop();
  };

  recognition.start();
}
"""

with gr.Blocks(title="Voice-Enabled Chatbot") as demo:
    gr.Markdown(
        """
        # 🎤 Voice-Enabled Chatbot
        Speech Recognition (browser Web Speech API) + Deep Learning intent
        classification (Keras feedforward neural network).

        Click **🎤 Speak** and talk, or type a message below. Works best in
        **Chrome / Edge** (Web Speech API is not supported in Firefox — use
        the text box there instead).
        """
    )

    chatbot_ui = gr.Chatbot(label="Conversation", type="messages")

    with gr.Row():
        voice_box = gr.Textbox(
            elem_id="voice_textbox",
            label="Recognized speech / your message",
            placeholder="Click Speak, or type here...",
            scale=4,
        )
        speak_btn = gr.Button("🎤 Speak", scale=1)

    status_html = gr.HTML('<div id="voice_status" style="color: gray;">Idle.</div>')
    submit_btn = gr.Button("Send", elem_id="submit_btn", variant="primary")
    clear_btn = gr.Button("Clear conversation")

    speak_btn.click(fn=None, inputs=None, outputs=None, js=SPEECH_JS)
    submit_btn.click(fn=respond, inputs=[voice_box, chatbot_ui], outputs=[chatbot_ui, voice_box])
    voice_box.submit(fn=respond, inputs=[voice_box, chatbot_ui], outputs=[chatbot_ui, voice_box])
    clear_btn.click(fn=lambda: ([], ""), inputs=None, outputs=[chatbot_ui, voice_box])

if __name__ == "__main__":
    demo.launch()
