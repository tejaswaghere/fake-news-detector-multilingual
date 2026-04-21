import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# ── 1. Load model and tokenizer ──────────────────────────────────────────────
# We load from the local model/final/ folder — no internet needed
MODEL_PATH = '/content/drive/MyDrive/fake-news-detector/model/final/'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # set to evaluation mode — disables dropout layers

print("Model loaded successfully!")

# ── 2. Text cleaning ─────────────────────────────────────────────────────────
# Same cleaning function we used in preprocessing — must match exactly
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

# ── 3. Prediction function ───────────────────────────────────────────────────
# This is the core function Gradio calls every time the user clicks Submit
def predict(title, content):
    # Combine title and content exactly like we did in preprocessing
    combined = title.strip() + '. ' + content.strip()
    cleaned  = clean_text(combined)

    # Tokenize
    inputs = tokenizer(
        cleaned,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        padding='max_length'
    )

    # Run through model — no_grad means we don't compute gradients (faster, less memory)
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities using softmax
    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    fake_prob = probs[1].item()
    real_prob = probs[0].item()

    # Build result
    label = "🚨 FAKE NEWS" if fake_prob > real_prob else "✅ REAL NEWS"
    confidence = max(fake_prob, real_prob) * 100

    result = f"{label}\n\nConfidence: {confidence:.1f}%"
    details = f"Real: {real_prob*100:.1f}%  |  Fake: {fake_prob*100:.1f}%"

    return result, details

# ── 4. Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Multilingual Fake News Detector") as demo:
    gr.Markdown("# 🔍 Fake News Detector")
    gr.Markdown("Enter a news article below to check if it's real or fake.")

    with gr.Row():
        with gr.Column():
            title_input = gr.Textbox(
                label="Article Title",
                placeholder="Enter the headline here...",
                lines=2
            )
            content_input = gr.Textbox(
                label="Article Content",
                placeholder="Paste the article text here...",
                lines=10
            )
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column():
            result_output  = gr.Textbox(label="Verdict", lines=3)
            details_output = gr.Textbox(label="Probability Breakdown", lines=2)

    submit_btn.click(
        fn=predict,
        inputs=[title_input, content_input],
        outputs=[result_output, details_output]
    )

    gr.Markdown("### Try these examples:")
    gr.Examples(
        examples=[
            ["Scientists discover water on Mars", "NASA researchers have confirmed the presence of liquid water beneath the surface of Mars using radar data from the Mars Express spacecraft."],
            ["SHOCKING: Government putting chemicals in water to control minds", "Anonymous sources reveal that the deep state has been adding mind control substances to municipal water supplies across America since 1995."],
        ],
        inputs=[title_input, content_input]
    )

# ── 5. Launch ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=True)