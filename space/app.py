import os
import sys

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from model import build_model
from data.dataset import _pad_or_crop, SEGMENT_SAMPLES

CONFIG_NAME = os.environ.get("MODEL_CONFIG", "sinc_no_sa_no_da")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/best.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL, _ = build_model(CONFIG_NAME)

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    MODEL.load_state_dict(ckpt["model_state"], strict=False)
    CHECKPOINT_STATUS = f"checkpoint loaded: {CHECKPOINT_PATH}"
else:
    CHECKPOINT_STATUS = ("no checkpoint found at " + CHECKPOINT_PATH +
                          " — running with random, UNTRAINED weights. "
                          "Upload best.pt via the Space's Files tab.")

MODEL.to(DEVICE)
MODEL.eval()


def predict(audio_path, threshold):
    if audio_path is None:
        return "No audio provided", None, ""

    data, sr = sf.read(audio_path, dtype="float32")
    wav = torch.from_numpy(data)
    wav = wav.unsqueeze(0) if wav.ndim == 1 else wav.T
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = _pad_or_crop(wav, target_len=SEGMENT_SAMPLES, train=False).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(wav)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        raw_score = float(logits[0, 1] - logits[0, 0])

    bonafide_pct = float(probs[1] * 100)
    score_breakdown = {"bona fide": float(probs[1]), "spoofed": float(probs[0])}

    if bonafide_pct >= threshold:
        verdict = f"GENUINE — {bonafide_pct:.1f}% bona fide (threshold {threshold:.0f}%)"
    elif bonafide_pct >= threshold - 15:
        verdict = f"UNCERTAIN — {bonafide_pct:.1f}% bona fide, flag for review (threshold {threshold:.0f}%)"
    else:
        verdict = f"FLAGGED — {bonafide_pct:.1f}% bona fide, likely cloned (threshold {threshold:.0f}%)"

    detail = f"raw CM score: {raw_score:.3f}  ·  model: {CONFIG_NAME}  ·  {CHECKPOINT_STATUS}"
    return verdict, score_breakdown, detail


CUSTOM_CSS = """
.gradio-container { background: #0B0D10 !important; font-family: 'Space Grotesk', sans-serif !important; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.block { background: #14171B !important; border-color: #262B31 !important; }
label, .label-wrap span { color: #9AA2AB !important; font-size: 12px !important; }
textarea, input[type=text] { font-family: 'JetBrains Mono', monospace !important; color: #E6E8EA !important; }
button.primary { background: #33D6B0 !important; color: #0B0D10 !important; border: none !important; font-weight: 600 !important; }
footer { display: none !important; }
"""

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Space Grotesk"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)

with gr.Blocks(title="Voiceprint — impersonation risk console", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "# Voiceprint\n"
        "Voice-clone / impersonation risk console — reproduction of "
        "*\"Automatic speaker verification spoofing and deepfake detection "
        "using wav2vec 2.0 and data augmentation\"* (Tak et al., Odyssey 2022). "
        "Source: [GitHub repo](https://github.com/YOUR_USERNAME/YOUR_REPO)."
    )
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(type="filepath", label="Upload or record a clip")
            threshold = gr.Slider(0, 100, value=50, step=1,
                                   label="Decision threshold (bona fide %)")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            verdict_out = gr.Textbox(label="Verdict", interactive=False)
            probs_out = gr.Label(label="Score breakdown")
            detail_out = gr.Textbox(label="Detail", interactive=False)

    btn.click(predict, inputs=[audio_in, threshold],
              outputs=[verdict_out, probs_out, detail_out])
    threshold.change(predict, inputs=[audio_in, threshold],
                      outputs=[verdict_out, probs_out, detail_out])

if __name__ == "__main__":
    demo.launch()