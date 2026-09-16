"""
Gradio app for Hugging Face Spaces (ZeroGPU hardware). `import spaces` must
be the very first import, before torch or anything CUDA-related — ZeroGPU
manages CUDA setup itself and breaks if something else touches it first.
"""
import spaces  # must be first, before torch/torchaudio

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

# Load on CPU at startup — ZeroGPU only attaches a real GPU inside functions
# decorated with @spaces.GPU, not at module import time.
MODEL, _ = build_model(CONFIG_NAME)

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    MODEL.load_state_dict(ckpt["model_state"], strict=False)
    CHECKPOINT_STATUS = f"checkpoint loaded: {CHECKPOINT_PATH}"
else:
    CHECKPOINT_STATUS = ("no checkpoint found at " + CHECKPOINT_PATH +
                          " — running with random, UNTRAINED weights. "
                          "Upload best.pt via the Space's Files tab.")

MODEL.eval()


@spaces.GPU
def predict(audio_path, threshold):
    if audio_path is None:
        return "No audio provided", None, ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL.to(device)

    data, sr = sf.read(audio_path, dtype="float32")
    wav = torch.from_numpy(data)
    wav = wav.unsqueeze(0) if wav.ndim == 1 else wav.T
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = _pad_or_crop(wav, target_len=SEGMENT_SAMPLES, train=False).to(device)

    with torch.no_grad():
        logits = model(wav)
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


with gr.Blocks(title="Voiceprint — impersonation risk console") as demo:
    gr.Markdown(
        "# Voiceprint\n"
        "Voice-clone / impersonation risk console — reproduction of "
        "*\"Automatic speaker verification spoofing and deepfake detection "
        "using wav2vec 2.0 and data augmentation\"* (Tak et al., Odyssey 2022). "
        "Source: [GitHub repo](https://github.com/AyeshaKODER/ssl-aasist-reproduction.git)."
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