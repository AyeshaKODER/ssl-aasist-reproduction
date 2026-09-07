import sys

print("1. starting imports...", flush=True)
import torch
print("2. torch imported, version:", torch.__version__, flush=True)
import soundfile as sf
print("3. soundfile imported", flush=True)

from model.aasist import SSLAASIST
print("4. model module imported", flush=True)

from data.dataset import ASVspoofDataset
print("5. dataset module imported", flush=True)

print("6. building dataset (protocol parse only, no audio read yet)...", flush=True)
ds = ASVspoofDataset(
    "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "data/LA/ASVspoof2019_LA_train/flac",
    train=True,
    subset_frac=0.01,
)
print(f"7. dataset built, {len(ds)} items", flush=True)

print("8. reading item 0 (raw soundfile read)...", flush=True)
filename, label = ds.items[0]
path = f"data/LA/ASVspoof2019_LA_train/flac/{filename}.flac"
print("   path:", path, flush=True)
data, sr = sf.read(path, dtype="float32")
print(f"9. audio read OK, shape={data.shape}, sr={sr}", flush=True)

print("10. running full __getitem__(0)...", flush=True)
wav, lbl, name = ds[0]
print(f"11. __getitem__ OK, wav shape={wav.shape}, label={lbl}", flush=True)

print("12. building model (sinc, no SA)...", flush=True)
model = SSLAASIST(frontend="sinc", use_sa=False)
print("13. model built", flush=True)

print("14. running forward pass on one sample...", flush=True)
with torch.no_grad():
    out = model(wav.unsqueeze(0))
print(f"15. forward pass OK, output shape={out.shape}", flush=True)

print("ALL STEPS PASSED — the crash is happening in train.py's loop/DataLoader/tqdm machinery, not the core pipeline.", flush=True)