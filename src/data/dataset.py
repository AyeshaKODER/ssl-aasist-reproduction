"""
Dataset loaders for the ASVspoof protocol file formats.

Audio is loaded lazily per-__getitem__ call via torchaudio — nothing is
pre-loaded into RAM, matching the memory-conscious setup in the README.

ASVspoof2019 LA protocol line format (train/dev):
  SPEAKER_ID  AUDIO_FILE_NAME  -  SYSTEM_ID  KEY(bonafide|spoof)

ASVspoof2021 eval protocol line format:
  SPEAKER_ID  AUDIO_FILE_NAME  CODEC  TRANSMISSION  SYSTEM_ID  KEY  ...
(exact column count varies slightly by track LA/DF — we only require the
file name and, when present, the bonafide/spoof key column.)
"""
import os
import random
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

SEGMENT_SAMPLES = 64600  # ~4s at 16kHz, Section 6.3


def _pad_or_crop(wav, target_len=SEGMENT_SAMPLES, train=True, rng=None):
    """Crop or concatenate to a fixed segment length (Section 6.3:
    'Audio data are cropped or concatenated giving segments of approximately
    4 seconds duration (64,600 samples)')."""
    n = wav.shape[-1]
    if n >= target_len:
        if train:
            rng = rng or random
            start = rng.randint(0, n - target_len)
        else:
            start = 0  # deterministic for eval
        return wav[..., start:start + target_len]
    # concatenate (repeat) to reach target length
    n_repeats = target_len // n + 1
    wav = wav.repeat(1, n_repeats)
    return wav[..., :target_len]


class ASVspoofDataset(Dataset):
    def __init__(self, protocol_path, audio_dir, audio_ext=".flac",
                 train=True, rawboost_fn=None, subset_frac=1.0, seed=0):
        self.audio_dir = audio_dir
        self.audio_ext = audio_ext
        self.train = train
        self.rawboost_fn = rawboost_fn
        self.rng = random.Random(seed)

        self.items = []  # (filename, label) label: 1=bonafide, 0=spoof
        with open(protocol_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            filename = parts[1]
            # key is typically the last column ("bonafide"/"spoof")
            key = parts[-1].lower()
            label = 1 if "bonafide" in key else 0
            self.items.append((filename, label))

        if subset_frac < 1.0:
            n_keep = max(1, int(len(self.items) * subset_frac))
            self.rng.shuffle(self.items)
            self.items = self.items[:n_keep]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filename, label = self.items[idx]
        path = os.path.join(self.audio_dir, filename + self.audio_ext)
        # Use soundfile directly, not torchaudio.load — newer torchaudio
        # versions route load() through the optional torchcodec backend
        # (needs its own ffmpeg setup), which we don't need for plain FLAC/WAV.
        data, sr = sf.read(path, dtype="float32")
        wav = torch.from_numpy(data)
        wav = wav.unsqueeze(0) if wav.ndim == 1 else wav.T  # -> (channels, L)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # mono
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = _pad_or_crop(wav, train=self.train, rng=self.rng)

        if self.rawboost_fn is not None and self.train:
            np_wav = wav.squeeze(0).numpy().astype(np.float32)
            np_wav = self.rawboost_fn(np_wav, sr=16000, seed=self.rng.randint(0, 1 << 30))
            wav = torch.from_numpy(np_wav).unsqueeze(0)

        return wav.squeeze(0), label, filename


def collate_fn(batch):
    wavs, labels, names = zip(*batch)
    wavs = torch.stack(wavs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return wavs, labels, list(names)
