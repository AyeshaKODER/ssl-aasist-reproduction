"""
Front-end systems, Section 3 & 4 / Fig 2:

(a) sinc-layer baseline: 70 mel-scaled sinc filters, kernel size 129.
(b) wav2vec 2.0 XLS-R (0.3B) front-end, fine-tuned jointly with the backend,
    with an FC layer reducing the transformer output dim (1024 -> 128), per
    Table 1 and Section 4.3/4.4.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# (a) Sinc-layer front-end (Ravanelli & Bengio, SincNet — cited as [46])
# --------------------------------------------------------------------------
class SincConv(nn.Module):
    """Parameterised sinc band-pass filterbank front-end. 70 mel-scaled
    filters, kernel size 129, as specified in Section 3."""

    def __init__(self, out_channels=70, kernel_size=129, sample_rate=16000,
                 in_channels=1, stride=1, min_low_hz=0, min_band_hz=50):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # force odd kernel for symmetric filters
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Mel-scaled initial filter bank cutoffs
        low_hz = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel = torch.linspace(self._to_mel(low_hz), self._to_mel(high_hz), out_channels + 1)
        hz = self._to_hz(mel)

        self.low_hz_ = nn.Parameter(hz[:-1].view(-1, 1))
        self.band_hz_ = nn.Parameter((hz[1:] - hz[:-1]).view(-1, 1))

        n_lin = torch.linspace(0, (kernel_size / 2) - 1, steps=int(kernel_size / 2))
        self.register_buffer("window_", 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size))
        n = (kernel_size - 1) / 2.0
        self.register_buffer("n_", 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate)

    @staticmethod
    def _to_mel(hz):
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, waveforms):
        # waveforms: (B, 1, L)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                            self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low))
                           / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(waveforms, filters, stride=self.stride,
                         padding=self.kernel_size // 2)


class SincFrontend(nn.Module):
    """Sinc-layer front-end + post-processing to a spectro-temporal map,
    matching Fig 2(a): f = max_T(abs(S)), t = max_F(abs(S)) are computed
    downstream in the aggregation layer, not here — this module just
    produces S."""

    def __init__(self, out_channels=70, kernel_size=129, sample_rate=16000):
        super().__init__()
        self.sinc = SincConv(out_channels=out_channels, kernel_size=kernel_size,
                              sample_rate=sample_rate)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SELU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.out_channels = 1  # channel dim added downstream, matching Table 1

    def forward(self, x):
        # x: (B, L) raw waveform
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.sinc(x)  # (B, 70, L')
        x = self.act(self.bn(torch.abs(x)))
        x = self.maxpool(x)  # (B, 70, T)
        return x.unsqueeze(1)  # (B, 1, 70, T) — matches "add channel" in Table 1


# --------------------------------------------------------------------------
# (b) wav2vec 2.0 XLS-R (0.3B) front-end, Section 4
# --------------------------------------------------------------------------
class Wav2Vec2XLSRFrontend(nn.Module):
    """wav2vec2 XLS-R (0.3B) front-end, fine-tuned jointly with the
    downstream classifier (Section 4.3). Uses HuggingFace `transformers`
    instead of fairseq for a lighter local footprint (README note).

    Output: (B, 1024, T) transformer features (Table 1: (201,1024) for a
    64,600-sample / ~4s input at the 20ms CNN stride, Section 4.1), then a
    FC layer reduces 1024 -> 128 (Table 1: "FC (fine-tuning) -> (201,128)").
    """

    HF_CHECKPOINT = "facebook/wav2vec2-xls-r-300m"

    def __init__(self, reduced_dim=128, freeze_feature_extractor=True):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.encoder = Wav2Vec2Model.from_pretrained(self.HF_CHECKPOINT)
        hidden_size = self.encoder.config.hidden_size  # 1024 for XLS-R 300M

        if freeze_feature_extractor:
            # Paper fine-tunes the whole model jointly (Section 4.3), but the
            # low-level CNN feature extractor is conventionally kept frozen
            # in most wav2vec2 fine-tuning recipes to save memory/stabilise
            # training — flip this off if you want to match "fully joint"
            # optimisation exactly.
            self.encoder.feature_extractor._freeze_parameters()

        self.fc = nn.Linear(hidden_size, reduced_dim)  # Table 1: FC (fine-tuning)

    def forward(self, x):
        # x: (B, L) raw waveform, no masking applied during fine-tuning
        # (Section 4.3: "input masking is not applied to hidden features
        # z_{1:N} during fine-tuning")
        out = self.encoder(x).last_hidden_state  # (B, T, 1024)
        out = self.fc(out)  # (B, T, 128) — Table 1 post-FC shape
        out = out.transpose(1, 2)  # (B, 128, T) — Table 1: "transpose o=(128,201) (F,T)"
        return out.unsqueeze(1)  # (B, 1, 128, T) — "add channel"
