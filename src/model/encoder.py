"""
Residual encoder stages, Table 1:

  Post-processing: add channel -> Maxpool-2D(3) -> BN & SeLU
  Res-block x2: [Conv2D((2,3),1,32), BN & SeLU, Conv2D((2,3),1,32)] -> (32,42,67)
  Res-block x4: [Conv2D((2,3),1,64), BN & SeLU, Conv2D((2,3),1,64)] -> (64,42,67)
  BN & SeLU

ENGINEERING NOTE: Table 1 keeps the spatial shape (42,67) fixed across both
res-block stages. A (2,3) kernel with ordinary symmetric padding changes the
even (2-sized) dimension by design, so each conv here uses F.pad to hold
H×W constant (asymmetric pad of (0,1) on the height/freq axis) rather than
guessing a padding tuple that silently drifts the shape — flagged in README
as an engineering choice since the paper doesn't spell out the pad scheme.

SECOND ENGINEERING NOTE (found via the sanity-check run, not guessed up
front): Table 1's (42,67) shape is derived from the wav2vec2 front-end's
already-downsampled ~201-frame output (20ms CNN stride, Section 4.1). The
sinc-layer front-end operates on the raw 64,600-sample waveform directly and
has no equivalent stride-based downsampling specified in the paper — feeding
its much longer time axis straight into the graph attention layers blows up
the O(N^2) attention memory. The paper doesn't give sinc-path shapes at all
(the original AASIST paper it's based on, cited as [10], handles this with
additional strided pooling not detailed here). Fix: an AdaptiveMaxPool2d at
the end of the encoder pins BOTH front-ends to Table 1's stated (42,67)
target, so the wav2vec2 path is unaffected (it already lands near there) and
the sinc path is forced into a comparable, tractable shape. This is a
necessary engineering addition, not a paper-specified detail.
"""
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.SELU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(2, 3))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.SELU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(2, 3))
        self.shortcut = (nn.Conv2d(in_ch, out_ch, kernel_size=1)
                          if in_ch != out_ch else nn.Identity())

    @staticmethod
    def _same_pad(x, kernel=(2, 3)):
        # Pad so a (2,3) kernel with stride 1 leaves H,W unchanged.
        kh, kw = kernel
        pad_h = kh - 1  # asymmetric on the freq axis (even kernel)
        pad_w = kw - 1
        return F.pad(x, (pad_w // 2, pad_w - pad_w // 2, 0, pad_h))

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act1(self.bn1(x))
        out = self.conv1(self._same_pad(out))
        out = self.act2(self.bn2(out))
        out = self.conv2(self._same_pad(out))
        return out + identity


class ResidualEncoder(nn.Module):
    """Post-processing + 2-then-4 residual block stages, Table 1."""

    def __init__(self, in_channels=1, n_blocks_stage1=2, n_blocks_stage2=4,
                 stage1_ch=32, stage2_ch=64, maxpool_kernel=3):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.act0 = nn.SELU(inplace=True)

        stage1 = [ResBlock2D(in_channels if i == 0 else stage1_ch, stage1_ch)
                  for i in range(n_blocks_stage1)]
        self.stage1 = nn.Sequential(*stage1)

        stage2 = [ResBlock2D(stage1_ch if i == 0 else stage2_ch, stage2_ch)
                  for i in range(n_blocks_stage2)]
        self.stage2 = nn.Sequential(*stage2)

        self.bn_final = nn.BatchNorm2d(stage2_ch)
        self.act_final = nn.SELU(inplace=True)
        self.out_channels = stage2_ch

        # Pin the output to Table 1's stated (42,67) target regardless of
        # front-end — see "SECOND ENGINEERING NOTE" above. Uses adaptive
        # pooling rather than a fixed-stride scheme so it works for both the
        # wav2vec2 path (already near this shape) and the much longer
        # sinc-layer path.
        self.shape_pin = nn.AdaptiveMaxPool2d((42, 67))

    def forward(self, x):
        # x: (B, 1, Freq, Time)
        x = self.maxpool(x)
        x = self.act0(self.bn0(x))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.act_final(self.bn_final(x))
        x = self.shape_pin(x)
        return x  # (B, 64, 42, 67) — S in the paper's notation, Table 1
