"""
Residual encoder stages, Table 1:

  Post-processing: add channel -> Maxpool-2D(3) -> BN & SeLU
  Res-block x2: [Conv2D((2,3),1,32), BN & SeLU, Conv2D((2,3),1,32)] -> (32,42,67)
  Res-block x4: [Conv2D((2,3),1,64), BN & SeLU, Conv2D((2,3),1,64)] -> (64,42,67)
  BN & SeLU
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
        kh, kw = kernel
        pad_h = kh - 1
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
    """Post-processing (incl. shape-pin to Table 1's (42,67)) + 2-then-4
    residual block stages, operating at that fixed size throughout."""

    def __init__(self, in_channels=1, n_blocks_stage1=2, n_blocks_stage2=4,
                 stage1_ch=32, stage2_ch=64, maxpool_kernel=3):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.act0 = nn.SELU(inplace=True)

        # Shape-pin moved HERE, before the residual blocks — see module
        # docstring. Works for both front-ends: wav2vec2 is already close to
        # this size so it's a no-op in practice; sinc gets shrunk to a cheap
        # working size before any real compute happens.
        self.shape_pin = nn.AdaptiveMaxPool2d((42, 67))

        stage1 = [ResBlock2D(in_channels if i == 0 else stage1_ch, stage1_ch)
                  for i in range(n_blocks_stage1)]
        self.stage1 = nn.Sequential(*stage1)

        stage2 = [ResBlock2D(stage1_ch if i == 0 else stage2_ch, stage2_ch)
                  for i in range(n_blocks_stage2)]
        self.stage2 = nn.Sequential(*stage2)

        self.bn_final = nn.BatchNorm2d(stage2_ch)
        self.act_final = nn.SELU(inplace=True)
        self.out_channels = stage2_ch

    def forward(self, x):
        # x: (B, 1, Freq, Time)
        x = self.maxpool(x)
        x = self.act0(self.bn0(x))
        x = self.shape_pin(x)  # -> (B, 1, 42, 67), THEN the residual blocks run
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.act_final(self.bn_final(x))
        return x  # (B, 64, 42, 67) — S in the paper's notation, Table 1