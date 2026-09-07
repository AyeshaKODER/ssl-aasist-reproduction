"""
Self-attentive aggregation layer, Section 5, Eq. 3-5 (paper-exact):

    W = Softmax(conv2d(BN(SeLU(conv2d(S)))))                       (Eq. 3)
    t = sum_F(S * W)                                                (Eq. 4)
    f = sum_T(S * W)                                                (Eq. 5)

"2-D attention maps... using a 2-D convolutional (conv2d) layer with one
kernel-size rather than conventional conv1d based attention applied to a
single domain."

Also implements plain max-pooling extraction (Eq. 1-2 style, used for the
non-SA baseline rows of Table 2/3):
    f = max_T(abs(S)),  t = max_F(abs(S))
"""
import torch
import torch.nn as nn


class SelfAttentiveAggregation(nn.Module):
    """Produces spectral (f) and temporal (t) representations via the 2D
    attention map of Eq. 3-5."""

    def __init__(self, in_channels, kernel_size=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size)
        self.act = nn.SELU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size)

    def forward(self, S):
        # S: (B, C, Freq, Time)
        w = self.conv1(S)
        w = self.act(w)
        w = self.bn(w)
        w = self.conv2(w)
        # Softmax over the joint (Freq, Time) attention map, per channel —
        # this is the "2-D attention weight matrix" described in the text.
        b, c, f, t = w.shape
        W = torch.softmax(w.view(b, c, -1), dim=-1).view(b, c, f, t)

        weighted = S * W  # element-wise multiplication, per Eq. 4-5
        t_repr = weighted.sum(dim=2)  # sum over Freq -> (B, C, Time)   Eq. 4
        f_repr = weighted.sum(dim=3)  # sum over Time -> (B, C, Freq)  Eq. 5
        return f_repr, t_repr


class MaxPoolAggregation(nn.Module):
    """Baseline (non-SA) extraction, Eq. 1-2 style:
    f = max_T(abs(S)), t = max_F(abs(S))."""

    def forward(self, S):
        f_repr = torch.amax(torch.abs(S), dim=3)  # max over Time -> (B, C, Freq)
        t_repr = torch.amax(torch.abs(S), dim=2)  # max over Freq -> (B, C, Time)
        return f_repr, t_repr
