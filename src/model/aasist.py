"""
Full model: front-end -> residual encoder -> (SA or max-pool) aggregation ->
parallel graph modules -> MGO -> readout -> FC(2), matching Fig. 1 + Table 1.
"""
import torch
import torch.nn as nn

from .frontends import SincFrontend, Wav2Vec2XLSRFrontend
from .encoder import ResidualEncoder
from .aggregation import SelfAttentiveAggregation, MaxPoolAggregation
from .graph_modules import GraphModule, MaxGraphOperation, readout


class SSLAASIST(nn.Module):
    def __init__(self, frontend="wav2vec2", use_sa=True,
                 d_st=64, pool_ratio=0.5, freeze_wav2vec_feature_extractor=True):
        super().__init__()
        if frontend == "sinc":
            self.frontend = SincFrontend()
        elif frontend == "wav2vec2":
            self.frontend = Wav2Vec2XLSRFrontend(
                freeze_feature_extractor=freeze_wav2vec_feature_extractor)
        else:
            raise ValueError(f"Unknown frontend: {frontend}")

        self.encoder = ResidualEncoder(in_channels=1)
        enc_ch = self.encoder.out_channels  # 64, per Table 1

        self.use_sa = use_sa
        self.aggregation = (SelfAttentiveAggregation(enc_ch) if use_sa
                             else MaxPoolAggregation())

        # Graph modules operate on node-sequences; treat the pooled
        # freq/time axis as the node dimension, channel count as node dim.
        self.graph_s = GraphModule(enc_ch, d_st, pool_ratio=pool_ratio)
        self.graph_t = GraphModule(enc_ch, d_st, pool_ratio=pool_ratio)

        self.mgo = MaxGraphOperation(d_st, d_st, d_st, pool_ratio=pool_ratio)

        readout_dim = d_st // 2 * 4 + d_st  # 4 pooled-half stats + stack node
        # NB: MGO branches operate on half-node splits of d_st-dim nodes, so
        # readout width is derived from the actual tensor at runtime in
        # forward(); this attribute is a fallback estimate only.
        self.fc_out = None  # lazily built once we see the true readout width
        self._d_st = d_st

    def _build_fc(self, in_dim, device):
        self.fc_out = nn.Linear(in_dim, 2).to(device)

    def forward(self, x):
        # x: (B, L) raw waveform, L = 64,600 (Section 6.3)
        S = self.frontend(x)  # (B, 1, Freq, Time)
        S = self.encoder(S)  # (B, 64, F', T')

        f_repr, t_repr = self.aggregation(S)  # (B, C, F'), (B, C, T')
        f_repr = f_repr.transpose(1, 2)  # (B, F', C) — node-sequence format
        t_repr = t_repr.transpose(1, 2)  # (B, T', C)

        g_s = self.graph_s(f_repr)  # spectral graph
        g_t = self.graph_t(t_repr)  # temporal graph

        g_st, stack = self.mgo(g_s, g_t)
        r = readout(g_st, stack)  # (B, readout_dim)

        if self.fc_out is None:
            self._build_fc(r.shape[-1], r.device)
        return self.fc_out(r)  # (B, 2) — bona fide / spoofed logits


# --- Config presets matching every row of Table 2 / Table 3 ---------------
PRESETS = {
    "sinc_no_sa_no_da":   dict(frontend="sinc",     use_sa=False, da=None),
    "wav2vec_no_sa_no_da": dict(frontend="wav2vec2", use_sa=False, da=None),
    "sinc_sa_no_da":      dict(frontend="sinc",     use_sa=True,  da=None),
    "wav2vec_sa_no_da":   dict(frontend="wav2vec2", use_sa=True,  da=None),
    "sinc_sa_da":         dict(frontend="sinc",     use_sa=True,  da="la"),
    "wav2vec_sa_da":      dict(frontend="wav2vec2", use_sa=True,  da="la"),
    # DF-database variants (Table 3) use the DF-optimised RawBoost config
    "wav2vec_sa_da_df":   dict(frontend="wav2vec2", use_sa=True,  da="df"),
    "sinc_sa_da_df":      dict(frontend="sinc",     use_sa=True,  da="df"),
}


def build_model(preset_name, **overrides):
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Options: {list(PRESETS)}")
    cfg = dict(PRESETS[preset_name])
    cfg.update(overrides)
    da = cfg.pop("da", None)
    model = SSLAASIST(frontend=cfg["frontend"], use_sa=cfg["use_sa"])
    return model, da
