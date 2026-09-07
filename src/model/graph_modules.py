"""
Graph attention modules, Section 3:

  G_t = graph_module(max_F(abs(S)))                                (Eq. 1)
  G_s = graph_module(max_T(abs(S)))                                 (Eq. 2)

  "Spectral and temporal graphs G_s and G_t are modelled using a pair of
  parallel graph modules, each comprising a graph attention network (GAT)
  [48] and a graph-pooling layer [49]."

  "A heterogeneous spectro-temporal graph (G_st) is then formed by combining
  temporal and spectral graphs using a heterogeneous stacking graph attention
  layer (HS-GAL)... An HS-GAL contains an attention mechanism modified to
  accommodate graph heterogeneity [50] and an additional stack node [28]."

  "HS-GALs are applied with a max graph operation (MGO) where two branches,
  each consisting of two HS-GALs, learn to detect different spoofing
  artefacts in parallel... An element-wise maximum operation is applied to
  the branch outputs."

  "The readout scheme uses node-wise maximum and average operations. The
  output is formed from the concatenation of five nodes... the fifth is the
  copied stack node."

ENGINEERING NOTE (see README): the paper cites [48] (GAT) and [50]
(heterogeneous attention) for the full attention formula rather than giving
a closed form itself. Implemented below is the standard single-head GAT
attention plus a heterogeneous variant that lets nodes from two different
input spaces (spectral, temporal) attend to each other after being projected
to a common dimension d_st, with an extra learned "stack node" carried
between successive HS-GALs as described.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphPool(nn.Module):
    """Graph-pooling layer [49]: learns a per-node score and keeps the
    top-k*ratio nodes (k = pooling ratio, 0.5 throughout — Section 6.3)."""

    def __init__(self, in_dim, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.score = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: (B, N, D)
        scores = torch.sigmoid(self.score(x)).squeeze(-1)  # (B, N)
        n_keep = max(1, int(x.shape[1] * self.ratio))
        topk = torch.topk(scores, n_keep, dim=1)
        idx = topk.indices  # (B, n_keep)
        gate = topk.values.unsqueeze(-1)  # (B, n_keep, 1)
        x_kept = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        return x_kept * gate  # gated pooled nodes


class GATLayer(nn.Module):
    """Standard single-head graph attention [Velickovic et al., GAT] over a
    fully-connected node graph (each node attends to every other node)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x: (B, N, D)
        h = self.W(x)  # (B, N, D')
        B, N, D = h.shape
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        e = self.act(self.a(torch.cat([h_i, h_j], dim=-1))).squeeze(-1)  # (B, N, N)
        alpha = torch.softmax(e, dim=-1)
        out = torch.bmm(alpha, h)  # (B, N, D')
        return F.elu(out)


class GraphModule(nn.Module):
    """GAT layer + graph-pooling layer, Eq. 1-2's "graph_module(...)"."""

    def __init__(self, in_dim, out_dim, pool_ratio=0.5):
        super().__init__()
        self.gat = GATLayer(in_dim, out_dim)
        self.pool = GraphPool(out_dim, ratio=pool_ratio)

    def forward(self, x):
        x = self.gat(x)
        x = self.pool(x)
        return x


class HeteroStackingGAL(nn.Module):
    """Heterogeneous stacking graph attention layer (HS-GAL).

    Projects spectral & temporal node sets to a common dimension d_st,
    concatenates them (plus an incoming stack node) into one heterogeneous
    graph, applies GAT-style attention across the combined node set, and
    emits an updated stack node to pass to the next HS-GAL — matching "the
    stack node of each preceding HS-GAL is fed to the following HS-GAL so
    that information in both temporal and spectral graphs is preserved."
    """

    def __init__(self, d_s, d_t, d_st, pool_ratio=0.5):
        super().__init__()
        self.proj_s = nn.Linear(d_s, d_st)
        self.proj_t = nn.Linear(d_t, d_st)
        self.gat = GATLayer(d_st, d_st)
        self.pool = GraphPool(d_st, ratio=pool_ratio)
        self.stack_update = nn.Linear(d_st, d_st)

    def forward(self, g_s, g_t, stack_node=None):
        # g_s: (B, N_s, d_s), g_t: (B, N_t, d_t)
        s_proj = self.proj_s(g_s)
        t_proj = self.proj_t(g_t)
        nodes = torch.cat([s_proj, t_proj], dim=1)  # (B, N_s+N_t, d_st) = G_st

        if stack_node is None:
            stack_node = nodes.mean(dim=1, keepdim=True)  # init stack node
        nodes_with_stack = torch.cat([nodes, stack_node], dim=1)

        attended = self.gat(nodes_with_stack)
        pooled = self.pool(attended[:, :-1, :])  # pool everything but stack node
        new_stack = self.stack_update(attended[:, -1:, :])
        return pooled, new_stack


class MaxGraphOperation(nn.Module):
    """Two parallel branches, each two stacked HS-GALs sharing a common
    stack node, combined with an element-wise maximum (MGO)."""

    def __init__(self, d_s, d_t, d_st, pool_ratio=0.5):
        super().__init__()
        self.branch1_l1 = HeteroStackingGAL(d_s, d_t, d_st, pool_ratio)
        self.branch1_l2 = HeteroStackingGAL(d_st, d_st, d_st, pool_ratio)
        self.branch2_l1 = HeteroStackingGAL(d_s, d_t, d_st, pool_ratio)
        self.branch2_l2 = HeteroStackingGAL(d_st, d_st, d_st, pool_ratio)

    def forward(self, g_s, g_t):
        # Branch 1
        n1, stack1 = self.branch1_l1(g_s, g_t)
        half = n1.shape[1] // 2
        n1, stack1 = self.branch1_l2(n1[:, :half], n1[:, half:], stack1)

        # Branch 2 (independent parallel branch, per "two branches... learn
        # to detect different spoofing artefacts in parallel")
        n2, stack2 = self.branch2_l1(g_s, g_t)
        half2 = n2.shape[1] // 2
        n2, stack2 = self.branch2_l2(n2[:, :half2], n2[:, half2:], stack2)

        n_min = min(n1.shape[1], n2.shape[1])
        g_st = torch.maximum(n1[:, :n_min], n2[:, :n_min])  # element-wise max
        stack = torch.maximum(stack1, stack2)
        return g_st, stack


def readout(g_st, stack_node):
    """Node-wise max & average over spectral/temporal nodes, concatenated
    with the (copied) stack node — 5 nodes total, per the paper's readout
    description."""
    max_all = g_st.amax(dim=1)  # (B, D)
    avg_all = g_st.mean(dim=1)  # (B, D)
    half = g_st.shape[1] // 2
    max_spec = g_st[:, :half].amax(dim=1)
    avg_spec = g_st[:, :half].mean(dim=1)
    max_temp = g_st[:, half:].amax(dim=1) if g_st.shape[1] > half else max_all
    stack = stack_node.squeeze(1)
    return torch.cat([max_spec, avg_spec, max_temp, avg_all, stack], dim=-1)
