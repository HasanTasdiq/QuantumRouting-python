"""
PyTorch port of QRoutingGATFlat (originally in src/rl/GNN.py).

Architecture (mirrors TF version exactly):
  Input (B, state_dim=20228)
    → Linear(1024, relu)          [projection_1]
    → Linear(num_nodes*hidden_dim, relu)  [projection_2]
    → reshape (B, num_nodes, hidden_dim)
    → MultiheadAttention(heads=4, embed=hidden_dim) + residual + LayerNorm
    → FFN: Linear(hidden_dim, relu) → Linear(hidden_dim) + residual + LayerNorm
    → Linear(hidden_dim, 1)
    → squeeze → (B, num_nodes)
"""
import torch
import torch.nn as nn


class QRoutingGATFlat(nn.Module):
    def __init__(self, num_nodes=100, input_flat_dim=20228, hidden_dim=64, num_heads=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        self.proj1 = nn.Linear(input_flat_dim, 1024)
        self.proj2 = nn.Linear(1024, num_nodes * hidden_dim)

        # batch_first=True → input/output shape (B, seq_len, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.ln1 = nn.LayerNorm(hidden_dim)

        # FFN matches TF: Dense(hidden_dim, relu) → Dense(hidden_dim, linear)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, input_flat_dim)
        h = torch.relu(self.proj1(x))
        h = torch.relu(self.proj2(h))
        h = h.view(-1, self.num_nodes, self.hidden_dim)   # (B, N, D)

        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = self.ln1(h + attn_out)

        h = self.ln2(h + self.ffn(h))

        return self.q_out(h).squeeze(-1)   # (B, num_nodes)
