"""
PyTorch port of QMixer (originally in src/rl/DQRLAgentDist_API.py:1118).

Monotonic hypernetwork that maps (agent_qs, global_state) → Q_tot.
Monotonicity enforced via abs() on w1 and w2.
Activation: ELU for hidden layer (matches TF tf.nn.elu).
"""
import torch
import torch.nn as nn


class QMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        # Hypernetwork 1 — weights for 1st mixing layer
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Hypernetwork 2 — weights for 2nd (output) layer
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        # Bias network (V(s)) — two-layer MLP, no abs needed
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        agent_qs : (B, n_agents)  — Q-values of selected actions (masked)
        states   : (B, state_dim) — global state
        returns  : (B, 1)         — Q_tot
        """
        B = agent_qs.size(0)

        # First layer weights (monotonicity via abs)
        w1 = torch.abs(self.hyper_w1(states)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(B, 1, self.embed_dim)

        # agent_qs as row vector: (B, 1, n_agents)
        qs = agent_qs.view(B, 1, self.n_agents)
        # (B, 1, n_agents) @ (B, n_agents, embed_dim) → (B, 1, embed_dim)
        hidden = torch.nn.functional.elu(torch.bmm(qs, w1) + b1)

        # Second layer weights (monotonicity via abs)
        w2 = torch.abs(self.hyper_w2(states)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(B, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2   # (B, 1, 1)
        return q_tot.view(B, 1)
