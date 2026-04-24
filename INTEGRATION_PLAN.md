# QuRA ↔ RELiQ Unification: Execution Plan

**Audience**: Sonnet (executing agent)
**Branch**: new branch `qura-pytorch-reliq` off `dqrl_dist_claude`
**Goal**: Full PyTorch port of QuRA + RELiQ integrated as a baseline inside QuRA's repo. Single framework, single physics, single experiment runner.

---

## 0. Ground Rules (Read First)

- **Do not modify `main` or `dqrl_dist_claude`.** Create `qura-pytorch-reliq` and work there.
- **Framework**: PyTorch 2.9.x + torch_geometric 2.7.x, CPU-only. No TensorFlow imports anywhere in new code.
- **Physics is canonical QuRA**: Werner swap `f1*f2 + (1-f1)*(1-f2)/3`, `entanglement_lifetime = 10`, Waxman topology with existing params. RELiQ's env is replaced with a thin adapter over QuRA's `Topo`/`AlgorithmBase` — RELiQ's *algorithm* is untouched.
- **Full-fidelity distributed training port**: Redis + HTTP workers, FedAvg, QMixer, all four variants (Seq / Flock / Guard / Hive). No simplifications, no toy reimplementations.
- **RELiQ is a baseline**: behaves like any other `AlgorithmBase` subclass in `Run.py`. Do not alter its algorithm (GNN + NetMon + DQN).
- **Keep action space `Discrete(3)`** in the RELiQ adapter. Keep Waxman `(100, 0.9, 5, 0.0002, 6, gridSize=10)`.
- **Validate phase-by-phase.** Each phase has a checkpoint; do not proceed until it passes.

---

## 1. Branch & Directory Layout

### 1.1 Create branch
```bash
cd "/Users/tasdiqulislam/Documents/Quantum Network/Routing/QuantumRouting-python"
git checkout dqrl_dist_claude
git checkout -b qura-pytorch-reliq
```

### 1.2 Target structure (new dirs in **bold**)
```
QuantumRouting-python/
├── src/
│   ├── quantum/                     # unchanged (physics, Topo, AlgorithmBase)
│   │   ├── algorithm/
│   │   │   ├── AlgorithmBase.py
│   │   │   ├── DQRL_dist1.py        # QuRA-Seq  (TF) — keep as reference, not imported
│   │   │   ├── DQRL_dist2.py        # QuRA-Flock
│   │   │   ├── DQRL_dist3.py        # QuRA-Guard
│   │   │   ├── DQRL_dist4.py        # QuRA-Hive
│   │   │   ├── **DQRL_dist1_pt.py**    # QuRA-Seq  (PyTorch)
│   │   │   ├── **DQRL_dist2_pt.py**    # QuRA-Flock
│   │   │   ├── **DQRL_dist3_pt.py**    # QuRA-Guard
│   │   │   ├── **DQRL_dist4_pt.py**    # QuRA-Hive
│   │   │   ├── **RELiQ_Adapter.py**    # RELiQ as AlgorithmBase
│   │   │   └── Run.py                  # modified to use *_pt variants + RELiQ
│   │   └── topo/Topo.py             # unchanged
│   ├── rl/
│   │   ├── DQRLAgentDist_API.py     # TF agent — keep as reference
│   │   ├── GNN.py                   # TF GAT — keep as reference
│   │   ├── **pt/**                  # PyTorch port of the RL stack
│   │   │   ├── __init__.py
│   │   │   ├── gat_flat.py          # QRoutingGATFlat (PyTorch)
│   │   │   ├── qmixer.py            # QMixer (PyTorch)
│   │   │   ├── agent.py             # DQRLAgentDist (PyTorch)
│   │   │   ├── replay.py            # ReplayBuffer
│   │   │   ├── redis_io.py          # model checkpoint ↔ Redis
│   │   │   ├── server.py            # Flask/FastAPI HTTP worker
│   │   │   └── fedavg.py            # FedAvg aggregator
│   │   ├── deploy_full.sh           # modified for *_pt variants
│   │   └── save_trained_model.py    # modified to save .pt
│   └── **reliq/**                   # verbatim copy of RELiQ's src/
│       ├── model.py                 # DQN, DQNR, DGN, CommNet, NetMon
│       ├── env/
│       │   ├── entanglementenv.py   # ← modify physics here only
│       │   ├── quantum_network.py
│       │   ├── environment.py
│       │   ├── wrapper.py
│       │   └── constants.py         # ← set FIDELITY params to QuRA's
│       ├── buffer.py
│       ├── replaybuffer.py
│       ├── policy.py
│       ├── util.py
│       └── train.py                 # trimmed training entry point for RELiQ
├── INTEGRATION_PLAN.md              # this file
├── requirements.txt                 # updated
└── runs_quantum/                    # RELiQ training artifacts go here
```

**Rule**: the four `DQRL_distN.py` files and `DQRLAgentDist_API.py` / `GNN.py` remain in the tree but are **not imported** from Run.py after the port. Keep them for A/B sanity checks only.

---

## 2. Dependencies

### 2.1 `requirements.txt` (replace, not append)
```
torch==2.9.1
torch-geometric==2.7.0
numpy>=1.26,<2.0
networkx>=3.0
gymnasium==1.2.3
redis>=5.0
flask>=3.0
requests>=2.31
tqdm
pandas
matplotlib
scipy
psutil
```

Remove `tensorflow`, `keras`. Install:
```bash
pip install -r requirements.txt
```

Verify:
```bash
python -c "import torch; import torch_geometric; import gymnasium; print(torch.__version__)"
```

---

## 3. Phase 1 — Port QuRA Model Layer to PyTorch

**Source files to port (reference only)**:
- `src/rl/GNN.py` → `QRoutingGATFlat` (TF Keras)
- `src/rl/DQRLAgentDist_API.py` → `QMixer` (TF Keras), agent class, `_train_step`

### 3.1 `src/rl/pt/gat_flat.py` — `QRoutingGATFlat`

**Exact architecture to replicate** (from `GNN.py`):
```
Input:  (B, 20228)                       # 128 + SIZE + 2*SIZE^2, SIZE=100
  → Dense(1024, relu)                    # projection_1
  → Dense(num_nodes * hidden_dim, relu)  # projection_2, hidden_dim=64, num_nodes=100
  → reshape to (B, num_nodes, hidden_dim)
  → MultiHeadAttention(num_heads=4, key_dim=hidden_dim)  # self-attn over nodes
  → residual + LayerNorm
  → FFN: Dense(ff_dim=128, relu) → Dense(hidden_dim)
  → residual + LayerNorm
  → Dense(1) per node
  → squeeze last dim → (B, num_nodes)    # Q-values over nodes
```

**PyTorch implementation skeleton**:
```python
# src/rl/pt/gat_flat.py
import torch
import torch.nn as nn

class QRoutingGATFlat(nn.Module):
    def __init__(self, state_dim=20228, num_nodes=100, hidden_dim=64,
                 num_heads=4, ff_dim=128):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.proj1 = nn.Linear(state_dim, 1024)
        self.proj2 = nn.Linear(1024, num_nodes * hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):            # x: (B, 20228)
        h = torch.relu(self.proj1(x))
        h = torch.relu(self.proj2(h))
        h = h.view(-1, self.num_nodes, self.hidden_dim)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.ln1(h + a)
        h = self.ln2(h + self.ffn(h))
        return self.head(h).squeeze(-1)  # (B, num_nodes)
```

**Parity test** (`tests/test_gat_parity.py`, optional but recommended): build the TF model, export weights to numpy, load into PyTorch module, feed identical random input, assert `max(|out_tf - out_pt|) < 1e-4`.

### 3.2 `src/rl/pt/qmixer.py` — `QMixer`

**Reference**: `DQRLAgentDist_API.py:1118`. Monotonic hypernetwork.

```python
# src/rl/pt/qmixer.py
import torch
import torch.nn as nn

class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs, state):
        # agent_qs: (B, n_agents), state: (B, state_dim)
        B = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        hidden = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(B)  # (B,)
```

Monotonicity via `abs()` matches the TF original. `embed_dim=32`, `state_dim` = global-state size (set per variant; for Hive this is `128 + SIZE + 2*SIZE**2 = 20228`).

### 3.3 `src/rl/pt/replay.py` — Replay buffer

Copy the interface used in `DQRLAgentDist_API.py`. Store as CPU tensors / numpy arrays. Support both per-agent and joint (padded) sampling for QMIX. Minimum API:
```python
class ReplayBuffer:
    def __init__(self, capacity): ...
    def push(self, s, a, r, s_next, done, global_s=None, global_s_next=None): ...
    def sample(self, batch_size): ...   # returns tensors
    def __len__(self): ...
```

Include a `PaddedEpisodeBuffer` variant used by Hive (stores full episodes with `padded_states`, `padded_actions`, `global_states`, `next_global_states` — see TF `_train_step` signature).

### 3.4 `src/rl/pt/agent.py` — `DQRLAgentDist` (PyTorch)

Mirrors `DQRLAgentDist_API.py:112`. Exposes the same public methods used elsewhere:
- `__init__(state_size, action_size, num_nodes=100, hidden_dim=64, ...)`
- `act(state, eps)` — ε-greedy over Q-values → int in `[0, action_size)`
- `remember(...)` → replay push
- `replay(batch_size)` — single-agent DQN update (for Seq/Flock/Guard)
- `qmix_train_step(padded_states, padded_actions, rewards, dones, global_states, next_global_states)` — joint QMIX update for Hive
- `load_weights(path)` / `save_weights(path)` — `.pt` files
- `load_from_redis(key)` / `save_to_redis(key)` — via `redis_io.py`

**Key details**:
- Use Double DQN: `a* = argmax_a Q_online(s', a)`, target = `r + γ * Q_target(s', a*)`.
- Target network hard-update every `target_update_every` steps (keep the TF cadence).
- `γ`, `lr`, `batch_size`, `target_update_every`, `epsilon` schedule: read from the TF defaults in `DQRLAgentDist_API.py` and preserve them verbatim.
- Loss: Huber (`SmoothL1Loss`). Optimizer: `Adam`, same `lr`.
- `INFERENCE_MODE` env var: if truthy, `replay()` and `qmix_train_step()` are no-ops.
- CPU-only: `device = torch.device("cpu")` everywhere.

### 3.5 Phase 1 checkpoint
```bash
python -c "
import torch
from src.rl.pt.gat_flat import QRoutingGATFlat
from src.rl.pt.qmixer import QMixer
m = QRoutingGATFlat()
x = torch.randn(2, 20228)
print('GAT out:', m(x).shape)           # expect (2, 100)
mix = QMixer(n_agents=4, state_dim=20228)
qs = torch.randn(2, 4); s = torch.randn(2, 20228)
print('Mix out:', mix(qs, s).shape)     # expect (2,)
"
```

---

## 4. Phase 2 — Distributed Training Infrastructure

Preserve the Redis + HTTP architecture from `DQRL_dist1.py`. Do not collapse it into an in-process trainer.

### 4.1 `src/rl/pt/redis_io.py`
```python
import io, redis, torch

def save_model_to_redis(r, key, model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    r.set(key, buf.getvalue())

def load_model_from_redis(r, key, model):
    raw = r.get(key)
    if raw is None: return False
    buf = io.BytesIO(raw)
    model.load_state_dict(torch.load(buf, map_location="cpu"))
    return True
```
Keep the same Redis key schema used by the TF stack: `model_{variant}`, `batch_{timeSlot}`, `action_{actionId}`, `reqId_{reqId}_ent_matrix`, etc. Do not rename keys.

### 4.2 `src/rl/pt/server.py` — HTTP worker
Replicate the Flask routes used by `DQRL_dist1.py` (`call_learn_and_predict_api`, `call_update_reward`, etc.). One process per worker port; Run.py already round-robins via `_WORKER_PORTS` and `_instance_counter` — leave that logic intact.

Routes (keep names and payload shapes identical so Run.py glue doesn't change):
- `POST /predict` → returns action id
- `POST /update_reward` → pushes transition into the replay buffer
- `POST /train` → triggers one `replay()` or `qmix_train_step()` call
- `GET /health`

Boot sequence matches `deploy_full.sh` — one server per variant per port.

### 4.3 `src/rl/pt/fedavg.py`
Periodic weight averaging across workers. Same trigger cadence as TF implementation; copy the schedule constants. Aggregate via `state_dict()` mean, push back to Redis under `model_{variant}`.

### 4.4 `deploy_full.sh`
Modify to launch PyTorch workers instead of TF. Same ports (`8080`, `8081`, ...), same number of workers per variant. Redis startup unchanged. Verify each worker responds to `/health` before Run.py starts.

### 4.5 Phase 2 checkpoint
- Start Redis, launch 1 worker, `curl /health` → 200.
- Seed replay via `/update_reward`, call `/train`, confirm weights in Redis change.
- FedAvg: launch 2 workers with different seeds, verify post-aggregation `state_dict` is mean of both.

---

## 5. Phase 3 — Port the Four Variants

For each TF variant file, create the `*_pt.py` sibling. Keep class names the same so Run.py edits stay minimal.

### 5.1 Files
| TF source | PyTorch target | Class |
|---|---|---|
| `DQRL_dist1.py` | `DQRL_dist1_pt.py` | `QuRA_DQRL_DIST` → rename class to `QuRA_Seq_DIST` for clarity, alias old name |
| `DQRL_dist2.py` | `DQRL_dist2_pt.py` | `QuRA_Flock_DIST` |
| `DQRL_dist3.py` | `DQRL_dist3_pt.py` | `QuRA_Guard_DIST` |
| `DQRL_dist4.py` | `DQRL_dist4_pt.py` | `QuRA_Hive_DIST` |

Each subclasses `AlgorithmBase` and keeps the `work(pairs, time_)` contract and `p2`/`tryEntanglement`/`p4`/`postProcess` flow. **Do not refactor the routing logic**, only swap the RL backend:
- `call_learn_and_predict_api(...)` → new PyTorch worker endpoint (same URL shape).
- Shared-memory layout (`reqId_{reqId}_ent_matrix`, etc.) unchanged.
- State construction (`curr_emb(64) + context_vec(64) + local(100) + ent_flat(10000) + dist_flat(10000) = 20228`) unchanged.
- Reward weighting (Eq. 9, `λ=0.3, μ=1.0, ν=0.5`) unchanged.
- `MAX_REQUESTS` (200 paper / 15 smoke) preserved.
- `resolve_conflict()` (Guard/Flock) logic copied verbatim.

### 5.2 Hive-specific notes
`DQRL_dist4_pt.py` is the only variant that calls `qmix_train_step`. Build the `PaddedEpisodeBuffer` around the same per-timeslot batch structure the TF version uses (`padded_states`, `padded_actions`, `global_states`, `next_global_states`). Global state for QMixer = the 20228-dim flat global state; agents = per-request agents up to `MAX_REQUESTS`.

### 5.3 Phase 3 checkpoint
Smoke test each variant with `TTIME=1000 STEP=500 TIMES=1 TRAIN_LOAD=5 REQ_LOADS=5 INFERENCE_MODE=0`:
```bash
bash src/rl/deploy_full.sh
python src/quantum/algorithm/Run.py
```
Expected: `/tmp/qrouting_logs/progress_QuRA_{Seq,Flock,Guard,Hive}_DIST_req5.csv` exists with non-zero `successful_requests` rows.

---

## 6. Phase 4 — RELiQ as a Baseline

### 6.1 Copy source
```bash
cp -R "../RELiQ/src/"* src/reliq/
```
Remove the top-level `src/reliq/main.py` — replaced by `src/reliq/train.py` (below) and the adapter.

### 6.2 Physics patch (only file to change inside RELiQ)
`src/reliq/env/constants.py`:
- `FIDELITY_THRESHOLD = 0.5` → keep.
- Set decay params so that RELiQ's `decay_realistic` matches QuRA's Werner + `entanglement_lifetime=10`:
  - Replace `QuantumLink.decay()` formula with:
    ```python
    def swap_fidelity(f1, f2):
        return f1 * f2 + (1 - f1) * (1 - f2) / 3.0
    ```
  - In `entanglementenv.py`, wherever two links are swapped call `swap_fidelity` instead of RELiQ's native formula.
  - Entanglement lifetime: use QuRA's `entanglement_lifetime=10` timeslots (drop the link after 10 steps unused).

Leave the GNN / DQN / NetMon code untouched.

### 6.3 `src/reliq/train.py` (standalone trainer)
Trim RELiQ's original `main.py` to just the training loop with these args baked in:
```
--use-future-rewards --netmon-agg-type=sage --no-idle-action
--request-based-observation --fixed-requests --action-mask
--disable-progressbar --total-steps=15_000_000 --step-between-train=200
--step-before-train=100_000 --netmon --model=dqn --device=cpu
--capacity=100_000 --min-path-length=1 --output-dir=runs_quantum
--comment=RELiQ_QuRAPhysics
```
(Override `--device=cpu`, `--total-steps` can be reduced to `500_000` for a first CPU run — document this in the plan's run notes.)

Output: `runs_quantum/RELiQ_QuRAPhysics/model.pt`.

### 6.4 `src/quantum/algorithm/RELiQ_Adapter.py`
Wraps a trained RELiQ policy as an `AlgorithmBase`. Signature:
```python
from .AlgorithmBase import AlgorithmBase, AlgorithmResult
from src.reliq.model import DQN          # or DQNR, depending on --model
from src.reliq.env.wrapper import NetMonWrapper

class RELiQ_Adapter(AlgorithmBase):
    def __init__(self, topo, name='RELiQ', model_path='runs_quantum/RELiQ_QuRAPhysics/model.pt'):
        super().__init__(topo)
        self.name = name
        self.policy = self._load_policy(model_path)

    def work(self, pairs, time_):
        # 1. Translate Topo state → RELiQ obs/adj tensors (see §7)
        # 2. For each request in `pairs`, run policy greedily hop-by-hop
        # 3. Apply chosen paths through Topo via existing p2/tryEntanglement/p4 helpers
        # 4. Return AlgorithmResult with successfulRequestPerRound, rewardPerRound
        ...
```

Keep action space `Discrete(3)` on the RELiQ side. The adapter maps the 3 actions onto the neighbor list via the same convention RELiQ used in training (top-k by link fidelity / queue, configurable in `wrapper.py`). Conflict resolution inside `work()` follows the RELiQ paper: if two requests pick the same link in the same timeslot, drop the one with the longer remaining path (matches the paper's tie-break).

### 6.5 Phase 4 checkpoint
```bash
python -m src.reliq.train --total-steps=50000 --device=cpu   # short sanity train
python -c "
from src.quantum.topo.Topo import Topo
from src.quantum.algorithm.RELiQ_Adapter import RELiQ_Adapter
topo = Topo.generate(20, 0.9, 5, 0.0002, 6, gridSize=5)
alg = RELiQ_Adapter(topo)
print('loaded RELiQ policy ok')
"
```

---

## 7. Phase 5 — Physics / State Bridge

The adapter has to translate between two state representations:

| QuRA (Topo) | RELiQ (obs, adj) |
|---|---|
| `topo.nodes[i].remainingQubits` | node feature `qubits_free` |
| link fidelity `l.fidelity` | edge feature `fidelity` |
| entanglement matrix `E[i][j]` | node-pair feature `has_ent` |
| request `(src, dst)` | per-node `has_request`, `dst_onehot` |

Implementation lives in `src/quantum/algorithm/RELiQ_Adapter.py` as helper methods:
- `_build_obs(topo, request_list) -> (obs, adj)` — builds the `(num_nodes, feat_dim)` tensor and `(num_nodes, num_nodes)` adjacency.
- `_apply_action(topo, request, action) -> next_hop` — consumes one link's entanglement, applies Werner swap if needed, decrements qubit budgets.

Decay + lifetime tick happens inside `topo`'s existing loop; the adapter does not re-implement it.

---

## 8. Phase 6 — `Run.py` Integration

### 8.1 Imports (top of `Run.py`)
Replace:
```python
from DQRL_dist1 import QuRA_DQRL_DIST
from DQRL_dist2 import QuRA_Flock_DIST
from DQRL_dist3 import QuRA_Guard_DIST
from DQRL_dist4 import QuRA_Hive_DIST
```
With:
```python
from DQRL_dist1_pt import QuRA_Seq_DIST as QuRA_DQRL_DIST
from DQRL_dist2_pt import QuRA_Flock_DIST
from DQRL_dist3_pt import QuRA_Guard_DIST
from DQRL_dist4_pt import QuRA_Hive_DIST
from RELiQ_Adapter   import RELiQ_Adapter
```

### 8.2 Algorithm list (line ~344)
Append RELiQ alongside existing QuRA variants:
```python
algorithms.append(QuRA_DQRL_DIST(copy.deepcopy(topo), name='QuRA_Seq_DIST'))
algorithms.append(QuRA_Flock_DIST(copy.deepcopy(topo), name='QuRA_Flock_DIST'))
algorithms.append(QuRA_Guard_DIST(copy.deepcopy(topo), name='QuRA_Guard_DIST'))
algorithms.append(QuRA_Hive_DIST(copy.deepcopy(topo), name='QuRA_Hive_DIST'))
algorithms.append(RELiQ_Adapter(copy.deepcopy(topo),  name='RELiQ'))
```
Existing baselines (REPS, SEERCACHE3_3, Q-CAST, …) stay as-is.

### 8.3 Environment variables
Unchanged: `TTIME`, `STEP`, `TIMES`, `TRAIN_LOAD`, `REQ_LOADS`, `INFERENCE_MODE`.

### 8.4 CSV log schema
Unchanged: `/tmp/qrouting_logs/progress_{algo.name}_req{count}.csv` with `timeslot, successful_requests, reward`. RELiQ rows written by the adapter via the base-class helper.

---

## 9. Phase 7 — Training & Evaluation Flow

### 9.1 Training (one-off, slow on CPU)
```bash
# 1. Train QuRA variants via distributed workers
export INFERENCE_MODE=0
bash src/rl/deploy_full.sh
python src/quantum/algorithm/Run.py     # writes model_* into Redis
python src/rl/save_trained_model.py     # dumps .pt snapshots

# 2. Train RELiQ (independent)
python -m src.reliq.train --device=cpu --total-steps=500000
```

### 9.2 Evaluation (apples-to-apples)
```bash
export INFERENCE_MODE=1
export TTIME=10000 STEP=1000 TIMES=1
export REQ_LOADS="5,10,25,50,75,100"
bash src/rl/deploy_full.sh              # workers in inference mode
python src/quantum/algorithm/Run.py
```
Every algorithm in the list (including RELiQ) runs against the same Waxman topology seeds and the same request stream. Logs land in `/tmp/qrouting_logs/`.

---

## 10. Validation Checkpoints (Summary)

| Phase | Pass criterion |
|---|---|
| 1. Models | `QRoutingGATFlat(x).shape == (B, 100)`; `QMixer(qs, s).shape == (B,)`; weight-parity vs TF within 1e-4 (optional) |
| 2. Infra | Redis round-trip of `state_dict`; `/health`, `/predict`, `/train` return 200; FedAvg produces mean weights |
| 3. Variants | All four `progress_QuRA_*_DIST_req5.csv` files populated under smoke config |
| 4. RELiQ | `src/reliq/train.py` trains for 50k steps without error; adapter instantiates on a 20-node topo |
| 5. Bridge | `_build_obs` output shapes match RELiQ's expected `(num_nodes, feat_dim)` / adjacency |
| 6. Run.py | Full 5-algorithm run writes 5 CSVs per `REQ_LOAD` without crashes |
| 7. End-to-end | Inference mode produces monotonic `successful_requests` curves for each algorithm across `REQ_LOADS` |

---

## 11. Open Risks & Mitigations

- **CPU training time for RELiQ at 15M steps is infeasible.** Cap at 500k–1M steps for the first run; document in results. The algorithm comparison is still valid as long as every baseline uses the same compute budget envelope.
- **QMIX parity**: TF `MultiHeadAttention` uses a slightly different init than PyTorch's `nn.MultiheadAttention`. Retraining from scratch absorbs the difference — do not try to port trained TF weights.
- **Waxman determinism**: `Topo.generate` uses Python's `random`. Seed with a fixed value in `Run.py` before generating the topology so all algorithms see the identical graph.
- **Redis key collisions** between QuRA and a hypothetical future RELiQ worker: RELiQ does not touch Redis — keep it that way.
- **RELiQ action space mismatch**: if `Discrete(3)` is strictly smaller than the max node degree in the Waxman graph, the adapter must mask unavailable neighbors. Masking logic lives in `_build_obs` via `action_mask`.

---

## 12. Deliverables (end of execution)

- Branch `qura-pytorch-reliq` with all changes.
- Five algorithms runnable from the unmodified `Run.py` invocation.
- CSV logs for all five under `REQ_LOADS=5,10,25,50,75,100`.
- `runs_quantum/RELiQ_QuRAPhysics/model.pt` checkpoint.
- Redis snapshot of QuRA variant weights.
- Updated `requirements.txt` (PyTorch-only).

No changes to `main` or `dqrl_dist_claude`.
