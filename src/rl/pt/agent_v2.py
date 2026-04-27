"""
DQRLAgentV2 — integration layer for QuRA-v2.

Wraps QuantumGAT + EdgeQNet + QMixerV2.  All three modules train jointly.

Gradient-flow assertion (checked during __init__):
  Run one dummy train_step; assert at least one parameter changed.
  Failure means a module is wrapped in @torch.no_grad that shouldn't be.
"""
