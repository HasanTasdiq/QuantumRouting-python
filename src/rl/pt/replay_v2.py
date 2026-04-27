"""
Replay buffers for QuRA-v2.

NStepPERBuffer  — n-step returns + proportional Prioritized Experience Replay.
                  Used by Seq / Flock / Guard (single-agent DQN).

QMixEpisodicBuffer — collects full routing episodes (one per request-group
                     per timeslot) for QMIX joint training.
                     Uses ragged batches + mask; no MAX_REQUESTS padding.
"""
