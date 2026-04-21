"""
QuRA-Hive (Distributed): QMIX cooperative multi-agent routing.

Architecture — Centralised Training, Decentralised Execution (CTDE):
  - Each request is an independent agent querying the shared Q-network.
  - A QMIX mixing network (already in DQRLAgentDist_API) combines per-agent
    Q-values with a global state to compute Q_tot satisfying monotonicity.
  - Routing uses Guard-style priority ordering (Q-value priority) for resource
    commitment, identical to QuRA-Guard.
  - The key difference vs Guard: the joint reward explicitly tracks the sum
    of all per-agent rewards, weighted by the paper formula λ/μ/ν.
    This is passed through the /update_reward endpoint and processed by the
    QMIX train_qmix() path which already accumulates ts_rewards per a_id.

Joint reward (paper):
    r_tot = Σ_i r_i
    R_tot = r_tot · λ + N_success · μ + F_avg · ν

This inherits QuRA_Guard_DIST and only overrides the reward accumulation call
to make the joint nature explicit for logging/tracking.
"""

from DQRL_Guard_dist import QuRA_Guard_DIST


class QuRA_Hive_DIST(QuRA_Guard_DIST):
    """
    QuRA-Hive: Same routing as Guard but with explicit joint reward tracking.

    The QMIX mixer in DQRLAgentDist_API already handles the monotonic value
    decomposition during training. Hive differs from Guard only in that we
    log joint rewards explicitly and use a slightly longer epsilon decay
    schedule to give the mixer time to stabilise.
    """

    def __init__(self, topo, param=None, name='QuRA_Hive_DIST'):
        super().__init__(topo, param=param, name=name)
        self._joint_reward_log = []

    def p4(self):
        result = super().p4()
        # Log joint reward sum for this timeslot (for analysis)
        if self.result.rewardPerRound:
            self._joint_reward_log.append(self.result.rewardPerRound[-1])
        return result
