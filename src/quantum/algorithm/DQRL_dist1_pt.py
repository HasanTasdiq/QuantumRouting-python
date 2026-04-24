"""
QuRA-Seq (PyTorch): thin wrapper around DQRL_dist1.py that swaps out the
TF-dependent dist_agent_helper import for the TF-free pt_dist_helper.

The routing logic (shared memory, Redis, HTTP endpoint calls, conflict
resolution) is 100% inherited from QuRA_DQRL_DIST — no duplication.
The PyTorch predict server (src/rl/pt/server.py) runs behind the same
WORKER_PORTS so all HTTP calls land on PyTorch workers.
"""
import os
import sys

# Add rl/pt to path so pt_dist_helper is importable
_rl_pt = os.path.join(os.path.dirname(__file__), '..', '..', 'rl', 'pt')
if _rl_pt not in sys.path:
    sys.path.insert(0, _rl_pt)

# Replace dist_agent_helper in sys.modules BEFORE importing DQRL_dist1
# so that "from dist_agent_helper import ..." picks up the TF-free version.
if 'dist_agent_helper' not in sys.modules:
    import pt_dist_helper as _ptdh
    sys.modules['dist_agent_helper'] = _ptdh
elif not hasattr(sys.modules['dist_agent_helper'], '_is_pt'):
    # TF version already loaded — override with PT version
    import pt_dist_helper as _ptdh
    sys.modules['dist_agent_helper'] = _ptdh

from DQRL_dist1 import QuRA_DQRL_DIST  # noqa: re-export
