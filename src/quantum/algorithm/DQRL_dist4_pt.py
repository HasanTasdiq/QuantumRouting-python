"""
QuRA-Hive (PyTorch): thin wrapper — see DQRL_dist1_pt.py for rationale.
"""
import os, sys

_rl_pt = os.path.join(os.path.dirname(__file__), '..', '..', 'rl', 'pt')
if _rl_pt not in sys.path:
    sys.path.insert(0, _rl_pt)

if 'dist_agent_helper' not in sys.modules or not hasattr(
        sys.modules['dist_agent_helper'], '_is_pt'):
    import pt_dist_helper as _ptdh
    sys.modules['dist_agent_helper'] = _ptdh

from DQRL_Hive_dist import QuRA_Hive_DIST  # noqa: re-export
