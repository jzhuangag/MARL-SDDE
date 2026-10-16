"""Corrected entry point for post-stop TSP-V2 HARL qualification and runs.

The original preregistered executable remains byte-for-byte unchanged as
historical provenance.  This entry point replaces only its deterministic HARL
evaluation bridge before delegating to the frozen scientific implementation.
"""

from __future__ import annotations

import run_marl_online_drift_queue as frozen_runner
from harl_online_runtime_bridge import deterministic_horizon_return


frozen_runner.deterministic_horizon_return = deterministic_horizon_return


if __name__ == "__main__":
    frozen_runner.main()
