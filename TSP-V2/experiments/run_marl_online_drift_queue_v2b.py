"""Post-QUAL-001 entry point with exact pinned-HARL mask semantics."""

from __future__ import annotations

import run_marl_online_drift_queue as frozen_runner
from harl_online_runtime_bridge_v2 import deterministic_horizon_return


frozen_runner.deterministic_horizon_return = deterministic_horizon_return


if __name__ == "__main__":
    frozen_runner.main()
