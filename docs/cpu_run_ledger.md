# Local CPU run ledger

All timestamps use Asia/Shanghai. Pilot seeds are permanently excluded from
formal evidence. Large generated CSV files are not committed.

| Timestamp | Commit/diff | Command | Environment | Output | Status | Hashes |
|---|---|---|---|---|---|---|
| 2026-07-31 16:09 | preregistration diff later frozen as `f410d9c2ccb9517359c2edf487a9b69659d7bd37` | `python -m pytest -q experiments/dependence_delay_linear/test_adaptation_cost_pilot.py` | local Anaconda base, Python 3.12.7 | stdout | completed | 16 passed |
| 2026-07-31 16:13 | same preregistration diff | `C:\Users\jzhuangag\AppData\Local\anaconda3\envs\ust2\python.exe -m pytest -q experiments` | local `ust2`, Python 3.11.11, torch 2.6.0+cpu; locally added pytest 9.1.1 and numba 0.66.0 | stdout | completed | 133 passed in 9.18 s |
| 2026-07-31 16:14--16:27 | `f410d9c` plus pre-evidence mathematically equivalent complexity corrections | two attempted EXP-015A commands | local `ust2` CPU | none | stopped before output creation | not evidence; repeated eigendecomposition and explicit large lag vector replaced |
| 2026-07-31 16:28 | same registered model and corrected implementation | `...\ust2\python.exe experiments\dependence_delay_linear\run_adaptation_cost_pilot.py --output-dir experiments\dependence_delay_linear\results\exp015a_pilot_20260731` | local `ust2` CPU | `experiments/dependence_delay_linear/results/exp015a_pilot_20260731` | completed in 10.70 s; 55,296 rows; 7/8 gates, honest pilot failure | metrics `62284DAF...B12A`; summary `A99D811E...5976` |
| 2026-07-31 16:31 | unchanged implementation and seeds | same command with `tmp/exp015a_reproduction` | local `ust2` CPU | ignored reproduction directory | completed; metrics and summary byte-identical | same two hashes |
| 2026-07-31 16:38 | final optimized implementation plus validation docs | `...\ust2\python.exe -m pytest -q experiments` | local `ust2` CPU | `tmp/exp015a_full_pytest.txt` | completed; 134 passed in 5.87 s | stdout `884B35F9...1475` |
