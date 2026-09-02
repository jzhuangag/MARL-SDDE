## Material Passport

- Artifact type: outcome-free integration validation.
- Research stage: standard-MARL bridge, local G0.
- Data class: software traces and resource accounting only.
- Scientific outcome access: none; no reward, return, win rate or method
  comparison was emitted.
- Verification status: byte-exact reproduced.
- Next gate: HPC4 GPU task-shape/resource preflight, then a separate immutable
  pilot preregistration.

# Two Clocks Layer-0 G0 validation

Date: 2026-09-02.

## Decision

**Pass the local outcome-free G0 substage.**  The new runtime exercises the
actual Two Clocks common-step interface rather than the historical
strategic-debt scaler.  It closes local ownership, version-path, charging and
teardown contracts.  It does not authorize a scientific GPU pilot or formal
seeds.

## Implementation boundary

The framework-neutral `TwoClocksPacketLedger` never accepts a reward, return,
gradient or packet payload.  Service completion time and fixed work are
declared at launch.  A thin external-checkout runner attaches the HARL packet
payload while preserving the ledger boundary.

The runner uses the pinned, clean HARL checkout
`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2` and three distinct
`StochasticPolicy` actor blocks on continuous `simple_spread_v2`.  The
finite-policy Lyapunov--Krasovskii formula supplies a common step interface,
but its application to the unconstrained neural actor is explicitly labelled
`empirical_interface_only`.

No upstream HARL source was modified or vendored.  The run executed in the
previously isolated ignored Python 3.9 environment.

## Commands

Targeted contract tests:

```text
.venv/Scripts/python.exe -m pytest -q \
  experiments/clocked_async_mpg/test_two_clocks_packet_runtime.py \
  experiments/clocked_async_mpg/test_run_two_clocks_layer0_g0.py \
  experiments/clocked_async_mpg/test_finite_time_drift.py
```

Primary and reproduction:

```text
tmp/harl-smoke-py39/Scripts/python.exe -m \
  experiments.clocked_async_mpg.run_two_clocks_layer0_g0 \
  --harl-root tmp/HARL \
  --output tmp/two_clocks_layer0_g0_primary/summary.json

tmp/harl-smoke-py39/Scripts/python.exe -m \
  experiments.clocked_async_mpg.run_two_clocks_layer0_g0 \
  --harl-root tmp/HARL \
  --output tmp/two_clocks_layer0_g0_reproduction/summary.json
```

## Results

- Targeted tests before the final smoke: `32 passed in 0.42 s`.
- Packets launched, completed and applied: `12/12/12`.
- Final per-owner versions: `(5,4,3)`; their sum equals the applied updates.
- Completed work: 96 environment steps, 288 actor transitions and 23,976
  declared optimizer units.
- Cancelled work: zero; the runtime separately tests nonzero cancelled-work
  charging and upper-bound rejection.
- Maximum registered event delay: at most eight.
- Every common-step condition is at most one; the largest is one up to floating
  precision.
- Owner parameter self-freshness error: at most `1e-10` for every packet.
- At least one packet saw a positive teammate-version increment, so strategic
  staleness was actually exercised.
- Every diagnostic was finite, every environment instance closed, and teardown
  left no active or unapplied packet.
- Primary and reproduction summaries are byte-identical.

PettingZoo emitted only its known deprecation warnings for dictionary-style
space access.  There was no exception, leaked worker process or scientific
outcome field.

The final `clocked_async_mpg` package regression passed `378/378` tests in
42.41 seconds.  The complete repository experiment regression passed 1,302
tests with seven skips in 166.00 seconds.

## SHA-256 provenance

| Artifact | SHA-256 |
|---|---|
| packet runtime | `8AC8BC014304714EAF1E0E96277A54EDFA435AF29669138A4EC1C042E9CC51A0` |
| G0 runner | `8C1E0BB0197DED56FCC2E444E777024128407AF0B46AB674B4315E9D3E108106` |
| Lyapunov common-step implementation | `80C9A3B1A6563D4278A1A2C46F8F6BD1AAD9B7D79413CE44D23426CA6259F5C3` |
| Gaussian-KL helper overlay | `03F5B3D4E48E8F53641E5C506A03E577E47F6D282628458C052B115F0610B2EE` |
| shared HARL trajectory helpers | `A0A3B3C6D127AA8C4CCF51006E9639079504892CC3B4FBFE6AC4DAB212881D23` |
| runtime tests | `212CA7EBF36F8E53458C0C395BABFA36F7A4F6470A8512F5D6FBA96773D122E2` |
| runner tests | `7D10A839D0F63019F33FCF30E93AE7DB85FCAE5D76714080594C87223AACE291` |
| primary summary | `295EB137547F2D1E35C6E7F9B520DA4F8F7F24228E575E70B38E05E385211135` |
| reproduction summary | `295EB137547F2D1E35C6E7F9B520DA4F8F7F24228E575E70B38E05E385211135` |
| upstream HAA2C configuration | `30F87C99A7ED2A2DDC6F898CF7E23D4DC34CABCEE6E9E579FF3DBADBE7267E1E` |

Raw summaries remain ignored under `tmp/`; they contain only deterministic G0
traces and accounting.  No existing scientific artifact or frozen experiment
was modified.

## Remaining mandatory boundary

This local pass does not measure concurrent GPU service, utilization, memory,
task installation or process teardown under Slurm.  The HPC4-connected agent
must perform that outcome-free resource preflight on the frozen task shapes.
Only after it passes may a new commit preregister scientific pilot seeds,
budgets, comparators, numerical gates and artifact rules.  No pilot outcome may
be accessed before that commit.
