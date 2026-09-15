# T-027 outcome-free FrozenLake static scan preregistration

## Objective

Determine whether standard slippery FrozenLake 8x8 can supply a rigorous,
CPU-only fixed-participation phase-diagram calibration before any nonlinear
learning pilot. The scan uses the exact public transition table and generates
zero scientific trajectories.

## Frozen Markov construction

- Gymnasium 1.0.0 `FrozenLake-v1`, `map_name=8x8`, `is_slippery=True`;
- uniform random fixed policy over four actions;
- terminal observations reset to the standard start state on the next
  transition, matching episodic training with immediate reset;
- stationary distribution solved from the exact 64-state transition matrix;
- mixing stride is the first exact matrix power whose maximum row total
  variation distance to stationarity is at most 0.05.

The construction is outcome-free. No environment step, reward outcome,
gradient, or learned parameter is generated.

## Frozen resource/value scan

- public 3,169-parameter MLP and 65,536-byte server overhead;
- q in `{1,4,16,32}` and rho in `{0,0.1,0.5,0.9}`;
- target horizons 512 and 2,048;
- message- and environment-binding budget rays;
- delay fractions 0, 0.05, and 0.20;
- exact mechanism risk proxy
  `(rho + (1-rho)/q) / usable_horizon`;
- strong fallback selected independently for each horizon × budget ray by the
  geometric risk over all registered rho/delay cells;
- cellwise oracle is descriptive and used only as an optimistic static value
  ceiling.

The scan contains 192 q-arm rows and 48 oracle/fallback cells.

## Mandatory gates

1. S1: exact finite stochastic matrix and stationary residual at most 1e-10;
2. S2: TV 0.05 mixing stride exists and is at most 512;
3. S3: aggregate oracle improvement is at least 5%;
4. S4: strict oracle improvement occurs in at least 60% of all 48 cells;
5. S5: at least three distinct oracle q values occur;
6. S6: at least one message-binding cell has internal q in `{4,16}`;
7. S7: environment-binding oracle q is never below its matched message q;
8. S8: zero scientific trajectories and no outcome taint.

All eight gates are mandatory. Any failure stops FrozenLake as the proposed
nonlinear learning benchmark. Passing authorizes only a separate local-CPU
pilot preregistration; it does not authorize formal evidence, HPC4, or GPU.

## Frozen implementation

- runner: `experiments/nonlinear_markov_td/t027_frozenlake_static_scan.py`;
- tests: `experiments/nonlinear_markov_td/test_t027_frozenlake_static_scan.py`;
- output: ignored local `results/t027_static_scan_20260802/summary.json`;
- config SHA-256:
  `f9251599e0382309e5d08d115bf04def6feffb00bff2109de548543643269442`.
