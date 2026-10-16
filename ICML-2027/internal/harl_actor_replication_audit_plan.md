# Pinned MAPPO actor replication audit

Operational qualification only; no environment, return, or scientific effect
selection. Freeze source before execution. Preserve output and do not retry a
payload failure or alter tolerance after observing outcomes.

- HARL: clean b1af98b0dbab72a2eee9d160751cd09aedbb8ce2.
- CPU, one Torch thread; existing ust2 runtime and existing pure-Python PyYAML.
- Seeds: 63000001, 63000002.
- Action spaces: Discrete(5), Box(3); activations tanh and relu.
- Full q={1,2,4,8}, exact repeated 200-row buffer, identical initial model and
  Adam state. Exactly five full-batch actor updates per payload.
- Fixed normalized synthetic advantages, old action probabilities, and masks.
- 32 payloads total. Verify finite losses and exact-distribution policy KL;
  maximum parameter difference from q=1 must be at most 1e-5.
- Count processed model rows and optimizer steps explicitly. Synthetic row
  processing is not represented as environment transitions or team learning.

The purpose is to eliminate a false response mechanism before any new costly
performance study. Replication invariance is an operational result, not evidence
that q has no effect on genuine independent or evolving-policy trajectories.
