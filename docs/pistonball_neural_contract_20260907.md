## Material Passport

- Origin Skills: academic-research-suite; publishable-academic-writing
- Origin Mode: experiment-interface audit
- Origin Date: 2026-09-07
- Verification Status: VERIFIED STRUCTURAL RESULT; NOT A LEARNING RESULT
- Version Label: pistonball_neural_contract_v1

# Pistonball neural policy-cache contract

## Decision

The nonlinear implementation interface passes its outcome-free CPU contract.
This closes the structural gap between the signed Lyapunov rule and a standard
20-agent Pistonball actor--critic computation.  It does **not** show a return
gain and does not by itself authorize a formal benchmark.

The next justified step is to freeze the end-to-end asynchronous training
contract and preregister a small GPU pilot.  The pilot must retain strong
matched-resource baselines and may fail; no random-network quantity below is
used as efficacy evidence.

## What was implemented

For owner `i`, let

\[
 A_i(\chi_i)=
 \left\langle
 g_i^0,
 \nabla_{\theta_i}\ell_i(\theta_i,\chi_{-i};\phi)
 \right\rangle,
 \qquad
 g_i^0=\nabla_{\theta_i}\ell_i(\theta;\phi),
\]

where `phi` is a centralized critic and `chi` is the policy-version cache used
by the owner's rollout worker.  One reverse differentiation of this scalar
with respect to all eligible cached donor blocks returns

\[
 v_{j\to i}=\nabla_{\chi_{j\to i}}A_i(\chi_i),
 \qquad j\in\mathcal C_i.
\]

The refresh score uses the first-order lower alignment

\[
 \underline A_i(j)=A_i(\chi_i)+
 \langle v_{j\to i},\theta_j-\chi_{j\to i}\rangle-r_{j\to i},
\]

inside the registered drift-plus-queue index.  All donor VJPs are produced by
one cross-policy reverse call; candidate selection is a null-plus-one-edge
scan of size `1 + |C_i|`.  There is no candidate-specific environment rollout.

The CPU contract instantiates 20 distinct small actors, 20 worker-cache
copies, and one centralized critic on an actual Pistonball reset.  The
launch-state predictive tube supplies the eligible donors.  The selected
refresh, if any, copies only that donor's policy into its cache; the returned
gradient changes only the owner policy block.

## Outcome-free validation

Two isolated executions used seeds 73001 and 73002 on Python 3.11.13 and
PyTorch 2.6.0+cpu.  Their complete `summary.json` files were byte-identical:

```text
SHA-256 6CF52D6C9672AFDA124DB186D4B34E7665BB800AF67D8E866160B28E99935BE5
```

All eight gates passed:

1. exactly 20 distinct actor parameter blocks;
2. candidate count exactly `1 + eligible degree`;
3. one cross-policy VJP for all donors;
4. only the owner current-policy block changes at receipt;
5. only the selected donor cache changes, or no cache under the null action;
6. optional communication equals zero under null or exactly one actor payload;
7. all computed scores are finite;
8. launch decision reads no reward, termination, post-launch state, or
   environment step.

The median eligible degree was 4 rather than the 19 donors of a complete
graph.  Both random-network smokes selected the null action.  This is a valid
structural branch and is not interpreted as positive or negative learning
evidence.

Focused tests:

```text
8 passed in 39.23 s
```

The four algebraic tests separately exercise a positive edge selection, null
selection under a high queue price, rejection under a Taylor remainder, and
exact tensor-payload byte counting.

## Provenance

```text
neural_signed_cache.py
  ECB968674916278F5EA7EA0309666602C4A296C123EBC7DDA0ED63C0FF8047A0
pistonball_neural_contract.py
  DD91CF3D9CFDE9C90DB12A1A9EAEA4C745877C3E80F17F9712416D1AAFA466C7
run_pistonball_neural_contract.py
  F9D5A52D229E16D863478ED9FBD96D3516AC97905AC1F964A00A387479338F9E
test_neural_signed_cache.py
  1AEAB821611158C7360E8A6C6791468473B8FF73FB975B2A256848931C777143
test_pistonball_neural_contract.py
  E1B5A8331A87284CCBDB2902383C8B009B47F06E33BD82FEF97BC7408880D1D0
```

Raw primary and reproduction summaries remain ignored below
`tmp/policy_dependency_sync/`.  No GPU, HPC4, scientific trajectory, reward,
or return was used.

## Remaining theorem and benchmark obligations

This smoke intentionally does not certify the declared neural Taylor
coefficient.  The paper-level theorem must state a mixed-gradient regularity
assumption and include critic/VJP estimation error in the simultaneous score
radius.  A practical pilot must estimate this uncertainty from completed
training data; it cannot label the placeholder coefficient as a guarantee.

Before a GPU pilot, the frozen training contract must also specify:

- the owner-only actor update and centralized-critic update;
- launch and receipt times, in-flight packet ownership, and cache version path;
- the policy-byte and actor-transition charges;
- a replay-only, predictable score estimator;
- no-refresh, complete-refresh, age/mismatch, physical-graph, and strongest
  fixed-rate baselines under the same budgets;
- nontrivial return/sample-efficiency and budget gates with new pilot seeds.

Thus the result closes an implementation-interface question, not the central
empirical claim.
