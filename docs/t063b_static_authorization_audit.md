# T-063B static authorization audit

## Decision

The outcome-free T-063B design passes static authorization.  This audit does
not inspect T-063A efficacy endpoints to choose a gate; it only verifies the
frozen new specification, provenance, implementation, and tests.  A separate
commit changes the authorization flag to permit the primary CPU run.

## Checks

- T-063B configuration hash validates.
- T-063A base preregistration file hash and source pilot hash validate.
- Seed registry is contiguous, all-new, and disjoint from T-060A, T-061A,
  and T-063A.
- Aggregate collision gate is fixed to the exact one-sided binomial upper
  confidence rule; blockwise maximum is descriptive only.
- No T-063B result directory or scientific endpoint exists.
- Full nonlinear suite: `160 passed, 7 skipped`.
- No GPU, HPC4, `/project`, or formal artifact was used for this audit.

## Fixed hashes

| file | SHA-256 |
|---|---|
| `docs/t063b_reward_free_controller_formal_preregistration.json` | `2900c15b395d220e2a4e35b908468cca8326b05fdfbf20dd9b5b073114395f53` |
| `docs/t063b_reward_free_controller_formal_preregistration.md` | `cf99d9e6da40e39322b365ac380086666753b2eea84ec01ae08a0dd362f046a1` |
| `experiments/nonlinear_markov_td/run_t063b_reward_free_controller_formal.py` | `6922878512ec439f81c7631803e67093519bc7a4ffb40834dc8d4f85a89e4ac8` |
| `experiments/nonlinear_markov_td/analyze_t063b_reward_free_controller_formal.py` | `3bfaf71aecee0b298509c394b791a44ce0b5a504f855e25a3201858670dcbaf9` |
| `experiments/nonlinear_markov_td/test_t063b_reward_free_controller_formal.py` | `05342c395fb49366bee7358ce540fac16daca1d74057e2ed3fa493526c316969` |
| `docs/t063b_test_manifest.json` | `4078cd5a298492c70028e872652bf99f3b3314f226d5a1209dd5736ed47ce647` |

The primary and clean reproduction are authorized only after the separate
authorization commit is pushed.  Any failed T-063B scientific gate stops the
new claim without changing T-063A.
