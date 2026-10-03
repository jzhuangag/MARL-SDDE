# T-063B Amendment 1 static audit

## Decision

Amendment 1 passes outcome-free static checks but is not yet authorized for
scientific execution.  It only changes numeric replay comparison to tolerate
CSV serialization at `1e-12` while preserving mandatory byte-exact artifacts.

Checks passed:

- amended configuration hash validates;
- base T-063A provenance remains pinned;
- targeted T-063B tests: `7 passed`;
- full nonlinear suite: `164 passed, 7 skipped`;
- no T-063B result directory or endpoint exists;
- no GPU, HPC4, or `/project` use.

Authorization remains false until a separate commit explicitly authorizes the
amended specification.

| file | SHA-256 |
|---|---|
| `docs/t063b_reward_free_controller_formal_preregistration.json` | `6b162497bf211b69499ecaa04e087f7bb8b7253d6d313951162a0aea64764b2d` |
| `docs/t063b_reward_free_controller_formal_preregistration.md` | `de86ec74ddadfa1fee2a1abd4d55b80e789a792636903d86bb0702f0a4d5ccb6` |
| `docs/t063b_amendment_1_replay_gate.md` | `2ea3aef5d636b45b171af59a9f9f8b1999677aa17bd1d5ecf21b3eacd6890363` |
| `experiments/nonlinear_markov_td/analyze_t063b_reward_free_controller_formal.py` | `b4ec5bb46360119c2a92c28d499b50294cca7e78be382879210fd167fc257d1c` |
| `experiments/nonlinear_markov_td/test_t063b_reward_free_controller_formal.py` | `23da424b217b9b2cb28b62cc884e329c3ba501bf5e3cc029b568d1b443974c2b` |
| `docs/t063b_test_manifest.json` | `c64be305b3bbdf30f7ae55c1afad17840e2401ed70cbe90b6fcd4b73e10692ab` |
