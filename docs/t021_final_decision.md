# T-021 final decision

## Decision: retain the theory-first fixed-q mainline

T-020 permanently ends EXP-017B and the present online-adaptive controller
line. It does **not** end the paper. The retained ICML-facing question is why
multi-agent Markov learning stops achieving linear speedup under cross-agent
correlation, delay, mixing, and dual resource budgets.

The current theorem inventory is sufficient to state a rigorous scoped
mainline: decorrelated delayed affine Markov TD, exact Gaussian
correlation-limited speedup, a dual-budget participation threshold, the
unrestricted unknown-mixing impossibility boundary, and a nonlinear
fixed-parameter variance identity. No open theorem is silently promoted.

The existing nonlinear evidence is only descriptive design evidence. Its four
fixed-q optima and direction counts justify a new fixed-q validation design,
but its two pilot seeds cannot support a paper claim. Formal nonlinear evidence
must measure gradient variance/covariance directly and use seed-level clustered
inference.

## Authorization

- T-021 CPU static audit: complete.
- EXP-017B: permanently stopped.
- New CPU implementation/preregistration stage: permitted under a new ID.
- GPU/HPC4 pilot or formal run: not yet authorized.
- SDDE: retained as interpretation; no SDDE-to-discrete claim.
- Online adaptive controller: excluded from title, abstract, and main theorem
  narrative.

The paper remains a plausible ICML 2027 project if the prospective fixed-q
nonlinear mechanism validation succeeds and the final manuscript stays within
these claim boundaries. It is not yet an ICML-ready empirical package.

