# T-024 final decision after EXP-018B

## Decision: retain the mechanism theorem, do not revive the failed controller

EXP-018B supplies formal and exactly reproducible evidence for the
correlation--participation variance factor in frozen nonlinear TD gradients.
This closes the narrow experimental mechanism gap exposed by EXP-018A without
changing the failed pilot's gates.

It does not reverse EXP-017A/T-020. The old nonlinear adaptive controller and
EXP-017B remain stopped because their strong-baseline adaptation-value ceiling
is too small. A mechanism can be correct while the selected benchmark offers
too little decision value for online adaptation.

## Consequence for the ICML main line

The defensible main line is now:

1. dependence and participation jointly determine effective gradient noise;
2. this factor is exact in the registered common-factor model and formally
   calibrated in frozen nonlinear TD gradients;
3. delay, mixing, and dual budgets determine whether information about that
   factor can be acquired and acted upon;
4. adaptation has an identification/opportunity cost and may be impossible or
   valueless outside separated, mixing-certified regimes.

The main line is not yet an ICML-complete empirical package. It still needs a
new outcome-free nonlinear benchmark whose strong fixed-q envelope leaves at
least the already frozen 5% aggregate and 60% directional adaptation value,
plus a proved observable surrogate/no-harm certificate. Until those static
conditions hold, another GPU controller pilot would be scientifically
premature.

## Next authorized work

- integrate EXP-018B into the theorem/claim dependency ledger;
- write the fixed-gradient covariance lemma and its assumptions in paper-ready
  form, explicitly separating it from online convergence;
- continue CPU-only outcome-free benchmark/theory screening;
- request GPU/HPC4 only after a new benchmark passes every T-020 static
  authorization condition in a separate preregistration commit.

No new nonlinear GPU experiment is authorized by T-024.

