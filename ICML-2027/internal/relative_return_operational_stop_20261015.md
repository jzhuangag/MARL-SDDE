# Original estimator qualification: operational stop and A1 amendment

The run registered at `a9c9916` stopped after saving context 0 and before
saving context 1. No return means or scientific scores from either context
were inspected. The original root remains unchanged and is ineligible for
the A1 scientific decision.

The identity control at stage 16 raised `ValueError: invalid probability
vector`. For equal normalized categorical vectors, summing their coordinate
minima can round to `1-epsilon`. The original code interprets this as a
positive disagreement mass although both residual vectors are exactly zero.
That is an implementation defect, not a failed scientific gate.

A1 retains every original source file. A separate entry point supplies an
exact-equality overlap branch (`c=1` only when the two normalized vectors are
exactly equal). No epsilon cutoff suppresses a real policy difference.
All scientific seeds are replaced with 75101001 and 75101002. Candidate
policies, methods, sample counts, gates, and analysis formulas stay unchanged.
The amended code is locked and committed before new scientific execution.

This is not an outcome retry. The stopped partial outcomes remain unopened.
The operational smoke seed remains ineligible and separate from all old/new
scientific seeds. Additional regression tests cover 2,000 equal categorical
laws, a genuine 1e-10 difference, and a 25-step exact-identity trajectory.
