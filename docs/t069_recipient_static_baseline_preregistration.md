# T-069 preregistration: recipient-specific static baseline audit

T-068A shows a positive exact ceiling for recipient-wise dynamic safe mixing,
but it fails its preregistered simple phase gates.  Its strong fixed comparator
uses one scalar mixing strength for all recipients.  T-069 therefore asks the
claim-critical question that T-068A cannot answer: does temporal adaptation
retain material value after every recipient receives its own optimally selected
but time-invariant mixing strength?

The audit reuses exactly the 648 frozen T-068A cells and enumerates all
`6^4=1,296` recipient-specific vectors.  Each vector is held fixed at all six
communication checkpoints and pays no sensing cost.  The dynamic safe oracle
retains its twelve charged probe transitions.  This deliberately favors the
static comparator.

Exact linearity permits precomputing two deterministic mean responses and two
covariance responses per vector and delay.  The 839,808 risks are then evaluated
without sampling or changing the T-068A law.  The common-alpha subset must
reproduce the old cellwise fixed comparator before any new comparison is read.

Q1--Q10 are mandatory.  Failure of Q4 or Q5 means the apparent T-068A dynamic
value is adequately explained by static recipient personalization and stops a
sampled safe-mixing controller.  Passing cannot rescue T-068A; it only permits a
separate theorem-aligned controller preregistration.

Frozen SHA-256 values before execution are:

- configuration: `A33AB8E7D3FED00D4033AB909AE844A22CE03F6522E02CDFA3D12F601F978981`;
- runner: `22048C53560A0E2C18D101949B6102204C58117BF8BE99449190BC8EC90418F4`;
- batched exact core: `E425E0EB972FC01AB43C9DB6F8229D18161BEA9219492B6FB64994C80F260250`.
