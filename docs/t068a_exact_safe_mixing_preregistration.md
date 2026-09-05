# T-068A preregistration: exact shadow-anchored mixing phase scan

T-068A is the first falsification gate for the safe personalized-collaboration
mainline.  It propagates exact first and second moments of four scalar affine
learners.  The agents have distinct fixed points, correlated local noise, and
stale model exchange.  A single stationary law is used throughout each cell;
any collaborate-then-personalize transition is caused by shrinking learning
error rather than an exogenous regime switch.

Every agent maintains a collaborative model and a same-noise local shadow.
The safe oracle uses true moments only to bound the maximum mechanism value;
it is not an implementable controller or paper-facing result.  At six frozen
decision blocks it selects recipient-specific mixing from the registered
simplex grid, subject to exact risk no-harm relative to the charged shadow.
Its twelve probe transitions are removed from learning.  The no-probe local
and fixed-mixing comparators use all 240 transitions for learning.

The strong fixed comparator is selected separately within every cell.  The
two-phase oracle is also selected within every cell from all 36 frozen
early/late pairs.  This makes the main phase-value gates intentionally hard:
dynamic value cannot be manufactured by comparing against a poor global
constant.

The design contains 648 cells and 27,864 exact policy rows.  P1--P12 are
mandatory.  Failure of any scientific gate stops a sampled T-068 pilot; no
threshold or cell may be changed after the independent preregistration commit.
The scan is local-CPU only and cannot authorize GPU or HPC4 work.

Frozen SHA-256 values before execution are:

- configuration: `C33914498EC711DB657D0668C2EE321C633C5108F0115461D8B9933772A88A28`;
- runner: `32C10FAA7173E8E4745CE219A4F2CDDCF3458E910F93D92E8E5978039AD6E330`;
- exact-moment core: `DE2C4C83DD7DE5A8C6B36C30669E07A5AA2534E7AF885290D5E8A9C04D63A41F`.
