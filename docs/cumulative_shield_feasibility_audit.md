# Cumulative shield feasibility: accounting is not information

## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theory/code validation
- Origin Date: 2026-08-31
- Verification Status: ANALYZED
- Version Label: audit_v1

Internal derivations only. No new efficacy experiment, manuscript claim of novelty, or revision of historical safety metrics.

## A cumulative guarantee that actually follows from an observable bound

For each recipient i, let s_(k,i) be the executed persistent local-shadow prediction at block k, and z_(k,i) the chosen post-mix prediction. Assume simultaneous target intervals [h_(k,i)-r_(k,i),h_(k,i)+r_(k,i)] cover all evaluated targets on event E. The earlier target-coverage audit supplies a sufficient construction only under its explicit stationary Gaussian noise, fixed-block, and public variance/mixing-bound assumptions.

Define true excess g_(k,i)=(z_(k,i)-theta_(k,i))^2-(s_(k,i)-theta_(k,i))^2 and its observable upper bound

    U_(k,i)=(z_(k,i)-h_(k,i))^2-(s_(k,i)-h_(k,i))^2
             +2*r_(k,i)*|z_(k,i)-s_(k,i)|.

Maintain spendable credit C_(0,i)=B_i>=0. Before committing a prediction, require U_(k,i)<=C_(k,i)+epsilon_(k,i), where epsilon_(k,i)>=0 is a predeclared allowance. Then update

    C_(k+1,i)=C_(k,i)+epsilon_(k,i)-U_(k,i).

This is a prospective acceptance rule, not the old T-081 debt update. The shadow action gives U=0, so feasibility is always available if the shadow can be executed. By induction C>=0. Telescoping and g<=U on E prove, simultaneously for every prefix t,

    sum_{k<t} g_(k,i) <= B_i+sum_{k<t} epsilon_(k,i)-C_(t,i)
                     <= B_i+sum_{k<t} epsilon_(k,i).

Negative certified excess earns credit that can fund a later positive excess. This bound concerns the block-end risk metric only, not unobserved within-block risks. It does not require block independence after simultaneous coverage has been established. It is a high-probability cumulative guarantee, not an unconditional expectation guarantee.

## Constant-cost acceptance on an interpolation segment

For proposal p and reference s, choose z=s+beta*(p-s), beta in [0,1]. Set v=p-s, c=v*(s-h)+r*|v| and allowance A=C+epsilon. Solve v^2 beta^2+2*c*beta<=A. For v!=0 the maximum allowed beta is min(1,root), where

    root = A/(sqrt(c^2+v^2*A)+c)       if c>0,
    root = (-c+sqrt(c^2+v^2*A))/v^2   if c<=0.

The first branch avoids cancellation. At v=0 select beta=1. At A=0 and c=0 use root=0. This is scalar algebra per recipient once p,h,r,s exist; it does not establish low complexity for constructing p or a valid confidence interval.

## Why cumulative credit does not solve cold start by itself

If B=0, every epsilon=0, and s lies in [h-r,h+r], then U(z)>0 for every z!=s. Therefore the only permitted action is shadow. It earns U=0 and no credit. If this interval containment persists, cumulative shielding remains at shadow indefinitely. This is not fixed by renaming the queue or allowing a larger candidate graph set.

Positive initial B or positive epsilon permits expenditure, but changes the safety contract. A linear total allowance T*epsilon is a linear cumulative-damage allowance, not vanishing average harm. B/T vanishes for fixed B, but a fixed exploration budget need not be enough to obtain meaningful adaptation. Neither can be chosen retrospectively to rescue formal results. Uncertified apparent gains must never be credited as though U<0.

## A public-model diagnostic for strict shielding availability

In the scalar affine local shadow with deterministic initial s0, gain eta and deterministic targets, write observation noise over N steps as xi with covariance v*R_N(lambda). At the end of block k, the shadow has noise coefficients w_j=eta*(1-eta)^(N-j), j=1,...,N. The current block mean has coefficients b_j=1/m on its last m indices and zero elsewhere. Thus D=s-h is Gaussian with

    mu_D = E[s]-theta_k,
    tau_D^2 = v*(w-b)^T R_N(lambda)*(w-b).

For tau_D>0 and deterministic interval radius r,

    Pr(|D|>r)=Phi((-r-mu_D)/tau_D)+1-Phi((r-mu_D)/tau_D).

For tau_D=0 the probability is the indicator |mu_D|>r. This is the exact marginal probability that *some unrestricted scalar action* has strictly negative robust excess. It is only an upper bound on feasible strict graph improvement: donor geometry and resource constraints may reduce it. It neither measures realized controller gain nor grants per-trajectory certification before observing h,s. Spatial covariance does not enter this individual shadow diagnostic; it matters for actual donor value.

This diagnostic uses only model parameters and target schedule, not formal endpoints. A future bounded analytic feasibility study can evaluate it before spending seeds. Arbitrary stationary short-block settings cannot be presumed to have useful activation probability simply because the target interval has valid coverage.

## Lyapunov status and next decision

The credit invariant above is a safety accounting lemma, not a deep new Lyapunov convergence result. A composite Lyapunov argument must still couple learning-error contraction, delayed collaboration and resource costs to an implementable action objective. The old Q*||w-e_i||^2 penalty has not been justified by this derivation. No claim of a completed main theorem is made.

Next: implement the public-model activation-probability diagnostic with deterministic covariance checks, and define a bounded outcome-free feasibility protocol. Keep strict per-step and cumulative-safety contracts distinct. Do not launch an efficacy pilot until a valid interface and nontrivial learning mechanism are established.

## Execution record

Starting checkout `a6e7e3a`, clean worktree, no Python processes observed. Command `.venv/Scripts/python.exe -m pytest experiments/dependence_delay_linear/test_cumulative_shield_audit.py -q`: **11 passed in 0.13s**. Tests verify credit arithmetic, nine scalar-root cases, and zero-credit interval obstruction. No empirical activation probability or effectiveness was computed. No formal data read for tuning, no frozen files changed, no GPU/HPC4 operations. No external bibliographic claim or manuscript deliverable is introduced.
