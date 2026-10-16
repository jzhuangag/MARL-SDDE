# T-078 delay-extremes persistent-certificate calibration

T-078 is a new, reduced calibration rather than a continuation of timed-out
T-074/T-076/T-077. It retains all three schedules, all target scales, both
initial conditions, both noise levels, both spatial and temporal correlations,
and selects only delay 0 and the maximum registered delay 3. This produces 288
cells and 9,216 endpoints. Selection follows the delay-stability question and
does not inspect outcomes.

The persistent certificate, continuous QP, old design seeds, strong-static
comparison, resource accounting, and safety constants remain unchanged.
R1--R14 freeze scientific, stratified-safety, certificate-direction,
complexity, nine-minute runtime, reproducibility, and repository-test gates.
Passing can authorize only a separate theory audit; these reused seeds remain
tainted design evidence.
