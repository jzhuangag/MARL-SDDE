# TSP-V3-CERT-SENS-001 operational amendment 1

The first development invocation stopped before creating an output directory
because the local NumPy version does not expose `numpy.trapezoid`.

The runner now selects `numpy.trapezoid` when available and otherwise uses its
historical equivalent `numpy.trapz`.  This changes neither the trapezoidal AUC
definition nor any model, task, policy, seed, budget, threshold, or gate.

No scientific result was written or inspected before this amendment.  The
failed invocation left
`TSP-V3/results/certificate_sensitivity_development_20261003` absent.
