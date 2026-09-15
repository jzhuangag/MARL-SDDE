## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-free cached-behavior validation
- Origin Date: 2026-09-07
- Verification Status: 5/5 GATES PASS; BYTE-EXACT REPRODUCTION
- Version Label: pursuit_cached_rollout_qualification_v1_validation

# Pursuit owner-cache rollout validation

The actual Pursuit environment was driven for 320 steps by owner-specific joint
policy profiles: current self actor plus persistent cached teammate actors. All
five frozen gates pass on untouched seeds `92200--92207`.

- non-null refreshes: `198`;
- full actor payload bytes charged: `1,942,776`;
- refreshed actions matching the current donor: `198/198`;
- refreshes that changed the realized cached action: `181`;
- state-dependent graph changes: `202`.

Primary and isolated reproduction JSON are byte-identical with SHA-256
`d540c2104ab92501dd30d9d3c568f8fe18e3308772d0d93b4737e0986660d7d8`.
No reward value was read or summarized and no policy was trained. This closes
the cached-behavior state-machine interface; it does not establish return gain,
neural alignment calibration, or GPU authorization.
