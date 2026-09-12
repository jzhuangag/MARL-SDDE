# Baseline-as-sensing analytic headroom audit

This is an outcome-free deterministic calculation on the frozen public T-018
model grid. It reads no trajectory, pilot, or formal result. The oracle is
given the hidden low/high regime free of charge, so its advantage is an upper
bound rather than evidence that a sensing algorithm attains it.

- Finite scenario-by-budget cells: `30502`
- Cells with no feasible catalogue action: `4010`
- Median ideal headroom: `0.000000`
- Fraction with ideal headroom at least 10%: `0.217625`
- Canonical per-cell record SHA-256:
  `0bcb5321f0d5e3764fad1a287b29e709d5a598a2618768652c21e2d1417a000e`
- Frozen gate (median >= 10% and prevalence >= 60%): `false`
- Decision: **STOP: ideal oracle headroom gate fails; do not authorize a scientific experiment**

The strong baseline is the regime-blind minimax action from the full catalogue,
not the all-agent action. The all-agent calculation is descriptive only.

## Stratified results

```json
{
  "Q": {
    "16": {
      "count": 10172,
      "fraction_at_least_10_percent": 0.2509830908375934,
      "max": 0.30256361998309533,
      "median": 0.004498553577521436,
      "min": 0.0,
      "p90": 0.14977816345368278
    },
    "32": {
      "count": 10242,
      "fraction_at_least_10_percent": 0.23071665690294865,
      "max": 0.4583335644443458,
      "median": 0.0011221743086238245,
      "min": 0.0,
      "p90": 0.14275644844222424
    },
    "8": {
      "count": 10088,
      "fraction_at_least_10_percent": 0.17069785884218874,
      "max": 0.35757575757575744,
      "median": 0.0,
      "min": 0.0,
      "p90": 0.12677359446226577
    }
  },
  "budget_ray": {
    "balanced": {
      "count": 10414,
      "fraction_at_least_10_percent": 0.10706740925676973,
      "max": 0.35757575757575744,
      "median": 0.0,
      "min": 0.0,
      "p90": 0.10520388021802074
    },
    "environment_limited": {
      "count": 10638,
      "fraction_at_least_10_percent": 0.07689415303628501,
      "max": 0.268880264481817,
      "median": 0.0,
      "min": 0.0,
      "p90": 0.08593247375967161
    },
    "message_limited": {
      "count": 9450,
      "fraction_at_least_10_percent": 0.4978835978835979,
      "max": 0.4583335644443458,
      "median": 0.09963119582886293,
      "min": 0.0,
      "p90": 0.17952190181658456
    }
  },
  "delay": {
    "0": {
      "count": 8626,
      "fraction_at_least_10_percent": 0.2573614653373522,
      "max": 0.4583335644443458,
      "median": 0.0008983382427639741,
      "min": 0.0,
      "p90": 0.14602198409466405
    },
    "12": {
      "count": 7194,
      "fraction_at_least_10_percent": 0.1939115929941618,
      "max": 0.2805664191806132,
      "median": 0.0,
      "min": 0.0,
      "p90": 0.13363985260116962
    },
    "24": {
      "count": 6664,
      "fraction_at_least_10_percent": 0.2036314525810324,
      "max": 0.2816879284082253,
      "median": 0.0,
      "min": 0.0,
      "p90": 0.1419612766767655
    },
    "4": {
      "count": 8018,
      "fraction_at_least_10_percent": 0.20778248939885258,
      "max": 0.28007570125311676,
      "median": 0.0003573384373152244,
      "min": 0.0,
      "p90": 0.13520339104558798
    }
  },
  "lambda": {
    "0.2": {
      "count": 7412,
      "fraction_at_least_10_percent": 0.4440097139773341,
      "max": 0.4583335644443458,
      "median": 0.08333327219119474,
      "min": 0.0,
      "p90": 0.1948870015074089
    },
    "0.7": {
      "count": 7542,
      "fraction_at_least_10_percent": 0.2141341819146115,
      "max": 0.44426765395348844,
      "median": 0.009014983268554044,
      "min": 0.0,
      "p90": 0.13361587939377906
    },
    "0.9": {
      "count": 7736,
      "fraction_at_least_10_percent": 0.12215615305067218,
      "max": 0.26500075815384605,
      "median": 0.0,
      "min": 0.0,
      "p90": 0.11008370798340983
    },
    "0.94": {
      "count": 7812,
      "fraction_at_least_10_percent": 0.10074244751664106,
      "max": 0.20916279819027273,
      "median": 0.0,
      "min": 0.0,
      "p90": 0.10024852007520291
    }
  },
  "theta_high": {
    "0.5": {
      "count": 8108,
      "fraction_at_least_10_percent": 0.13085841144548593,
      "max": 0.19642814040813272,
      "median": 0.0,
      "min": 0.0,
      "p90": 0.10767272742929568
    },
    "1.0": {
      "count": 7756,
      "fraction_at_least_10_percent": 0.21183599793708097,
      "max": 0.23203609701071637,
      "median": 0.0,
      "min": 0.0,
      "p90": 0.1278383214909904
    },
    "2.0": {
      "count": 7472,
      "fraction_at_least_10_percent": 0.25481798715203424,
      "max": 0.4461525858461538,
      "median": 0.0,
      "min": 0.0,
      "p90": 0.16159920260989902
    },
    "4.0": {
      "count": 7166,
      "fraction_at_least_10_percent": 0.2832821657828635,
      "max": 0.4583335644443458,
      "median": 0.0008409604708793372,
      "min": 0.0,
      "p90": 0.1879692887550678
    }
  }
}
```
