"""Operational amendment: eliminate a phantom disagreement for equal laws.

The frozen original algorithm is unchanged except for this exact-identity
branch. Do not use an epsilon threshold to suppress small genuine differences.
"""
import numpy as np
from relative_return_estimator import probability


def stable_overlap(p,q):
    p,q=probability(p),probability(q)
    common=np.minimum(p,q)
    if np.array_equal(p,q):
        return p,q,common,1.0
    return p,q,common,min(1.0,float(common.sum()))
