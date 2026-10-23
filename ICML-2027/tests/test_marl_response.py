import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from marl_response import QuadraticResponse, discounted_returns, episode_seed


def test_discounted_complete_episode_returns():
    np.testing.assert_allclose(discounted_returns(np.array([1., 2., 3.]), .5), [2.75, 3.5, 3.])


def test_shared_seeds_and_disjoint_roles():
    assert len({episode_seed(1, 0, 10, 0, w, 1.) for w in range(8)}) == 1
    assert len({episode_seed(1, 0, 10, 0, w, 0.) for w in range(8)}) == 8
    assert len({episode_seed(1, 0, role, 0, 0, 1.) for role in (10, 20, 30, 40)}) == 4


def test_quadratic_response_exact_and_lifted_coordinates():
    function = lambda p: (p[0].square()+3*p[1].square()).sum()
    initial = [torch.tensor([1.]), torch.tensor([2.])]
    model = QuadraticResponse(function, initial)
    following = [torch.tensor([1.1]), torch.tensor([1.8])]
    prediction, linear = model.forecast(following)
    np.testing.assert_allclose(prediction, float(function(following)-function(initial)), atol=2e-6)
    naive, _ = model.forecast(following, freeze_last=1)
    assert abs(naive-prediction) > 1
