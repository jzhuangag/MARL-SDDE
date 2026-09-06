from __future__ import annotations

import torch

from .async_pistonball_ctde import (
    PistonActor,
    PistonCentralCritic,
    PolicyCacheBank,
    signed_refresh_for_batch,
)
from .neural_signed_cache import parameter_bytes, signed_cache_choice


def test_one_vjp_recovers_exact_bilinear_alignment_changes() -> None:
    owner = torch.tensor([0.7], requires_grad=True)
    donor_zero = torch.tensor([0.2], requires_grad=True)
    donor_one = torch.tensor([-0.4], requires_grad=True)
    current_donors = {0: torch.tensor([0.8]), 1: torch.tensor([-0.1])}
    current_loss = 0.5 * owner.square().sum() + owner @ sum(
        current_donors.values()
    )
    cached_loss = 0.5 * owner.square().sum() + owner @ (donor_zero + donor_one)
    choice = signed_cache_choice(
        current_owner_loss=current_loss,
        cached_owner_loss=cached_loss,
        owner_parameters=(owner,),
        donor_parameters={0: (donor_zero,), 1: (donor_one,)},
        donor_displacements={
            0: (current_donors[0] - donor_zero.detach(),),
            1: (current_donors[1] - donor_one.detach(),),
        },
        taylor_remainder_by_donor={0: 0.0, 1: 0.0},
        step=0.1,
        smoothness=1.0,
        packet_second_moment_upper=4.0,
        receipt_motion_upper=0.0,
        communication_queue=0.0,
        learning_weight=1.0,
    )
    assert choice.donor == 0
    assert choice.candidate_count == 3
    assert choice.vjp_calls == 1
    assert choice.best_edge_donor == 0
    assert choice.best_edge_index is not None
    assert choice.best_edge_index < choice.null_index
    assert choice.best_edge_learning_index_delta is not None
    assert choice.best_edge_cache_reset_benefit == 0.0
    assert choice.best_edge_queue_price == 0.0


def test_queue_price_can_select_null_action() -> None:
    owner = torch.tensor([0.7], requires_grad=True)
    donor = torch.tensor([0.2], requires_grad=True)
    current = torch.tensor([0.8])
    current_loss = 0.5 * owner.square().sum() + owner @ current
    cached_loss = 0.5 * owner.square().sum() + owner @ donor
    choice = signed_cache_choice(
        current_owner_loss=current_loss,
        cached_owner_loss=cached_loss,
        owner_parameters=(owner,),
        donor_parameters={0: (donor,)},
        donor_displacements={0: (current - donor.detach(),)},
        taylor_remainder_by_donor={0: 0.0},
        step=0.1,
        smoothness=1.0,
        packet_second_moment_upper=4.0,
        receipt_motion_upper=0.0,
        communication_queue=100.0,
        learning_weight=1.0,
    )
    assert choice.donor is None


def test_taylor_remainder_can_reject_an_uncertain_refresh() -> None:
    owner = torch.tensor([0.7], requires_grad=True)
    donor = torch.tensor([0.2], requires_grad=True)
    current = torch.tensor([0.8])
    current_loss = 0.5 * owner.square().sum() + owner @ current
    cached_loss = 0.5 * owner.square().sum() + owner @ donor
    optimistic = signed_cache_choice(
        current_owner_loss=current_loss,
        cached_owner_loss=cached_loss,
        owner_parameters=(owner,),
        donor_parameters={0: (donor,)},
        donor_displacements={0: (current - donor.detach(),)},
        taylor_remainder_by_donor={0: 0.0},
        step=0.1,
        smoothness=1.0,
        packet_second_moment_upper=4.0,
        receipt_motion_upper=0.0,
        communication_queue=0.0,
        learning_weight=1.0,
    )
    conservative = signed_cache_choice(
        current_owner_loss=current_loss,
        cached_owner_loss=cached_loss,
        owner_parameters=(owner,),
        donor_parameters={0: (donor,)},
        donor_displacements={0: (current - donor.detach(),)},
        taylor_remainder_by_donor={0: 10.0},
        step=0.1,
        smoothness=1.0,
        packet_second_moment_upper=4.0,
        receipt_motion_upper=0.0,
        communication_queue=0.0,
        learning_weight=1.0,
    )
    assert optimistic.donor == 0
    assert conservative.donor is None


def test_parameter_bytes_counts_tensor_payload() -> None:
    tensors = (torch.zeros(3, 4, dtype=torch.float32), torch.zeros(2, dtype=torch.float64))
    assert parameter_bytes(tensors) == 3 * 4 * 4 + 2 * 8


def test_exact_cache_reset_benefit_can_prevent_null_absorption() -> None:
    owner = torch.tensor([0.7], requires_grad=True)
    donor = torch.tensor([0.2], requires_grad=True)
    current = torch.tensor([0.8])
    current_loss = 0.5 * owner.square().sum() + owner @ current
    cached_loss = 0.5 * owner.square().sum() + owner @ donor
    choice = signed_cache_choice(
        current_owner_loss=current_loss,
        cached_owner_loss=cached_loss,
        owner_parameters=(owner,),
        donor_parameters={0: (donor,)},
        donor_displacements={0: (current - donor.detach(),)},
        taylor_remainder_by_donor={0: 10.0},
        cache_reset_benefit_by_donor={0: 2.0},
        step=0.1,
        smoothness=1.0,
        packet_second_moment_upper=4.0,
        receipt_motion_upper=0.0,
        communication_queue=0.0,
        learning_weight=1.0,
    )
    assert choice.donor == 0
    assert choice.cache_reset_benefit == 2.0
    assert choice.best_edge_cache_reset_benefit == 2.0


def test_smooth_critic_exposes_mixed_policy_curvature_that_relu_hides() -> None:
    torch.manual_seed(771)
    observations = torch.randn(2, 4, 3, 64, 32)
    states = torch.randn(2, 3, 64, 64)

    def learning_delta(activation: str) -> float:
        actors = tuple(PistonActor(activation=activation) for _ in range(4))
        caches = PolicyCacheBank(actors)
        with torch.no_grad():
            actors[1].action_head.bias.add_(0.2)
        critic = PistonCentralCritic(4, activation=activation)
        choice = signed_refresh_for_batch(
            owner=0,
            eligible_donors=(1,),
            actors=actors,
            caches=caches,
            critic=critic,
            observations=observations,
            states=states,
            step=0.01,
            communication_queue=0.0,
            learning_weight=1.0,
            smoothness=1.0,
            packet_second_moment_upper=1.0,
            receipt_motion_upper=0.0,
            taylor_coefficient=0.0,
            cache_debt_weight=0.0,
        )
        assert choice.best_edge_learning_index_delta is not None
        return choice.best_edge_learning_index_delta

    relu_delta = learning_delta("relu")
    silu_delta = learning_delta("silu")
    assert relu_delta == 0.0
    assert abs(silu_delta) > 1e-12
