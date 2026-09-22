"""Raw-observation selected-packet factor critic for Pursuit."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .pursuit_delayed_alignment_interface import FEATURE_NAMES, AlignmentLaunch
from .pursuit_profile_factor_critic import (
    ACTION_COUNT,
    MAXIMUM_DEGREE,
    _probability_from_context,
)
from .pursuit_pair_factor_development import heldout_pair_metrics


OBSERVATION_SHAPE = (3, 7, 7)
IDENTITY_COUNT = 8


@dataclass(frozen=True)
class RawSelectedPacketBatch:
    owner_observation: np.ndarray
    donor_observation: np.ndarray
    donor_profile: np.ndarray
    donor_mask: np.ndarray
    owner_identity: np.ndarray
    donor_identity: np.ndarray
    owner_action: np.ndarray
    cost_return: np.ndarray
    seed: np.ndarray
    packet_id: np.ndarray


@dataclass(frozen=True)
class RawCriticScale:
    target_location: float
    target_scale: float


class LocalObservationEncoder(nn.Module):
    def __init__(self, embedding_dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(12 * 7 * 7, int(embedding_dimension)),
            nn.SiLU(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation.float())


class RawObservationFactorCritic(nn.Module):
    """Shared convolutional CTDE critic with additive donor-profile factors."""

    def __init__(
        self,
        *,
        embedding_dimension: int = 32,
        hidden_dimension: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = LocalObservationEncoder(embedding_dimension)
        self.self_head = nn.Sequential(
            nn.Linear(embedding_dimension + IDENTITY_COUNT, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, ACTION_COUNT),
        )
        pair_dimension = (
            2 * embedding_dimension
            + ACTION_COUNT
            + 2 * IDENTITY_COUNT
        )
        self.pair_head = nn.Sequential(
            nn.Linear(pair_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, ACTION_COUNT),
        )
        final_pair = self.pair_head[-1]
        if isinstance(final_pair, nn.Linear):
            nn.init.zeros_(final_pair.weight)
            nn.init.zeros_(final_pair.bias)

    def action_values(
        self,
        owner_observation: torch.Tensor,
        donor_observation: torch.Tensor,
        donor_profile: torch.Tensor,
        donor_mask: torch.Tensor,
        owner_identity: torch.Tensor,
        donor_identity: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, degree = donor_mask.shape
        owner_embedding = self.encoder(owner_observation)
        values = self.self_head(torch.cat((owner_embedding, owner_identity), dim=-1))
        donor_embedding = self.encoder(
            donor_observation.reshape(batch_size * degree, *OBSERVATION_SHAPE)
        ).reshape(batch_size, degree, -1)
        expanded_owner = owner_embedding.unsqueeze(1).expand(-1, degree, -1)
        expanded_owner_identity = owner_identity.unsqueeze(1).expand(
            -1, degree, -1
        )
        pair_input = torch.cat(
            (
                expanded_owner,
                donor_embedding,
                donor_profile,
                expanded_owner_identity,
                donor_identity,
            ),
            dim=-1,
        )
        pair_values = self.pair_head(pair_input)
        return values + torch.sum(pair_values * donor_mask.unsqueeze(-1), dim=1)

    def forward(
        self,
        owner_observation: torch.Tensor,
        donor_observation: torch.Tensor,
        donor_profile: torch.Tensor,
        donor_mask: torch.Tensor,
        owner_identity: torch.Tensor,
        donor_identity: torch.Tensor,
        owner_action: torch.Tensor,
    ) -> torch.Tensor:
        values = self.action_values(
            owner_observation,
            donor_observation,
            donor_profile,
            donor_mask,
            owner_identity,
            donor_identity,
        )
        return values.gather(1, owner_action.long().unsqueeze(1)).squeeze(1)


def _observation_tensor(flattened: Sequence[float]) -> np.ndarray:
    values = np.asarray(flattened, dtype=np.float32)
    if values.shape != (7 * 7 * 3,) or not np.all(np.isfinite(values)):
        raise ValueError("invalid flattened Pursuit observation")
    return values.reshape(7, 7, 3).transpose(2, 0, 1)


def raw_selected_packet_batch(
    rows: Sequence[AlignmentLaunch],
    *,
    maximum_degree: int = MAXIMUM_DEGREE,
) -> RawSelectedPacketBatch:
    """Construct raw-observation critic data from executed packets only."""

    if not rows or maximum_degree <= 0:
        raise ValueError("raw selected packet rows and degree must be nonempty")
    count = len(rows)
    owner_observation = np.zeros((count, *OBSERVATION_SHAPE), dtype=np.float32)
    donor_observation = np.zeros(
        (count, maximum_degree, *OBSERVATION_SHAPE), dtype=np.float32
    )
    donor_profile = np.zeros(
        (count, maximum_degree, ACTION_COUNT), dtype=np.float32
    )
    donor_mask = np.zeros((count, maximum_degree), dtype=np.float32)
    owner_identity = np.zeros((count, IDENTITY_COUNT), dtype=np.float32)
    donor_identity = np.zeros(
        (count, maximum_degree, IDENTITY_COUNT), dtype=np.float32
    )
    owner_action = np.empty(count, dtype=np.int64)
    cost_return = np.empty(count, dtype=np.float32)
    seeds = np.empty(count, dtype=np.int64)
    packet_ids = np.empty(count, dtype=np.int64)
    for row_index, row in enumerate(rows):
        contexts = dict(row.candidate_contexts)
        observations = dict(row.candidate_donor_observation)
        donors = sorted(value for value in contexts if value is not None)
        if set(contexts) != set(observations) or len(donors) > maximum_degree:
            raise ValueError("raw donor support is inconsistent")
        owner_observation[row_index] = _observation_tensor(row.owner_observation)
        owner_identity[row_index, int(row.owner)] = 1.0
        for donor_index, donor in enumerate(donors):
            donor_observation[row_index, donor_index] = _observation_tensor(
                observations[donor]
            )
            donor_profile[row_index, donor_index] = _probability_from_context(
                contexts[donor], current=donor == row.donor
            )
            donor_mask[row_index, donor_index] = 1.0
            donor_identity[row_index, donor_index, int(donor)] = 1.0
        owner_action[row_index] = int(row.launch_owner_action)
        cost_return[row_index] = -float(row.discounted_reward)
        seeds[row_index] = int(row.seed)
        packet_ids[row_index] = int(row.packet_id)
    return RawSelectedPacketBatch(
        owner_observation=owner_observation,
        donor_observation=donor_observation,
        donor_profile=donor_profile,
        donor_mask=donor_mask,
        owner_identity=owner_identity,
        donor_identity=donor_identity,
        owner_action=owner_action,
        cost_return=cost_return,
        seed=seeds,
        packet_id=packet_ids,
    )


def _tensors(batch: RawSelectedPacketBatch):
    return tuple(
        torch.as_tensor(value)
        for value in (
            batch.owner_observation,
            batch.donor_observation,
            batch.donor_profile,
            batch.donor_mask,
            batch.owner_identity,
            batch.donor_identity,
            batch.owner_action,
            batch.cost_return,
        )
    )


def train_raw_factor_critic(
    train: RawSelectedPacketBatch,
    validation: RawSelectedPacketBatch,
    *,
    seed: int,
    epochs: int = 250,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 35,
) -> tuple[RawObservationFactorCritic, RawCriticScale, dict[str, float | int]]:
    """Train on selected packets and freeze at a validation epoch."""

    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("training controls must be positive")
    torch.manual_seed(int(seed))
    target_location = float(np.mean(train.cost_return))
    target_scale = max(float(np.std(train.cost_return)), 1e-6)
    scale = RawCriticScale(target_location, target_scale)
    model = RawObservationFactorCritic()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_tensors = _tensors(train)
    validation_tensors = _tensors(validation)
    train_target = (train_tensors[-1] - target_location) / target_scale
    validation_target = (validation_tensors[-1] - target_location) / target_scale
    generator = torch.Generator().manual_seed(int(seed) + 1)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = -1
    remaining = int(patience)
    for epoch in range(int(epochs)):
        model.train()
        permutation = torch.randperm(len(train.cost_return), generator=generator)
        for start in range(0, len(permutation), int(batch_size)):
            indices = permutation[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                *(value[indices] for value in train_tensors[:-1])
            )
            loss = torch.mean((prediction - train_target[indices]) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = model(*validation_tensors[:-1])
            validation_loss = float(
                torch.mean((validation_prediction - validation_target) ** 2).item()
            )
        if validation_loss < best_validation - 1e-8:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            remaining = int(patience)
        else:
            remaining -= 1
            if remaining == 0:
                break
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, scale, {
        "best_epoch": int(best_epoch),
        "epochs_executed": int(epoch + 1),
        "best_validation_standardized_mse": float(best_validation),
    }


def predict_raw_selected_cost(
    model: RawObservationFactorCritic,
    scale: RawCriticScale,
    batch: RawSelectedPacketBatch,
) -> np.ndarray:
    with torch.no_grad():
        standardized = model(*_tensors(batch)[:-1]).cpu().numpy().astype(float)
    return scale.target_location + scale.target_scale * standardized


def predict_raw_alignment_candidates(
    *,
    model: RawObservationFactorCritic,
    scale: RawCriticScale,
    row: AlignmentLaunch,
) -> dict[int | None, float]:
    contexts = dict(row.candidate_contexts)
    observations = dict(row.candidate_donor_observation)
    donors = sorted(value for value in contexts if value is not None)
    owner_observation = torch.as_tensor(
        _observation_tensor(row.owner_observation)
    ).unsqueeze(0)
    owner_identity = torch.zeros((1, IDENTITY_COUNT), dtype=torch.float32)
    owner_identity[0, int(row.owner)] = 1.0
    with torch.no_grad():
        owner_embedding = model.encoder(owner_observation)
        values = model.self_head(
            torch.cat((owner_embedding, owner_identity), dim=-1)
        ).squeeze(0)
        cached_pair: dict[int, torch.Tensor] = {}
        current_pair: dict[int, torch.Tensor] = {}
        for donor in donors:
            donor_observation = torch.as_tensor(
                _observation_tensor(observations[donor])
            ).unsqueeze(0)
            donor_embedding = model.encoder(donor_observation)
            donor_identity = torch.zeros((1, IDENTITY_COUNT), dtype=torch.float32)
            donor_identity[0, int(donor)] = 1.0
            common = (owner_embedding, donor_embedding)
            cached = torch.as_tensor(
                _probability_from_context(contexts[donor], current=False),
                dtype=torch.float32,
            ).unsqueeze(0)
            current = torch.as_tensor(
                _probability_from_context(contexts[donor], current=True),
                dtype=torch.float32,
            ).unsqueeze(0)
            cached_pair[donor] = model.pair_head(
                torch.cat(
                    (*common, cached, owner_identity, donor_identity), dim=-1
                )
            ).squeeze(0)
            current_pair[donor] = model.pair_head(
                torch.cat(
                    (*common, current, owner_identity, donor_identity), dim=-1
                )
            ).squeeze(0)
            values = values + cached_pair[donor]
        direction = torch.as_tensor(
            row.owner_probability_direction, dtype=torch.float32
        )
        null = float(scale.target_scale * (direction @ values))
        result: dict[int | None, float] = {None: null}
        for donor in donors:
            refreshed = values + current_pair[donor] - cached_pair[donor]
            result[donor] = float(scale.target_scale * (direction @ refreshed))
    return result


def raw_counterfactual_alignment_metrics(
    *,
    model: RawObservationFactorCritic,
    scale: RawCriticScale,
    rows: Sequence[AlignmentLaunch],
) -> dict[str, float | int]:
    truth: list[float] = []
    prediction: list[float] = []
    packet: list[int] = []
    null_alignment: list[float] = []
    for row in rows:
        if not row.reference_available or not row.candidate_factor_mean_alignment:
            continue
        actual = dict(row.candidate_factor_mean_alignment)
        forecast = predict_raw_alignment_candidates(model=model, scale=scale, row=row)
        if set(actual) != set(forecast) or None not in actual:
            raise ValueError("raw critic candidate sets are inconsistent")
        identifier = int(row.seed) * 1_000_000 + int(row.packet_id)
        null_alignment.append(float(actual[None]))
        for donor in sorted(value for value in actual if value is not None):
            truth.append(float(actual[donor] - actual[None]))
            prediction.append(float(forecast[donor] - forecast[None]))
            packet.append(identifier)
    metrics = heldout_pair_metrics(
        truth=np.asarray(truth),
        prediction=np.asarray(prediction),
        packet_id=np.asarray(packet),
    )
    edge_scale = float(np.mean(np.abs(truth)))
    null_scale = float(np.mean(np.abs(null_alignment)))
    metrics.update(
        {
            "mean_absolute_edge_effect": edge_scale,
            "mean_absolute_null_alignment": null_scale,
            "edge_effect_to_null_scale": edge_scale / max(null_scale, 1e-15),
        }
    )
    return metrics
