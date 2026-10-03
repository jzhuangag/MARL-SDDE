"""Selected-packet profile-conditioned factor critic for Pursuit.

The training view contains only executed trajectories.  Counterfactual branch
labels remain outside the critic API and are used solely by a held-out audit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .pursuit_delayed_alignment_interface import FEATURE_NAMES, AlignmentLaunch
from .pursuit_pair_factor_development import PAIR_FEATURE_NAMES, pair_feature_vector
from .pursuit_pair_factor_development import heldout_pair_metrics


ACTION_COUNT = 5
MAXIMUM_DEGREE = 4
_CURRENT_PROBABILITY_INDEX = tuple(
    FEATURE_NAMES.index(f"donor_current_probability_{action}")
    for action in range(ACTION_COUNT)
)
_CACHED_PROBABILITY_INDEX = tuple(
    FEATURE_NAMES.index(f"donor_cached_probability_{action}")
    for action in range(ACTION_COUNT)
)


@dataclass(frozen=True)
class SelectedPacketBatch:
    self_features: np.ndarray
    pair_features: np.ndarray
    donor_profiles: np.ndarray
    donor_mask: np.ndarray
    owner_action: np.ndarray
    cost_return: np.ndarray
    seed: np.ndarray
    packet_id: np.ndarray


@dataclass(frozen=True)
class FactorNormalization:
    self_location: np.ndarray
    self_scale: np.ndarray
    pair_location: np.ndarray
    pair_scale: np.ndarray


class ProfileConditionedFactorCritic(nn.Module):
    """Parameter-shared local critic with additive teammate-policy factors."""

    def __init__(self, *, feature_dimension: int, hidden_dimension: int = 48) -> None:
        super().__init__()
        self.self_head = nn.Sequential(
            nn.Linear(int(feature_dimension), int(hidden_dimension)),
            nn.SiLU(),
            nn.Linear(int(hidden_dimension), ACTION_COUNT),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(
                int(feature_dimension) + ACTION_COUNT,
                int(hidden_dimension),
            ),
            nn.SiLU(),
            nn.Linear(int(hidden_dimension), ACTION_COUNT),
        )

    def action_values(
        self,
        self_features: torch.Tensor,
        pair_features: torch.Tensor,
        donor_profiles: torch.Tensor,
        donor_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = self.self_head(self_features)
        pair_input = torch.cat((pair_features, donor_profiles), dim=-1)
        pair_values = self.pair_head(pair_input)
        return values + torch.sum(pair_values * donor_mask.unsqueeze(-1), dim=1)

    def forward(
        self,
        self_features: torch.Tensor,
        pair_features: torch.Tensor,
        donor_profiles: torch.Tensor,
        donor_mask: torch.Tensor,
        owner_action: torch.Tensor,
    ) -> torch.Tensor:
        values = self.action_values(
            self_features,
            pair_features,
            donor_profiles,
            donor_mask,
        )
        return values.gather(1, owner_action.long().unsqueeze(1)).squeeze(1)


def _probability_from_context(
    context: Sequence[float], *, current: bool
) -> np.ndarray:
    values = np.asarray(context, dtype=float)
    indices = (
        _CURRENT_PROBABILITY_INDEX if current else _CACHED_PROBABILITY_INDEX
    )
    result = values[np.asarray(indices, dtype=int)]
    if result.shape != (ACTION_COUNT,) or not np.all(np.isfinite(result)):
        raise ValueError("invalid donor policy profile")
    if abs(float(np.sum(result)) - 1.0) > 1e-6:
        raise ValueError("donor policy profile must sum to one")
    return result


def selected_packet_batch(
    rows: Sequence[AlignmentLaunch],
    *,
    maximum_degree: int = MAXIMUM_DEGREE,
) -> SelectedPacketBatch:
    """Remove every counterfactual outcome before constructing critic data."""

    if not rows or maximum_degree <= 0:
        raise ValueError("selected packet rows and degree must be nonempty")
    feature_dimension = len(PAIR_FEATURE_NAMES)
    self_features = np.zeros((len(rows), feature_dimension), dtype=np.float32)
    pair_features = np.zeros(
        (len(rows), maximum_degree, feature_dimension), dtype=np.float32
    )
    donor_profiles = np.zeros(
        (len(rows), maximum_degree, ACTION_COUNT), dtype=np.float32
    )
    donor_mask = np.zeros((len(rows), maximum_degree), dtype=np.float32)
    owner_action = np.empty(len(rows), dtype=np.int64)
    cost_return = np.empty(len(rows), dtype=np.float32)
    seeds = np.empty(len(rows), dtype=np.int64)
    packet_ids = np.empty(len(rows), dtype=np.int64)
    for row_index, row in enumerate(rows):
        contexts = dict(row.candidate_contexts)
        if None not in contexts:
            raise ValueError("selected packet row is missing its null context")
        self_features[row_index] = pair_feature_vector(contexts[None])
        donors = sorted(value for value in contexts if value is not None)
        if len(donors) > maximum_degree:
            raise ValueError("selected packet exceeds maximum degree")
        for donor_index, donor in enumerate(donors):
            context = contexts[donor]
            pair_features[row_index, donor_index] = pair_feature_vector(context)
            donor_profiles[row_index, donor_index] = _probability_from_context(
                context,
                current=donor == row.donor,
            )
            donor_mask[row_index, donor_index] = 1.0
        owner_action[row_index] = int(row.launch_owner_action)
        cost_return[row_index] = -float(row.discounted_reward)
        seeds[row_index] = int(row.seed)
        packet_ids[row_index] = int(row.packet_id)
    return SelectedPacketBatch(
        self_features=self_features,
        pair_features=pair_features,
        donor_profiles=donor_profiles,
        donor_mask=donor_mask,
        owner_action=owner_action,
        cost_return=cost_return,
        seed=seeds,
        packet_id=packet_ids,
    )


def fit_factor_normalization(batch: SelectedPacketBatch) -> FactorNormalization:
    self_location = np.mean(batch.self_features, axis=0)
    self_scale = np.std(batch.self_features, axis=0)
    active_pairs = batch.pair_features[batch.donor_mask.astype(bool)]
    if active_pairs.size == 0:
        raise ValueError("factor critic requires at least one active pair")
    pair_location = np.mean(active_pairs, axis=0)
    pair_scale = np.std(active_pairs, axis=0)
    self_scale[self_scale < 1e-6] = 1.0
    pair_scale[pair_scale < 1e-6] = 1.0
    return FactorNormalization(
        self_location=self_location.astype(np.float32),
        self_scale=self_scale.astype(np.float32),
        pair_location=pair_location.astype(np.float32),
        pair_scale=pair_scale.astype(np.float32),
    )


def normalize_selected_batch(
    batch: SelectedPacketBatch,
    normalization: FactorNormalization,
) -> SelectedPacketBatch:
    self_features = (
        batch.self_features - normalization.self_location
    ) / normalization.self_scale
    pair_features = (
        batch.pair_features - normalization.pair_location
    ) / normalization.pair_scale
    pair_features = pair_features * batch.donor_mask[:, :, None]
    return SelectedPacketBatch(
        self_features=self_features.astype(np.float32),
        pair_features=pair_features.astype(np.float32),
        donor_profiles=batch.donor_profiles.copy(),
        donor_mask=batch.donor_mask.copy(),
        owner_action=batch.owner_action.copy(),
        cost_return=batch.cost_return.copy(),
        seed=batch.seed.copy(),
        packet_id=batch.packet_id.copy(),
    )


def _tensor_batch(batch: SelectedPacketBatch):
    return tuple(
        torch.as_tensor(value)
        for value in (
            batch.self_features,
            batch.pair_features,
            batch.donor_profiles,
            batch.donor_mask,
            batch.owner_action,
            batch.cost_return,
        )
    )


def train_profile_factor_critic(
    train: SelectedPacketBatch,
    validation: SelectedPacketBatch,
    *,
    seed: int,
    hidden_dimension: int = 48,
    epochs: int = 300,
    learning_rate: float = 0.003,
    weight_decay: float = 1e-4,
    patience: int = 40,
) -> tuple[ProfileConditionedFactorCritic, dict[str, float | int]]:
    """Fit on selected packets and freeze at the best validation epoch."""

    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive")
    torch.manual_seed(int(seed))
    model = ProfileConditionedFactorCritic(
        feature_dimension=train.self_features.shape[1],
        hidden_dimension=hidden_dimension,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_tensors = _tensor_batch(train)
    validation_tensors = _tensor_batch(validation)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = -1
    remaining = int(patience)
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*train_tensors[:-1])
        loss = torch.mean((prediction - train_tensors[-1]) ** 2)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = model(*validation_tensors[:-1])
            validation_loss = float(
                torch.mean(
                    (validation_prediction - validation_tensors[-1]) ** 2
                ).item()
            )
        if validation_loss < best_validation - 1e-10:
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
    return model, {
        "best_epoch": int(best_epoch),
        "epochs_executed": int(epoch + 1),
        "best_validation_mse": float(best_validation),
    }


def predict_selected_cost(
    model: ProfileConditionedFactorCritic,
    batch: SelectedPacketBatch,
) -> np.ndarray:
    with torch.no_grad():
        return model(*_tensor_batch(batch)[:-1]).cpu().numpy().astype(float)


def predict_alignment_candidates(
    *,
    model: ProfileConditionedFactorCritic,
    normalization: FactorNormalization,
    row: AlignmentLaunch,
) -> dict[int | None, float]:
    """Score launch candidates without reading any counterfactual outcome."""

    contexts = dict(row.candidate_contexts)
    donors = sorted(value for value in contexts if value is not None)
    self_features = (
        pair_feature_vector(contexts[None]) - normalization.self_location
    ) / normalization.self_scale
    pair_features = np.vstack(
        [
            (
                pair_feature_vector(contexts[donor])
                - normalization.pair_location
            )
            / normalization.pair_scale
            for donor in donors
        ]
    ) if donors else np.zeros((0, len(PAIR_FEATURE_NAMES)), dtype=float)
    self_tensor = torch.as_tensor(self_features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        values = model.self_head(self_tensor).squeeze(0)
        cached_pair: dict[int, torch.Tensor] = {}
        current_pair: dict[int, torch.Tensor] = {}
        for index, donor in enumerate(donors):
            feature = torch.as_tensor(
                pair_features[index], dtype=torch.float32
            ).unsqueeze(0)
            cached = torch.as_tensor(
                _probability_from_context(contexts[donor], current=False),
                dtype=torch.float32,
            ).unsqueeze(0)
            current = torch.as_tensor(
                _probability_from_context(contexts[donor], current=True),
                dtype=torch.float32,
            ).unsqueeze(0)
            cached_pair[donor] = model.pair_head(
                torch.cat((feature, cached), dim=-1)
            ).squeeze(0)
            current_pair[donor] = model.pair_head(
                torch.cat((feature, current), dim=-1)
            ).squeeze(0)
            values = values + cached_pair[donor]
        direction = torch.as_tensor(
            row.owner_probability_direction, dtype=torch.float32
        )
        result: dict[int | None, float] = {None: float(direction @ values)}
        for donor in donors:
            refreshed = values + current_pair[donor] - cached_pair[donor]
            result[donor] = float(direction @ refreshed)
    return result


def counterfactual_alignment_metrics(
    *,
    model: ProfileConditionedFactorCritic,
    normalization: FactorNormalization,
    rows: Sequence[AlignmentLaunch],
) -> dict[str, float | int]:
    """Audit selected-packet critic predictions on unseen replicated means."""

    truth: list[float] = []
    prediction: list[float] = []
    packet: list[int] = []
    null_alignment: list[float] = []
    for row in rows:
        if not row.reference_available or not row.candidate_factor_mean_alignment:
            continue
        actual = dict(row.candidate_factor_mean_alignment)
        forecast = predict_alignment_candidates(
            model=model,
            normalization=normalization,
            row=row,
        )
        if set(actual) != set(forecast) or None not in actual:
            raise ValueError("candidate audit sets are inconsistent")
        identifier = int(row.seed) * 1_000_000 + int(row.packet_id)
        null_alignment.append(float(actual[None]))
        for donor in sorted(value for value in actual if value is not None):
            truth.append(float(actual[donor] - actual[None]))
            prediction.append(float(forecast[donor] - forecast[None]))
            packet.append(identifier)
    metrics = heldout_pair_metrics(
        truth=np.asarray(truth, dtype=float),
        prediction=np.asarray(prediction, dtype=float),
        packet_id=np.asarray(packet, dtype=int),
    )
    metrics["mean_absolute_null_alignment"] = float(
        np.mean(np.abs(null_alignment))
    )
    metrics["mean_absolute_edge_effect"] = float(np.mean(np.abs(truth)))
    metrics["edge_effect_to_null_scale"] = float(
        np.mean(np.abs(truth))
        / max(1e-15, np.mean(np.abs(null_alignment)))
    )
    return metrics
