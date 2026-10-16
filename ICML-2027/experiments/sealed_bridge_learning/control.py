"""Scale-adaptive risk/queue control and hard launch/receipt reservations.

These are science-backend primitives, not a registered or executed learning
study. No native rollout, evaluation or optimizer is launched by this module.
"""
from dataclasses import dataclass
import numpy as np


def simplex(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid simplex input')
    u = np.sort(x)[::-1]
    threshold = (np.cumsum(u)-1)/np.arange(1, len(u)+1)
    index = np.flatnonzero(u > threshold)[-1]
    return np.maximum(x-threshold[index], 0)


class AdaptiveRiskQueue:
    """Predictable decisions; post-feedback AdaGrad OGD, no tuned tiny meta rate.

The adaptive OGD step uses the current loss ONLY after the current action.
Variance feedback is unbiased for all options, not an upper certificate.
The nonnegative risk weight must be declared by the eventual study and cannot
be presented as a certified neural smoothness coefficient unless justified.
"""
    def __init__(self, options, targets, risk_weight, epsilon=1e-12, prices_enabled=True):
        self.options = tuple(options)
        self.targets = np.asarray(targets, float)
        if (not self.options or self.targets.ndim != 1 or
                not np.isfinite(self.targets).all() or np.any(self.targets <= 0) or
                not np.isfinite(risk_weight) or risk_weight < 0 or
                not np.isfinite(epsilon) or epsilon <= 0):
            raise ValueError('Invalid controller interface')
        self.risk_weight = float(risk_weight)
        self.prices_enabled = bool(prices_enabled)
        self.epsilon = float(epsilon)
        self.probabilities = np.full(len(options), 1/len(options))
        self.queues = np.zeros_like(self.targets)
        self.loss_energy = 0.0
        self.pending = None
        self.rounds = 0

    def begin(self, costs, rng, source_id, policy_hash):
        if self.pending is not None:
            raise RuntimeError('Unclosed receipt')
        costs = np.asarray(costs, float)
        if (costs.shape != (len(self.options), len(self.targets)) or
                not np.isfinite(costs).all() or np.any(costs < 0)):
            raise ValueError('Invalid prospective costs')
        index = int(rng.choice(len(self.options), p=self.probabilities))
        self.pending = dict(index=index, costs=costs.copy(),
                            probabilities=self.probabilities.copy(), queues=self.queues.copy(),
                            source_id=str(source_id), policy_hash=str(policy_hash))
        return dict(index=index, option=self.options[index], source_id=str(source_id),
                    policy_hash=str(policy_hash), before_anchor_and_bridge=True)

    def close(self, variances, actual_cost):
        if self.pending is None:
            raise RuntimeError('Missing pre-data decision')
        variances, actual_cost = np.asarray(variances, float), np.asarray(actual_cost, float)
        if (variances.shape != self.probabilities.shape or
                not np.isfinite(variances).all() or np.any(variances < 0)):
            raise ValueError('Invalid all-option variance feedback')
        state = self.pending
        if not np.array_equal(actual_cost, state['costs'][state['index']]):
            raise AssertionError('Receipt charge differs from reservation')
        loss = self.risk_weight*variances
        if self.prices_enabled:
            loss = loss+state['costs'] @ state['queues']
        # Constant loss offsets have zero inner product with p-u on a simplex.
        gradient = loss-loss.mean()
        self.loss_energy += float(gradient @ gradient)
        diameter = np.sqrt(2.0)
        rate = diameter/np.sqrt(self.epsilon+self.loss_energy)
        self.probabilities = simplex(state['probabilities']-rate*gradient)
        self.queues = np.maximum(0.0, state['queues']+actual_cost-self.targets)
        self.pending = None
        self.rounds += 1
        return dict(index=state['index'], source_id=state['source_id'],
                    probabilities_before=state['probabilities'].tolist(),
                    probabilities_after=self.probabilities.tolist(),
                    queues_before=state['queues'].tolist(), queues_after=self.queues.tolist(),
                    losses=loss.tolist(), centered_loss_energy=self.loss_energy,
                    post_feedback_rate=float(rate), variance_is_upper_certificate=False,
                    prices_enabled=self.prices_enabled,
                    source_or_current_return_used_for_selection=False)


@dataclass
class NativeReservation:
    source: int
    receipt: int
    minimum_receipt: int = 0
    source_started: bool = False
    source_done: bool = False


class NativeBudget:
    """No in-flight refund; exact independent training/evaluation ledgers.

At launch, reserve source AND the minimum affordable future bridge. Terminal
fresh work is legal only after every source receipt is drained. Partial final
episodes are a changed objective, not an accounting fix, and are prohibited.
"""
    def __init__(self, training_limit, episode_length):
        if (not isinstance(training_limit, int) or not isinstance(episode_length, int) or
                training_limit <= 0 or episode_length <= 0 or
                training_limit % episode_length):
            raise ValueError('Budget must be a positive integer number of full episodes')
        self.limit = training_limit
        self.horizon = episode_length
        self.paid = 0
        self.reservations = {}
        self.closed_ids = set()
        self.events = []
        self.evaluation_paid = 0

    @property
    def reserved(self):
        return sum(r.source+r.receipt for r in self.reservations.values())

    @property
    def free(self):
        return self.limit-self.paid-self.reserved

    def launch(self, identifier, source_episodes, min_bridge_episodes, max_bridge_episodes=None):
        if identifier in self.reservations or identifier in self.closed_ids:
            raise RuntimeError('Duplicate source cohort')
        if (not isinstance(source_episodes, int) or not isinstance(min_bridge_episodes, int) or
                source_episodes < 2 or min_bridge_episodes < 2):
            raise ValueError('Gradient/variance cohorts require at least two episodes')
        maximum = min_bridge_episodes if max_bridge_episodes is None else max_bridge_episodes
        if not isinstance(maximum, int) or maximum < min_bridge_episodes:
            raise ValueError('Invalid full-menu bridge reservation')
        source, bridge = source_episodes*self.horizon, maximum*self.horizon
        if source+bridge > self.free:
            raise RuntimeError('Launch cannot reserve its complete future receipt')
        self.reservations[identifier] = NativeReservation(source, bridge, min_bridge_episodes*self.horizon)
        self.events.append(dict(kind='reserve_launch', source_id=identifier,
                                source=source, receipt=bridge,
                                minimum_receipt=min_bridge_episodes*self.horizon))

    def start_source(self, identifier):
        r = self.reservations[identifier]
        if r.source_started:
            raise RuntimeError('Source cohort already started')
        # Irrevocable charge BEFORE rollout; failure never returns this credit.
        amount = r.source
        r.source = 0
        r.source_started = True
        self.paid += amount
        self.events.append(dict(kind='source_charge', source_id=identifier, transitions=amount))

    def finish_source(self, identifier):
        r = self.reservations[identifier]
        if not r.source_started or r.source_done:
            raise RuntimeError('Invalid source completion')
        r.source_done = True
        self.events.append(dict(kind='source_complete', source_id=identifier))

    def affordable_bridge(self, identifier, episode_count):
        r = self.reservations[identifier]
        if not isinstance(episode_count, int) or episode_count < 2:
            return False
        amount = episode_count*self.horizon
        return r.source_done and amount >= r.minimum_receipt and amount <= self.free+r.receipt

    def start_receipt(self, identifier, episode_count):
        if not self.affordable_bridge(identifier, episode_count):
            raise RuntimeError('Unaffordable or incomplete receipt')
        amount = episode_count*self.horizon
        self.paid += amount
        del self.reservations[identifier]
        self.closed_ids.add(identifier)
        self.events.append(dict(kind='receipt_charge', source_id=identifier, transitions=amount))

    def terminal_fresh(self, episode_count):
        self.charge_unreserved(episode_count, 'terminal_fresh_charge')

    def charge_unreserved(self, episode_count, event):
        if self.reservations:
            raise RuntimeError('In-flight source cohorts must drain before terminal fresh work')
        if not isinstance(episode_count, int) or episode_count <= 0:
            raise ValueError('Terminal work requires complete episodes')
        amount = episode_count*self.horizon
        if amount > self.free:
            raise RuntimeError('Terminal fresh work exceeds exact budget')
        self.paid += amount
        self.events.append(dict(kind=event, transitions=amount))

    def charge_evaluation(self, transitions, role):
        if not isinstance(transitions, int) or transitions < 0 or role not in ('benchmark', 'frozen_policy'):
            raise ValueError('Invalid separate evaluation role/charge')
        self.evaluation_paid += transitions
        self.events.append(dict(kind='evaluation_charge', transitions=transitions, role=role))

    def assert_exhausted(self):
        if self.reservations or self.paid != self.limit:
            raise AssertionError('Training budget not exactly exhausted with drained sources')


class RenewalRiskQueue:
    """Native-exposure control, not a cheap-receipt-count surrogate.

An exposure distribution w is converted to receipt probabilities p proportional
to w/duration. Thus E[duration*loss]/E[duration] = w.loss. Full-information OGD
sees E[duration] times the normalized loss, with all data used only NEXT round.
Durations are complete paid SOURCE+BRIDGE episodes, not current m alone.
"""
    def __init__(self, options, durations, resources, risk_weight, epsilon=1e-12, prices_enabled=True):
        self.options = tuple(options)
        self.durations = np.asarray(durations, float)
        self.resources = np.asarray(resources, float)
        if (self.durations.shape != (len(self.options),) or
                self.resources.shape != (len(self.options), 2) or
                not np.isfinite(self.durations).all() or np.any(self.durations <= 0) or
                not np.isfinite(self.resources).all() or np.any(self.resources <= 0) or
                not np.isfinite(risk_weight) or risk_weight < 0 or epsilon <= 0):
            raise ValueError('Invalid full-cost renewal catalogue')
        rates = self.resources/self.durations[:, None]
        self.scales = rates.max(axis=0)
        self.normalized = self.resources/self.scales
        # Public, outcome-free endpoint exposure mixture yields a Slater witness
        # whenever neither work nor bytes is constant across the m catalogue.
        ends = [int(self.durations.argmin()), int(self.durations.argmax())]
        witness_rate = np.mean(self.normalized[ends]/self.durations[ends, None], axis=0)
        self.targets = (1.+witness_rate)/2
        if not np.all(witness_rate < self.targets):
            raise ValueError('No strict endpoint resource slack; scientific freeze forbidden')
        self.slater_slack = self.targets-witness_rate
        self.debts = self.normalized-self.durations[:, None]*self.targets
        self.w = np.full(len(self.options), 1/len(self.options))
        self.queues = np.zeros(2)
        self.risk_weight, self.epsilon = float(risk_weight), float(epsilon)
        self.prices_enabled, self.loss_energy = bool(prices_enabled), 0.
        self.pending = None

    def begin(self, rng, source_id, policy_hash):
        if self.pending is not None:
            raise RuntimeError('Unclosed renewal receipt')
        inverse = self.w/self.durations
        probabilities = inverse/inverse.sum()
        expected_duration = 1./inverse.sum()
        index = int(rng.choice(len(self.options), p=probabilities))
        self.pending = dict(index=index, w=self.w.copy(), queues=self.queues.copy(),
                            expected_duration=float(expected_duration), probabilities=probabilities,
                            source_id=str(source_id), policy_hash=str(policy_hash))
        return dict(index=index, source_id=str(source_id), before_anchor_and_bridge=True,
                    policy_hash=str(policy_hash), expected_paid_episodes=float(expected_duration))

    def close(self, variances, actual_resources):
        if self.pending is None:
            raise RuntimeError('Missing predictable renewal decision')
        state = self.pending
        variances = np.asarray(variances, float)
        if (variances.shape != self.w.shape or not np.isfinite(variances).all() or
                np.any(variances < 0)):
            raise ValueError('Invalid all-option renewal feedback')
        if not np.array_equal(np.asarray(actual_resources, float), self.resources[state['index']]):
            raise AssertionError('Actual source+bridge work/bytes disagree with pre-data costs')
        # Exact deterministic quadratic queue remainder belongs in the score.
        losses = self.risk_weight*self.durations*variances
        if self.prices_enabled:
            losses += (self.debts @ state['queues']+.5*np.square(self.debts).sum(-1))/self.durations
        gradient = state['expected_duration']*(losses-losses.mean())
        self.loss_energy += float(gradient @ gradient)
        rate = np.sqrt(2.)/np.sqrt(self.epsilon+self.loss_energy)
        self.w = simplex(state['w']-rate*gradient)
        self.queues = np.maximum(0., state['queues']+self.debts[state['index']])
        self.pending = None
        return dict(index=state['index'], source_id=state['source_id'],
                    exposure_before=state['w'].tolist(), receipt_probabilities=state['probabilities'].tolist(),
                    exposure_after=self.w.tolist(), expected_paid_episodes=state['expected_duration'],
                    queues_before=state['queues'].tolist(), queues_after=self.queues.tolist(),
                    normalized_risk_losses=losses.tolist(), full_feedback_energy=self.loss_energy,
                    resource_scales=self.scales.tolist(), resource_rate_targets=self.targets.tolist(),
                    static_slater_slack=self.slater_slack.tolist(), prices_enabled=self.prices_enabled,
                    current_data_selected_action=False, variance_upper_certificate=False)
