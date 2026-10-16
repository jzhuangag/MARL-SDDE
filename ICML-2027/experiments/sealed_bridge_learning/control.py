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
    def __init__(self, options, targets, risk_weight, epsilon=1e-12):
        self.options = tuple(options)
        self.targets = np.asarray(targets, float)
        if (not self.options or self.targets.ndim != 1 or
                not np.isfinite(self.targets).all() or np.any(self.targets <= 0) or
                not np.isfinite(risk_weight) or risk_weight < 0 or
                not np.isfinite(epsilon) or epsilon <= 0):
            raise ValueError('Invalid controller interface')
        self.risk_weight = float(risk_weight)
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
        loss = self.risk_weight*variances + state['costs'] @ state['queues']
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
                    source_or_current_return_used_for_selection=False)


@dataclass
class NativeReservation:
    source: int
    receipt: int
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

    def launch(self, identifier, source_episodes, min_bridge_episodes):
        if identifier in self.reservations or identifier in self.closed_ids:
            raise RuntimeError('Duplicate source cohort')
        if (not isinstance(source_episodes, int) or not isinstance(min_bridge_episodes, int) or
                source_episodes < 2 or min_bridge_episodes < 2):
            raise ValueError('Gradient/variance cohorts require at least two episodes')
        source, bridge = source_episodes*self.horizon, min_bridge_episodes*self.horizon
        if source+bridge > self.free:
            raise RuntimeError('Launch cannot reserve its complete future receipt')
        self.reservations[identifier] = NativeReservation(source, bridge)
        self.events.append(dict(kind='reserve_launch', source_id=identifier,
                                source=source, receipt=bridge))

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
        return r.source_done and amount >= r.receipt and amount <= self.free+r.receipt

    def start_receipt(self, identifier, episode_count):
        if not self.affordable_bridge(identifier, episode_count):
            raise RuntimeError('Unaffordable or incomplete receipt')
        amount = episode_count*self.horizon
        self.paid += amount
        del self.reservations[identifier]
        self.closed_ids.add(identifier)
        self.events.append(dict(kind='receipt_charge', source_id=identifier, transitions=amount))

    def terminal_fresh(self, episode_count):
        if self.reservations:
            raise RuntimeError('In-flight source cohorts must drain before terminal fresh work')
        if not isinstance(episode_count, int) or episode_count <= 0:
            raise ValueError('Terminal work requires complete episodes')
        amount = episode_count*self.horizon
        if amount > self.free:
            raise RuntimeError('Terminal fresh work exceeds exact budget')
        self.paid += amount
        self.events.append(dict(kind='terminal_fresh_charge', transitions=amount))

    def charge_evaluation(self, transitions, role):
        if not isinstance(transitions, int) or transitions < 0 or role not in ('benchmark', 'frozen_policy'):
            raise ValueError('Invalid separate evaluation role/charge')
        self.evaluation_paid += transitions
        self.events.append(dict(kind='evaluation_charge', transitions=transitions, role=role))

    def assert_exhausted(self):
        if self.reservations or self.paid != self.limit:
            raise AssertionError('Training budget not exactly exhausted with drained sources')
