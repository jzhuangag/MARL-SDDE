"""Complete-cell learning integration; science driver/lock are not yet issued.

This callable backend NEVER runs on import. A later locked driver must own
execution, prospective resource feasibility, cap/replay/analysis and all gates.
Until that driver/source freeze exists, run_cell MUST NOT be invoked on native
data. The earlier outcome-free qualification remains a different implementation.
"""
import hashlib
import time
import numpy as np
import torch
from sealed_bridge.core import Option, SealedAnchor, risk_feedback, transported_direction, bridge_weights
from sealed_bridge_learning.control import NativeBudget, RenewalRiskQueue
from sealed_bridge_learning.native import (Charges, Sampler, actor_bytes, actor_parameter_count,
                                           bridge_gradients, fit_critic, qualified, snapshot,
                                           source_gradients)
from sealed_bridge_learning.updates import PredictableActorScale, joint_vtrace_update, ppo_update
from sealed_bridge_learning.guard import require_cell


CONTROLS = ('fresh_happo_half', 'fresh_happo', 'fresh_happo_double', 'fresh_mappo',
            'delayed_happo', 'online_joint_kl', 'joint_ma_trace_adaptation',
            'defensive_page_pg_adaptation', 'fresh_pg')
ABLATIONS = ('no_transport', 'fixed_allocation', 'no_price')
METHODS = CONTROLS+('full',)+ABLATIONS


def _immutable(data):
    for name in ('obs', 'actions', 'logs', 'states', 'rewards'):
        for x in data[name] if isinstance(data[name], list) else [data[name]]:
            x.setflags(write=False)


def _service(seed, ordinal, minimum, maximum):
    word = hashlib.sha256(f'{seed}:sealed_learning_service:{ordinal}'.encode()).digest()
    return minimum+int.from_bytes(word[:8], 'big') % (maximum-minimum+1)


def _set_actor_rate(actors, learning_rate):
    for actor in actors:
        for group in actor.actor_optimizer.param_groups:
            group['lr'] = learning_rate


def _adam_direction(actors, gradient, charges):
    """Declared reference Adam, never covered by the SGD transport theorem."""
    start = 0
    for actor in actors:
        actor.actor_optimizer.zero_grad()
        for parameter in actor.actor.parameters():
            end = start+parameter.numel()
            parameter.grad = -torch.as_tensor(gradient[start:end].reshape(parameter.shape), dtype=parameter.dtype).clone()
            start = end
        actor.actor_optimizer.step()
        charges.add('defensive_reference:server_adam', actor_adam_calls=1)
    if start != len(gradient):
        raise AssertionError('Reference joint-gradient geometry mismatch')


def run_cell(task_index, seed, method, plan, check_budget):
    require_cell(check_budget, task_index, seed, method, plan)
    if method not in METHODS:
        raise ValueError('Unregistered method')
    task = plan['tasks'][task_index]
    h = int(task['horizon'])
    training_limit = int(task['training_transitions'])
    if training_limit % h:
        raise ValueError('Training allocation must preserve complete episodes')
    p = qualified.base.config()
    p.update(episode_length=h, hidden_sizes=plan['hidden_sizes'], activation=plan['activation'],
             discount=plan['gamma'], actor_optimizer_steps_per_packet=plan['ppo_epochs'],
             critic_optimizer_steps_per_block=plan['critic_epochs'])
    start_cpu = time.process_time()
    charges = Charges()
    actors, critic, normalizer = qualified.initialize(task, seed, p)
    rates = dict(fresh_happo_half=.5, fresh_happo=1., fresh_happo_double=2.)
    _set_actor_rate(actors, plan['ppo_learning_rate']*rates.get(method, 1.))
    initial_hash = qualified.policy_hash(actors)
    frozen, _, _ = snapshot(actors, critic, normalizer, charges, 'frozen_policy:initial_copy')
    sampler = Sampler(task, h, seed, task_index, charges, plan['vector_width'])
    frozen_sampler = Sampler(task, h, seed, task_index, charges, plan['vector_width'])
    ledger = NativeBudget(training_limit, h)
    trace, curves, launches = [], [], []
    policy_version = 0
    eval_index = 0
    thresholds = np.linspace(0, training_limit, plan['evaluation_intervals']+1).astype(int)
    roles = plan['roles']
    rng = np.random.default_rng(qualified.base.seed_for(seed, task_index, 0, roles['controller'], 0))
    scale = None
    reference_episodes = plan['fresh_batch_episodes']
    maximum_exposure = max(reference_episodes, plan['source_episodes']+max(plan['bridge_episodes']))
    pg_family = method in ('full', 'fresh_pg')+ABLATIONS

    def evaluate(force=False):
        nonlocal eval_index
        while eval_index < len(thresholds) and (ledger.paid >= thresholds[eval_index] or force):
            if not force and thresholds[eval_index] == training_limit:
                # Final return belongs to the DRAINED algorithm, not the first
                # receipt after the last source samples happened to be paid.
                break
            check_budget()
            before = qualified.policy_hash(actors)
            # Separate collectors share evaluation world/action seeds intentionally.
            # Neither stream shares training roles or supplies controller feedback.
            current = sampler.collect(actors, eval_index, roles['evaluation'], plan['evaluation_episodes'],
                                      purpose='benchmark', deterministic=plan['deterministic_evaluation'])
            anchor = frozen_sampler.collect(frozen, eval_index, roles['evaluation'], plan['evaluation_episodes'],
                                             purpose='frozen_policy', deterministic=plan['deterministic_evaluation'])
            if current['identities'] != anchor['identities']:
                raise AssertionError('Frozen-policy control lost its same-world pairing')
            if qualified.policy_hash(actors) != before or qualified.policy_hash(frozen) != initial_hash:
                raise AssertionError('Evaluation modified a training/frozen policy')
            ledger.charge_evaluation(current['native_transitions'], 'benchmark')
            ledger.charge_evaluation(anchor['native_transitions'], 'frozen_policy')
            curves.append(dict(index=eval_index, threshold=int(thresholds[eval_index]),
                               training_transitions=ledger.paid, policy_version=policy_version,
                               benchmark_returns=current['returns'].tolist(),
                               frozen_returns=anchor['returns'].tolist(),
                               mean_return=float(current['returns'].mean()),
                               frozen_mean_return=float(anchor['returns'].mean()),
                               evaluation_ids=current['identities'], costs=charges.snapshot(),
                               evaluation_used_by_controller=False))
            eval_index += 1
            if not force and (eval_index == len(thresholds) or ledger.paid < thresholds[eval_index]):
                break

    def expose_step(actor_step, episodes):
        actor_step = dict(actor_step)
        actor_step['alpha_reference'] = actor_step['alpha']
        actor_step['paid_episode_exposure'] = episodes
        actor_step['alpha'] *= episodes/reference_episodes
        return actor_step

    def fresh_sgd(data, event, actor_step):
        nonlocal policy_version
        gradient, _ = source_gradients(actors, critic, normalizer, data, plan['gamma'], charges)
        direction = gradient.mean(0)
        before = qualified.policy_hash(actors)
        qualified.sgd_commit(actors, direction, actor_step['alpha'])
        charges.add(event, actor_sgd_calls=len(actors))
        scale.close(direction)
        policy_version += 1
        return dict(before=before, after=qualified.policy_hash(actors), actor_step=actor_step,
                    current_gradient_selected_step=False, source_optimizer_imported=False)

    try:
        evaluate()
        if pg_family:
            # The calibration data are additional PAID training, not qualification
            # outcomes. No policy update uses this bank, so its RMS is predictable
            # for the first later update. Critic fitting follows its gradient role.
            calibration = plan['calibration_episodes']
            ledger.charge_unreserved(calibration, 'calibration_charge')
            data = sampler.collect(actors, 0, roles['calibration'], calibration)
            gradients, _ = source_gradients(actors, critic, normalizer, data, plan['gamma'], charges)
            scale = PredictableActorScale(gradients, actor_parameter_count(actors),
                                          plan['pg_base_rate'],
                                          plan['pg_rate_cap']*reference_episodes/maximum_exposure)
            episode_bytes = qualified.payload_bytes(data)//calibration
            fit_critic(actors, critic, normalizer, data, p, charges)
            trace.append(dict(kind='charged_calibration', training_transitions=ledger.paid,
                              actor_updated=False, data_ids=data['identities'], costs=charges.snapshot()))
            evaluate()

        if method in ('full',)+ABLATIONS:
            options = [Option(m, beta) for m in plan['bridge_episodes']
                       for beta in ([0.] if method == 'no_transport' else plan['betas'])]
            source_n = plan['source_episodes']
            max_m, min_m = max(plan['bridge_episodes']), min(plan['bridge_episodes'])
            agents = len(actors)
            P = actor_bytes(actors)
            durations = np.asarray([source_n+o.episodes for o in options], dtype=float)
            # Exactly the source+bridge training work rows and logical payload;
            # evaluate only after critic fitting has completed the transaction.
            resources = np.asarray([[
                ((3*agents+1)*source_n+(5*agents+3+2*plan['critic_epochs'])*o.episodes)*h,
                (source_n+o.episodes)*episode_bytes+3*P] for o in options], dtype=float)
            controller = RenewalRiskQueue(options, durations, resources, 1., prices_enabled=method != 'no_price')
            # A shape/prespecified-rate normalization, not a learned upper L.
            risk_coefficient = (plan['variance_tradeoff']*2*reference_episodes**2 /
                                (plan['pg_base_rate']**2*actor_parameter_count(actors)))
            jobs, used = [], set()
            tick, ordinal = 0, 0
            while True:
                check_budget()
                # Reserve the WHOLE menu, so affordability never postselects the
                # statistical action set after the anchor is opened.
                if ledger.free >= (source_n+max_m)*h:
                    ledger.launch(ordinal, source_n, min_m, max_m)
                    sa, sc, sn = snapshot(actors, critic, normalizer, charges, 'source:immutable_snapshot')
                    source_hash = qualified.policy_hash(sa)
                    ledger.start_source(ordinal)
                    data = sampler.collect(sa, ordinal, roles['source'], source_n)
                    ledger.finish_source(ordinal)
                    _immutable(data)
                    delay = _service(seed, ordinal, plan['delay_min'], plan['delay_max'])
                    job = dict(id=ordinal, due=tick+delay, launch_tick=tick, version=policy_version,
                               actors=sa, critic=sc, normalizer=sn, data=data,
                               data_hash=qualified.array_hash(data), policy_hash=source_hash)
                    jobs.append(job)
                    launches.append(dict(id=ordinal, launch_tick=tick, due=job['due'],
                                         source_version=policy_version, source_hash=source_hash,
                                         data_hash=job['data_hash'], source_ids=data['identities'],
                                         source_episodes=source_n, reserved_bridge_max=max_m,
                                         costs=charges.snapshot()))
                    ordinal += 1
                ready = sorted((j for j in jobs if j['due'] <= tick), key=lambda j: (j['due'], j['id']))
                for job in ready:
                    check_budget()
                    if job['id'] in used:
                        raise AssertionError('Source cohort reused')
                    before = qualified.policy_hash(actors)
                    actor_step = scale.begin()
                    charge_before = charges.snapshot()
                    if method == 'fixed_allocation':
                        index = options.index(Option(plan['fixed_bridge_episodes'], plan['fixed_beta']))
                        decision = dict(index=index, source_id=str(job['id']), before_anchor_and_bridge=True)
                    else:
                        # Actual alpha is predictable. This coefficient is a
                        # declared risk tradeoff, NOT an estimated neural L.
                        controller.risk_weight = risk_coefficient*actor_step['alpha']**2/(2*reference_episodes**2)
                        decision = controller.begin(rng, job['id'], before)
                    selected = options[decision['index']]
                    actor_step = expose_step(actor_step, source_n+selected.episodes)
                    if not all(ledger.affordable_bridge(job['id'], o.episodes) for o in options):
                        raise AssertionError('Full prospective menu reservation failed')
                    if (qualified.array_hash(job['data']) != job['data_hash'] or
                            qualified.policy_hash(job['actors']) != job['policy_hash']):
                        raise AssertionError('Immutable delayed program changed')
                    sg, _ = source_gradients(job['actors'], job['critic'], job['normalizer'],
                                              job['data'], plan['gamma'], charges)
                    anchor = SealedAnchor(str(job['id']), sg)
                    ag = anchor.open_once(decision)
                    ledger.start_receipt(job['id'], selected.episodes)
                    data = sampler.collect(actors, job['id'], roles['bridge'], selected.episodes,
                                           source=job['actors'], source_fraction=plan['source_fraction'])
                    paired = bridge_gradients(actors, job['actors'], critic, normalizer,
                                                job['critic'], job['normalizer'], data, plan['gamma'], charges,
                                                plan['source_fraction'])
                    direction = transported_direction(paired['target'], paired['control'], ag, selected.beta)
                    risks = risk_feedback(paired['target'], paired['control'], ag, options)
                    charges.add('bridge:all_option_variance', variance_scalar_entries=
                                len(options)*selected.episodes*actor_parameter_count(actors))
                    if qualified.policy_hash(actors) != before:
                        raise AssertionError('Candidate policies changed during the bank')
                    qualified.sgd_commit(actors, direction, actor_step['alpha'])
                    charges.add('bridge:server_sgd', actor_sgd_calls=len(actors))
                    scale.close(direction)
                    policy_version += 1
                    fit_critic(actors, critic, normalizer, data, p, charges)
                    charge_after = charges.snapshot()
                    actual_resources = np.asarray([
                        charge_after['work_rows']-charge_before['work_rows']+agents*source_n*h,
                        charge_after['logical_bytes']-charge_before['logical_bytes']+
                        qualified.payload_bytes(job['data'])+P], dtype=float)
                    if not np.array_equal(actual_resources, resources[decision['index']]):
                        raise AssertionError('Complete source/bridge resource recurrence mismatch')
                    feedback = {} if method == 'fixed_allocation' else controller.close(risks, actual_resources)
                    used.add(job['id'])
                    trace.append(dict(kind='bridge_receipt', source_id=job['id'], tick=tick,
                                      source_version=job['version'], target_version=policy_version-1,
                                      actual_version_age=policy_version-1-job['version'],
                                      before=before, after=qualified.policy_hash(actors),
                                      decision=dict(index=decision['index'], episodes=selected.episodes,
                                                    beta=selected.beta, before_source_and_bridge=True),
                                      actor_step=actor_step, feedback=feedback,
                                      risk_feedback=risks.tolist(), actual_resources=actual_resources.tolist(),
                                      risk_weight=controller.risk_weight, resources=resources.tolist(),
                                      costs_before_receipt=charge_before,
                                      source_hash=job['policy_hash'], data_hash=job['data_hash'],
                                      anchor_hash=anchor.sha256, source_consumed_once=anchor.consumed,
                                      bridge_ids=data['identities'], complete_policy_modes=data['modes'],
                                      prefix_overlap_mean_by_time=paired['prefix_overlap'].mean(0).tolist(),
                                      current_return_ranked=False, evaluation_used=False,
                                      costs=charges.snapshot(), training_transitions=ledger.paid))
                    jobs.remove(job)
                    evaluate()
                if not jobs and ledger.free < (source_n+max_m)*h:
                    break
                tick += 1
            if len(used) != len(launches):
                raise AssertionError('Undrained source cohort at termination')
            terminal = 0
            while ledger.free:
                check_budget()
                amount = min(plan['fresh_batch_episodes'], ledger.free//h)
                actor_step = expose_step(scale.begin(), amount)
                ledger.terminal_fresh(amount)
                data = sampler.collect(actors, terminal, roles['terminal'], amount)
                details = fresh_sgd(data, 'terminal:server_fresh_sgd', actor_step)
                fit_critic(actors, critic, normalizer, data, p, charges)
                trace.append(dict(kind='terminal_fresh', episodes=amount, **details,
                                  data_ids=data['identities'],
                                  costs=charges.snapshot(), training_transitions=ledger.paid))
                terminal += 1
                evaluate()

        elif method in ('delayed_happo', 'online_joint_kl', 'joint_ma_trace_adaptation'):
            jobs, tick, ordinal = [], 0, 0
            used = set()
            while ledger.free or jobs:
                check_budget()
                if ledger.free:
                    amount = min(plan['fresh_batch_episodes'], ledger.free//h)
                    sa, _, _ = snapshot(actors, critic, normalizer, charges, 'delayed_reference:source_policy')
                    source_hash = qualified.policy_hash(sa)
                    ledger.charge_unreserved(amount, 'delayed_reference_source_charge')
                    data = sampler.collect(sa, ordinal, roles['source'], amount)
                    _immutable(data)
                    delay = _service(seed, ordinal, plan['delay_min'], plan['delay_max'])
                    jobs.append(dict(id=ordinal, due=tick+delay, launch_tick=tick,
                                     version=policy_version, hash=source_hash, data=data))
                    launches.append(dict(id=ordinal, launch_tick=tick, due=tick+delay,
                                         source_version=policy_version, source_hash=source_hash,
                                         source_ids=data['identities'], data_hash=qualified.array_hash(data),
                                         costs=charges.snapshot()))
                    ordinal += 1
                for job in sorted((j for j in jobs if j['due'] <= tick), key=lambda j: (j['due'], j['id'])):
                    before = qualified.policy_hash(actors)
                    version_before = policy_version
                    if method == 'joint_ma_trace_adaptation':
                        details = joint_vtrace_update(actors, critic, normalizer, job['data'], p, charges,
                                                      plan['vtrace_rho_cap'], plan['vtrace_c_cap'])
                        target = details.pop('critic_targets')
                        accepted = True
                    else:
                        target = None
                        details = ppo_update(actors, critic, normalizer, job['data'], p, charges,
                                             kl_cap=plan['joint_kl_cap'] if method == 'online_joint_kl' else None)
                        accepted = details['accepted']
                    policy_version += int(accepted)
                    fit_critic(actors, critic, normalizer, job['data'], p, charges, target=target)
                    trace.append(dict(kind='delayed_reference_receipt', source_id=job['id'], tick=tick,
                                      source_version=job['version'], target_version=version_before,
                                      actual_version_age=version_before-job['version'], before=before,
                                      after=qualified.policy_hash(actors), details=details,
                                      costs=charges.snapshot(), training_transitions=ledger.paid))
                    if job['id'] in used:
                        raise AssertionError('Repeated reference program')
                    used.add(job['id']); jobs.remove(job)
                    evaluate()
                tick += 1
            if len(used) != len(launches):
                raise AssertionError('Reference source drain incomplete')

        elif method == 'defensive_page_pg_adaptation':
            previous, old_critic, old_normalizer, cached = None, None, None, None
            ordinal = 0
            while ledger.free:
                check_budget()
                # PAGE refresh decision precedes the new bank; beta/cache choices
                # are not selected from current validation or evaluation returns.
                refresh = cached is None or rng.random() < plan['page_refresh_probability']
                amount = plan['page_refresh_episodes'] if refresh else plan['page_bridge_episodes']
                if amount*h > ledger.free:
                    amount, refresh = ledger.free//h, True
                before = qualified.policy_hash(actors)
                ca, cc, cn = snapshot(actors, critic, normalizer, charges, 'defensive_reference:previous_policy')
                ledger.charge_unreserved(amount, 'defensive_reference_charge')
                data = sampler.collect(actors, ordinal, roles['fresh'], amount,
                                       source=None if refresh else previous,
                                       source_fraction=plan['source_fraction'])
                if refresh:
                    g, _ = source_gradients(actors, critic, normalizer, data, plan['gamma'], charges)
                    direction = g.mean(0)
                else:
                    # Full-trajectory defensive PAGE reference, deliberately NOT
                    # our prefix estimator or a falsely labelled official run.
                    target, tl = source_gradients(actors, critic, normalizer, data, plan['gamma'], charges)
                    source, sl = source_gradients(previous, old_critic, old_normalizer, data, plan['gamma'], charges)
                    wt, ws = bridge_weights(tl, sl, plan['source_fraction'])
                    direction = cached+(wt[:, None]*target-ws[:, None]*source).mean(0)
                _adam_direction(actors, direction, charges)
                cached = np.array(direction, copy=True)
                charges.add('defensive_reference:cached_gradient_copy', copy_tensor_bytes=cached.nbytes)
                previous, old_critic, old_normalizer = ca, cc, cn
                policy_version += 1
                fit_critic(actors, critic, normalizer, data, p, charges)
                trace.append(dict(kind='defensive_page_reference', refresh=bool(refresh), episodes=amount,
                                  data_ids=data['identities'],
                                  before=before, after=qualified.policy_hash(actors),
                                  official_reproduction=False, evaluation_used=False,
                                  costs=charges.snapshot(), training_transitions=ledger.paid))
                ordinal += 1
                evaluate()

        else:
            ordinal = 0
            while ledger.free:
                check_budget()
                amount = min(plan['fresh_batch_episodes'], ledger.free//h)
                actor_step = expose_step(scale.begin(), amount) if method == 'fresh_pg' else None
                ledger.charge_unreserved(amount, 'fresh_reference_charge')
                data = sampler.collect(actors, ordinal, roles['fresh'], amount)
                before = qualified.policy_hash(actors)
                if method == 'fresh_pg':
                    details = fresh_sgd(data, 'fresh_reference:server_sgd', actor_step)
                else:
                    details = ppo_update(actors, critic, normalizer, data, p, charges,
                                         sequential=method != 'fresh_mappo')
                    policy_version += 1
                fit_critic(actors, critic, normalizer, data, p, charges)
                trace.append(dict(kind='fresh_reference', before=before,
                                  data_ids=data['identities'],
                                  after=qualified.policy_hash(actors), episodes=amount, details=details,
                                  costs=charges.snapshot(), training_transitions=ledger.paid))
                ordinal += 1
                evaluate()
        ledger.assert_exhausted()
        evaluate(force=True)
        if charges.values['native_training'] != training_limit:
            raise AssertionError('Actual native training and reservations disagree')
        if charges.values['native_benchmark']+charges.values['native_frozen_policy'] != ledger.evaluation_paid:
            raise AssertionError('Evaluation charge recurrence disagrees')
        return dict(task=task['name'], task_index=task_index, seed=seed, method=method,
                    policy_parameter_bytes=actor_bytes(actors), actor_parameter_count=actor_parameter_count(actors),
                    initial_hash=initial_hash, final_hash=qualified.policy_hash(actors),
                    training_transitions=training_limit, costs=charges.snapshot(),
                    charge_event_totals=charges.events, native_budget_events=ledger.events,
                    launches=launches, transactions=trace, curves=curves,
                    cpu_seconds=time.process_time()-start_cpu, all_sources_drained=True,
                    evaluation_used_by_controller=False, current_return_ranked=False,
                    physical_parallel_speedup_claimed=False,
                    algorithm_theorem_scope='conditional_own_objective_SGD; references/Adam_not_covered')
    except BaseException:
        if hasattr(check_budget, 'record_failure_metadata'):
            check_budget.record_failure_metadata(dict(costs=charges.snapshot(),
                charge_event_totals=charges.events, native_budget_events=ledger.events,
                launches_completed=len(launches), transactions_completed=len(trace),
                policy_version=policy_version, cpu_seconds=time.process_time()-start_cpu))
        raise
    finally:
        sampler.close()
        frozen_sampler.close()
