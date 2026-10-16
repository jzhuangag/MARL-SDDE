"""All-method complete-learning audit, frozen task gates and resource plots."""
import csv
import json
from pathlib import Path
import numpy as np
from sealed_bridge.core import Option
from sealed_bridge_learning.control import NativeBudget, RenewalRiskQueue
from sealed_bridge_learning.guard import dump, sha
from sealed_bridge_learning.benchmark import CONTROLS, METHODS, _service
from sealed_bridge_learning.native import Charges, qualified


def finite(value):
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(x) for x in value)
    return not isinstance(value, (int, float)) or bool(np.isfinite(value))


def audit(row, plan, require_staleness=True):
    assert finite(row), 'Nonfinite artifact'
    task = plan['tasks'][row['task_index']]
    assert row['task'] == task['name'] and row['method'] in METHODS
    assert row['training_transitions'] == task['training_transitions']
    assert row['all_sources_drained'] and not row['evaluation_used_by_controller']
    assert not row['current_return_ranked'] and not row['physical_parallel_speedup_claimed']
    costs = row['costs']
    aggregate = dict.fromkeys(Charges.FIELDS, 0)
    for event in row['charge_event_totals'].values():
        assert event['calls'] >= 1 and all(isinstance(v, int) and v >= 0 for v in event.values())
        for k, v in event.items():
            if k != 'calls':
                aggregate[k] += v
    assert all(aggregate[k] == costs[k] for k in aggregate)
    assert costs['logical_bytes'] == costs['policy_broadcast_bytes']+costs['program_upload_bytes']
    assert costs['work_rows'] == sum(costs[k] for k in ('sampling_actor_rows', 'actor_forward_rows',
        'actor_backward_rows', 'critic_forward_rows', 'critic_backward_rows'))
    # Reconstruct every hard reservation/debit, not just the last paid total.
    ledger = NativeBudget(task['training_transitions'], task['horizon'])
    for x in row['native_budget_events']:
        kind = x['kind']
        if kind == 'reserve_launch':
            ledger.launch(x['source_id'], x['source']//task['horizon'],
                          x['minimum_receipt']//task['horizon'], x['receipt']//task['horizon'])
        elif kind == 'source_charge':
            ledger.start_source(x['source_id'])
        elif kind == 'source_complete':
            ledger.finish_source(x['source_id'])
        elif kind == 'receipt_charge':
            ledger.start_receipt(x['source_id'], x['transitions']//task['horizon'])
        elif kind == 'evaluation_charge':
            ledger.charge_evaluation(x['transitions'], x['role'])
        else:
            ledger.charge_unreserved(x['transitions']//task['horizon'], kind)
        assert ledger.free >= 0
    ledger.assert_exhausted()
    assert ledger.events == row['native_budget_events']
    assert costs['native_training'] == ledger.paid
    assert costs['native_benchmark']+costs['native_frozen_policy'] == ledger.evaluation_paid
    h, E, A = task['horizon'], plan['critic_epochs'], 3 if row['task'] == 'spread' else 2
    expected_eval = (plan['evaluation_intervals']+1)*plan['evaluation_episodes']*h
    assert costs['native_benchmark'] == costs['native_frozen_policy'] == expected_eval
    launches = {x['id']: x for x in row['launches']}
    assert len(launches) == len(row['launches'])
    training_ids, evaluation_ids, seen = [], [], set()
    for ordinal, s in enumerate(row['launches']):
        assert s['id'] == ordinal
        assert s['due'] == s['launch_tick']+_service(row['seed'], ordinal, plan['delay_min'], plan['delay_max'])
        training_ids.extend(s['source_ids'])
    prior, version, ages = row['initial_hash'], 0, []
    rng = np.random.default_rng(qualified.base.seed_for(row['seed'], row['task_index'], 0,
                                                      plan['roles']['controller'], 0))
    controller = None
    for t in row['transactions']:
        assert not t.get('evaluation_used', False) and not t.get('current_return_ranked', False)
        training_ids.extend(t.get('data_ids', []))
        if 'before' in t:
            assert t['before'] == prior
            prior = t['after']
        kind = t['kind']
        if kind == 'charged_calibration':
            assert not t['actor_updated'] and len(t['data_ids']) == plan['calibration_episodes']
            continue
        if kind in ('bridge_receipt', 'delayed_reference_receipt'):
            sid = t['source_id']; s = launches[sid]
            assert sid not in seen
            seen.add(sid)
            assert t['source_version'] == s['source_version']
            assert t['target_version'] == version and t['tick'] >= s['due']
            assert t['actual_version_age'] == version-s['source_version'] >= 0
            ages.append(t['actual_version_age'])
        if kind == 'bridge_receipt':
            assert t['source_hash'] == s['source_hash'] and t['data_hash'] == s['data_hash']
            assert t['source_consumed_once'] and t['decision']['before_source_and_bridge']
            assert len(t['bridge_ids']) == t['decision']['episodes']
            assert len(t['complete_policy_modes']) == len(t['bridge_ids'])
            assert set(t['complete_policy_modes']) <= {0, 1}
            training_ids.extend(t['bridge_ids'])
            assert t['actor_step']['before_source_and_bridge']
            assert not t['actor_step']['neural_smoothness_certificate']
            n = plan['source_episodes']+t['decision']['episodes']
            assert t['actor_step']['paid_episode_exposure'] == n
            assert t['actor_step']['alpha'] == t['actor_step']['alpha_reference']*n/plan['fresh_batch_episodes']
            options = [Option(m, b) for m in plan['bridge_episodes']
                       for b in ([0.] if row['method'] == 'no_transport' else plan['betas'])]
            resources = np.asarray(t['resources'])
            assert resources.shape == (len(options), 2)
            for j, o in enumerate(options):
                assert resources[j, 0] == h*((3*A+1)*plan['source_episodes']+(5*A+3+2*E)*o.episodes)
            # Known payload is linear in episode count plus exactly three policy copies.
            i0, i1 = 0, len(options)-1
            epbytes = (resources[i1, 1]-resources[i0, 1])/(options[i1].episodes-options[i0].episodes)
            assert all(resources[j, 1] == (plan['source_episodes']+o.episodes)*epbytes+
                       3*row['policy_parameter_bytes'] for j, o in enumerate(options))
            before, after = t['costs_before_receipt'], t['costs']
            actual = [after['work_rows']-before['work_rows']+A*plan['source_episodes']*h,
                      after['logical_bytes']-before['logical_bytes']+plan['source_episodes']*epbytes+
                      row['policy_parameter_bytes']]
            assert actual == t['actual_resources'] == resources[t['decision']['index']].tolist()
            if row['method'] == 'fixed_allocation':
                assert options[t['decision']['index']] == Option(plan['fixed_bridge_episodes'], plan['fixed_beta'])
            else:
                if controller is None:
                    controller = RenewalRiskQueue(options, [plan['source_episodes']+o.episodes for o in options],
                                                  resources, t['risk_weight'], prices_enabled=row['method'] != 'no_price')
                controller.risk_weight = t['risk_weight']
                d = controller.begin(rng, sid, t['before'])
                assert d['index'] == t['decision']['index']
                assert controller.close(t['risk_feedback'], t['actual_resources']) == t['feedback']
        accepted = not (kind == 'delayed_reference_receipt' and not t['details'].get('accepted', True))
        version += int(accepted)
    assert prior == row['final_hash'] and seen == set(launches)
    if require_staleness and row['launches']:
        assert max(ages) > 0, 'No actual joint-policy version staleness'
    assert len(training_ids) == len(set(training_ids))
    assert len(training_ids)*h == costs['native_training']
    assert len(row['curves']) == plan['evaluation_intervals']+1
    for i, c in enumerate(row['curves']):
        assert c['index'] == i and not c['evaluation_used_by_controller']
        assert len(c['evaluation_ids']) == plan['evaluation_episodes']
        evaluation_ids.extend(c['evaluation_ids'])
    assert len(evaluation_ids) == len(set(evaluation_ids))
    assert set(training_ids).isdisjoint(evaluation_ids)
    assert row['curves'][-1]['training_transitions'] == task['training_transitions']
    assert row['curves'][-1]['policy_version'] == version
    assert row['curves'][-1]['costs'] == costs
    return dict(task=row['task'], seed=row['seed'], method=row['method'], payload_pass=True,
                max_actual_staleness=max(ages, default=0), costs=costs, cpu_seconds=row['cpu_seconds'])


def series(row, axis, frozen=False):
    pairs = {}
    for p in row['curves']:
        x = p['training_transitions'] if axis == 'native' else p['costs'][axis]
        pairs[x] = p['frozen_mean_return' if frozen else 'mean_return']
    xy = sorted(pairs.items())
    return np.asarray([x for x, _ in xy], float), np.asarray([y for _, y in xy], float)


def auc(row, axis, lo, hi):
    x, y = series(row, axis)
    grid = np.unique(np.r_[lo, x[(x > lo) & (x < hi)], hi])
    return float(np.trapezoid(np.interp(grid, x, y), grid)/(hi-lo))


def load_cells(root, expected):
    manifest = json.loads((root/'hashes.json').read_text())
    assert set(manifest) == set(expected)
    assert {p.name for p in root.glob('cell_*.json')} == set(expected)
    rows = []
    for name, mapping in expected.items():
        assert sha(root/name) == manifest[name]
        row = json.loads((root/name).read_text())
        assert (row['task_index'], row['seed'], row['method']) == mapping
        rows.append(row)
    return rows


def mapping(plan, replay=False):
    seeds = [plan['seeds'][plan['replay']['seed_index']]] if replay else plan['seeds']
    methods = plan['replay']['methods'] if replay else plan['methods']
    return {f'cell_{i}_{s}_{m}.json': (i, s, m) for i in range(len(plan['tasks'])) for s in seeds for m in methods}


def analyze(root, replay, plan, output):
    output = Path(output); output.mkdir(exist_ok=False)
    rows = load_cells(root, mapping(plan)); repeated = load_cells(replay, mapping(plan, True))
    checks = [audit(r, plan) for r in rows]
    originals = {(r['task_index'], r['seed'], r['method']): r for r in rows}
    for r in repeated:
        audit(r, plan)
        a = originals[r['task_index'], r['seed'], r['method']]
        assert {k: v for k, v in a.items() if k != 'cpu_seconds'} == {k: v for k, v in r.items() if k != 'cpu_seconds'}
    for ti in range(len(plan['tasks'])):
        for seed in plan['seeds']:
            assert len({r['initial_hash'] for r in rows if r['task_index'] == ti and r['seed'] == seed}) == 1
    metrics, decisions = [], []
    gates = plan['gates']; floor = gates['normalization_floor']
    for ti, task in enumerate(plan['tasks']):
        group = [r for r in rows if r['task_index'] == ti]
        per = {}
        for method in plan['methods']:
            selected = [originals[ti, s, method] for s in plan['seeds']]
            per[method] = np.array([auc(r, 'native', 0, task['training_transitions']) for r in selected])
        strongest = max(CONTROLS, key=lambda m: float(per[m].mean()))
        gain = 100*(per['full']-per[strongest])/max(abs(per[strongest].mean()), floor)
        final, learning, work, byte = [], [], [], []
        for seed in plan['seeds']:
            a, b = originals[ti, seed, 'full'], originals[ti, seed, strongest]
            af, bf = a['curves'][-1]['mean_return'], b['curves'][-1]['mean_return']
            frozen = b['curves'][-1]['frozen_mean_return']
            final.append(100*(af-bf)/max(abs(bf), floor))
            learning.append(100*(bf-frozen)/max(abs(frozen), floor))
            work.append(a['costs']['work_rows']/b['costs']['work_rows'])
            byte.append(a['costs']['logical_bytes']/b['costs']['logical_bytes'])
        common = {}
        for axis in ('messages', 'logical_bytes'):
            lo = max(series(r, axis)[0][0] for r in group)
            hi = min(series(r, axis)[0][-1] for r in group)
            assert hi > lo
            common[axis] = lo, hi
        for r in group:
            metrics.append(dict(task=task['name'], seed=r['seed'], method=r['method'], strongest_control=strongest,
                native_auc=auc(r, 'native', 0, task['training_transitions']),
                message_auc=auc(r, 'messages', *common['messages']),
                logical_byte_auc=auc(r, 'logical_bytes', *common['logical_bytes']),
                final_return=r['curves'][-1]['mean_return'], **r['costs'], cpu_seconds=r['cpu_seconds']))
        passed = dict(native_auc=gain.mean() >= gates['native_auc_gain_percent_min'],
            paired_seed_count=int((gain > 0).sum()) >= gates['positive_seed_count_min'],
            final_return=np.mean(final) >= gates['final_change_percent_min'],
            strong_control_learning=np.mean(learning) >= gates['strong_control_learning_percent_min'],
            work=np.mean(work) <= gates['work_ratio_max'],
            logical_bytes=np.mean(byte) <= gates['logical_byte_ratio_max'])
        decisions.append(dict(task=task['name'], strongest_control=strongest, gates={k: bool(v) for k, v in passed.items()},
            mean_native_auc_gain_percent=float(gain.mean()), positive_seeds=int((gain > 0).sum()),
            paired_native_auc_gain_percent=gain.tolist(), mean_final_change_percent=float(np.mean(final)),
            strongest_control_learning_percent=float(np.mean(learning)),
            mean_work_ratio=float(np.mean(work)), mean_logical_byte_ratio=float(np.mean(byte)),
            common_resource_horizons=common))
    with (output/'metrics.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(metrics[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(metrics)
    go = all(all(d['gates'].values()) for d in decisions)
    dump(output/'summary.json', dict(decision='DEVELOPMENT_GO' if go else 'DEVELOPMENT_NO_GO', tasks=decisions,
        confirmation_required=True, manuscript_admission=False, icml_ready=False,
        inference='three-seed developmental screen, not population significance'))
    dump(output/'integrity.json', dict(originals=len(rows), replays=len(repeated), cells=checks,
                                      semantic_replay=True, initial_policies_matched=True))
    return rows, go


def plots(rows, plan, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import findfont, FontProperties
    findfont(FontProperties(family='Times New Roman'), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    output = Path(output); output.mkdir(exist_ok=False)
    for axis, label in [('native', 'Native training transitions'), ('messages', 'Charged messages'),
                        ('logical_bytes', 'Charged logical bytes')]:
        fig, axs = plt.subplots(1, len(plan['tasks']), figsize=(12, 3.5))
        for ti, ax in enumerate(axs):
            group = [r for r in rows if r['task_index'] == ti]
            lo = max(series(r, axis)[0][0] for r in group)
            hi = min(series(r, axis)[0][-1] for r in group)
            grid = np.linspace(lo, hi, 101)
            for mi, method in enumerate(plan['methods']):
                curves = [np.interp(grid, *series(r, axis)) for r in group if r['method'] == method]
                mean, sd = np.mean(curves, 0), np.std(curves, 0, ddof=1)
                color = plt.get_cmap('tab20')(mi)
                ax.plot(grid, mean, color=color, linewidth=2 if method == 'full' else 1,
                        linestyle='-' if mi < 10 else '--', label=method)
                ax.fill_between(grid, mean-sd, mean+sd, color=color, alpha=.08)
            ax.grid(alpha=.2); ax.set_xlabel(label); ax.set_ylabel('Team return')
            ax.text(.5, -.34, f'({chr(97+ti)}) {plan["tasks"][ti]["name"]}', transform=ax.transAxes,
                    ha='center', va='top')
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=7)
        fig.subplots_adjust(top=.70, bottom=.27, wspace=.3)
        fig.savefig(output/f'return_{axis}.pdf', bbox_inches='tight')
        fig.savefig(output/f'return_{axis}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
