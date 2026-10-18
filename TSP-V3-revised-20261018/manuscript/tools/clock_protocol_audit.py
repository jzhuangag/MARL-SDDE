"""Source/ledger audit only: no environment, trajectory, optimizer or native rerun."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
CORE = ROOT / 'experiments/dependence_delay_linear'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    source = (CORE / 'run_adaptation_cost_pilot.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    names = {'available_updates', 'scheduled_updates'}
    functions = [ast.get_source_segment(source, node) for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    # Execute only these two extracted integer-accounting functions. No imports
    # from the scientific package (and therefore no sampler/JIT/cache writes).
    exec(compile('from __future__ import annotations\n' + '\n\n'.join(functions),
                 '<frozen-integer-accounting-only>', 'exec'), namespace)
    checked = 0
    for q in [1, 2, 8, 32]:
        for b in [1, 2, 4, 8]:
            for msg in [-1, 0, 20, 64, 200]:
                for env in [-1, 0, 20, 64, 200]:
                    for lag in [0, 1, 2, 8]:
                        scheduled = max(0, min(msg // (4 + q), env // b))
                        expected = max(0, scheduled - lag)
                        action = SimpleNamespace(q=q, b=b)
                        assert namespace['scheduled_updates'](action, msg, env, 4) == scheduled
                        assert namespace['available_updates'](action, msg, env, 4, lag) == expected
                        checked += 1
    action_path = CORE / 'results/affine_finite_time_certificate/selected_actions.csv'
    with action_path.open(newline='') as stream:
        actions = list(csv.DictReader(stream))
    maximum_error = 0.0
    for row in actions:
        q, b, n = (int(row[name]) for name in ['num_agents', 'gap', 'updates'])
        cost, budget = int(row['update_cost']), int(row['resource_budget'])
        assert cost == 4 + q + b and n == budget // cost
        assert (n - 1) * b + 1 <= n * b  # Last transition endpoint, first start0.
        assert n * cost <= budget
        eta, mu, curvature, lip, delta, omega, noise, tau = (
            float(row[key]) for key in ['eta', 'effective_monotonicity', 'curvature',
                                        'lipschitz', 'joint_delta', 'omega',
                                        'innovation_bound', 'rms_delay'])
        a = 1 - eta * mu + eta**2 * (2 * curvature + 4 * lip**2 * delta)
        beta = 4 * eta * noise**2 * delta**2 / mu + eta**2 * (2 * omega + 4 * noise**2 * delta)
        h = 2 * eta**4 * lip**4 * tau**2
        g = 2 * eta**4 * lip**2 * noise**2 * tau**2
        young = math.sqrt(h / a) if h else 0
        contraction = (1 + young) * a + (1 + 1 / young) * h if h else a
        forcing = (1 + young) * beta + (1 + 1 / young) * g if h else beta
        residual = forcing / (1 - contraction)
        blocks = n // (2 * int(row['maximum_actual_delay']) + 1)
        bound = residual + contraction**blocks * max(float(row['initial_error']) - residual, 0)
        for key, value in [('a_delta', a), ('beta_delta', beta), ('contraction', contraction),
                           ('forcing', forcing), ('finite_time_bound', bound)]:
            error = abs(float(row[key]) - value)
            maximum_error = max(maximum_error, error)
            assert math.isclose(float(row[key]), value, rel_tol=1e-9, abs_tol=1e-12), (key, error)
    paths = [CORE / name for name in ['run_adaptation_cost_pilot.py', 'run_exp016b_pilot.py',
             't018_static_scan.py', 'affine_markov_certificate.py', 'multistate_certificate.py']]
    paths += [ROOT / 'TSP/experiments/run_convergence_curves.py', action_path]
    return dict(
        checked_on='2026-10-18', native_execution=False, outcomes_ranked=False,
        source_and_evidence_sha256={str(path.relative_to(ROOT)): digest(path) for path in paths},
        integer_clock_cases=checked, all_integer_clock_cases_match=True,
        saved_affine_actions_checked=len(actions), scalar_cost_and_endpoint_checks=True,
        saved_coefficient_and_certificate_max_absolute_error=maximum_error,
        clock_conventions=dict(
            affine_certificate_and_curves='D=deterministic parameter lag; u=ell=0; K=floor(B/(h+q+b)); first transition start0; last endpoint (K-1)*b+1<=K*b.',
            exp016b='u=D retained terminal updates, ell=0; K=min(floor(Bmsg/(h+q)),floor(Benv/b)); all K charged, K-D usable; Gaussian observation indices start_environment+b*(1..K).',
            td_mixing='P is the transition-pair kernel. Exact TV uses P_pair**b, combined across q+1 common/private sources by union bound; fresh source-choice masks do not increase TV.'),
        saved_admission_scope='Prespecified B_value thresholds from t018, not retrospectively replaced by the repaired theorem gate. Exact-risk selector uses a stationary mean proxy; empirical Gaussian uses a pseudo-observation and FIFO layer reports an averaged parameter.',
        empirical_conditional_certificate_admission=False,
        conditional_theorem_scope='Applies when all selected actions have valid post-probe conditional upper certificates; the new theorem uses actual selected-action certificates and needs no global scalar-solver claim.',
        old_files_modified=False,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = audit()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
