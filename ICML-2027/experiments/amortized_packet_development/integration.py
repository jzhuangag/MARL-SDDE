"""Amortized native delayed-data PPO receipt loop; no stale worker optimizer is imported.

Logical transport ticks are independent of controller choices. They are not
measured hardware parallelism. Evaluation and unchanged-policy controls are
outcome-free in qualification; they never enter the controller.
"""
import copy
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS / "causal_recycle"))
sys.path.insert(0, str(EXPERIMENTS))
from native import base, collect, take, probabilities
from packet import digest, tensor_bytes
from current_packet_value_control import ProbeOption
from amortized_packet_value import amortized_proposals, eligible_probe_bytes
from amortized_packet_native.controller import AmortizedController
from finite_horizon_return_control import MonteCarloPotential
from shared_reference_return import contrasts


def policy_hash(actors):
    return digest([p for a in actors for p in a.actor.parameters()])


def optimizer_hash(actor):
    state = actor.actor_optimizer.state_dict()
    h = hashlib.sha256(json.dumps(state["param_groups"], sort_keys=True).encode())
    for key, value in sorted(state["state"].items()):
        h.update(str(key).encode())
        for name, item in sorted(value.items()):
            h.update(name.encode())
            h.update(item.detach().cpu().numpy().tobytes() if torch.is_tensor(item) else str(item).encode())
    return h.hexdigest()


def array_hash(arrays):
    h = hashlib.sha256()
    for x in arrays:
        h.update(str((x.dtype.str, x.shape)).encode()); h.update(x.tobytes())
    return h.hexdigest()


def make_program(data, i):
    # Only a local actor's records, not every actor's full trajectory.
    arrays = [np.array(x, copy=True) for x in
              (data["obs"][i], data["actions"][i], data["logs"][i], data["advantages"])]
    for x in arrays:
        x.setflags(write=False)
    return tuple(arrays)


class ProgramNetwork:
    def __init__(self, seed, n, plan):
        self.seed, self.n, self.plan = seed, n, plan
        self.jobs, self.events, self.used = [], [], set()
        self.versions, self.ordinals = [0] * n, [0] * n
        self.reserved = self.transmitted = 0

    def launch(self, frame, actor, source_versions, critic_version, arrays):
        ordinal = self.ordinals[actor]; self.ordinals[actor] += 1
        word = hashlib.sha256(f"{self.seed}:{actor}:{ordinal}".encode()).digest()
        delay = self.plan["delay_min"] + int.from_bytes(word[:8], "big") % (
            self.plan["delay_max"] - self.plan["delay_min"] + 1)
        job = dict(id=len(self.jobs), agent=actor, ordinal=ordinal, launch=frame,
                   due=frame + delay, source_versions=list(source_versions), critic_version=critic_version,
                   arrays=arrays, bytes=sum(a.nbytes for a in arrays), data_hash=array_hash(arrays))
        self.jobs.append(job)
        self.events.append({k: v for k, v in job.items() if k != "arrays"})
        self.reserved += job["bytes"]

    def receive(self, frame):
        arrived = [j for j in self.jobs if j["id"] not in self.used and j["due"] <= frame]
        ready = {}
        for job in arrived:
            assert array_hash(job["arrays"]) == job["data_hash"]
            self.used.add(job["id"]); self.transmitted += job["bytes"]
            i = job["agent"]
            if i not in ready or job["ordinal"] > ready[i]["ordinal"]:
                ready[i] = job
        return ready, arrived

    def commit(self, action):
        self.versions = [v + int(a != 0) for v, a in zip(self.versions, action)]

    def drain(self):
        if self.jobs:
            self.receive(max(j["due"] for j in self.jobs))
        assert self.reserved == self.transmitted and len(self.used) == len(self.jobs)


def proposal(actor, arrays, alpha, epochs, *, factor=None, budget=lambda: None):
    """Recompute PPO loss/gradient at every server-owned counterfactual epoch."""
    before_policy, before_optimizer = policy_hash([actor]), optimizer_hash(actor)
    candidate = copy.deepcopy(actor)
    obs, actions, logs, advantages = [np.array(x, copy=True) for x in arrays]
    count = len(obs); hidden = actor.actor.hidden_sizes[-1]
    zeros = np.zeros((count, 1, hidden), np.float32); ones = np.ones((count, 1), np.float32)
    sample = (obs, zeros, actions, ones, ones, logs, advantages, None,
              ones if factor is None else factor)
    rates = [g["lr"] for g in candidate.actor_optimizer.param_groups]
    for group, lr in zip(candidate.actor_optimizer.param_groups, rates):
        group["lr"] = lr * alpha
    trace = []
    for epoch in range(epochs):
        budget()
        policy_before = policy_hash([candidate])
        loss, entropy, norm, _ = candidate.update(sample)
        assert all(math.isfinite(float(x)) for x in (loss.detach(), entropy.detach(), norm))
        gradients = [p.grad for p in candidate.actor.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(x).all() for x in gradients)
        trace.append(dict(epoch=epoch, policy_before=policy_before,
                          policy_after=policy_hash([candidate]), gradient_hash=digest(gradients)))
    for group, lr in zip(candidate.actor_optimizer.param_groups, rates):
        group["lr"] = lr
    assert policy_hash([actor]) == before_policy and optimizer_hash(actor) == before_optimizer
    return candidate, dict(alpha=alpha, source_server_policy=before_policy,
        source_server_optimizer=before_optimizer, candidate_policy=policy_hash([candidate]),
        candidate_optimizer=optimizer_hash(candidate), gradient_trace=trace,
        rows=count * epochs, epochs=epochs, copied_program_bytes=sum(a.nbytes for a in arrays))


def menus(actors, variants):
    actions, hashes, seen = [], [], set()
    choices = [(0., *sorted(variants.get(i, {}))) for i in range(len(actors))]
    for action in itertools.product(*choices):
        policies = [variants[i][a] if a else actor for i, (a, actor) in enumerate(zip(action, actors))]
        key = policy_hash(policies)
        if key not in seen:
            seen.add(key); actions.append(action); hashes.append(key)
    assert actions[0] == (0.,) * len(actors)
    return actions, hashes


def feedback(actors, variants, actions, probe, potential):
    m, h = probe["rewards"].shape
    local, forwards = [], 0
    for i in range(len(actors)):
        values = {0.: np.zeros((m, h))}
        if not variants.get(i):
            local.append(values); continue
        a = probe["actions"][i].reshape(-1)
        reference = probe["logs"][i].reshape(-1)
        for alpha, candidate in variants.get(i, {}).items():
            p = probabilities(candidate, probe["obs"][i]); forwards += m * h
            values[alpha] = (np.log(p[np.arange(len(p)), a]) - reference).reshape(m, h)
        local.append(values)
    values = np.zeros((m, h)) if potential is None else potential.predict(probe["states"])
    samples, max_weight = [], 1.
    for action in actions:
        lr = sum(local[i][a] for i, a in enumerate(action))
        score, weight = contrasts(lr, probe["rewards"], values)
        samples.append(score["td_control_variate"]); max_weight = max(max_weight, float(weight.max()))
    result = np.stack(samples, axis=1)
    assert np.isfinite(result).all() and np.all(result[:, 0] == 0)
    return result, forwards, max_weight


def run_cell(task, ti, seed, method, plan, budget):
    p = base.config(); p.update(episode_length=plan["horizon"],
        critic_optimizer_steps_per_block=plan["critic_epochs"])
    actors, critic, normalizer = base.initialize(task, seed, p)
    lr_multiplier = .5 if method == "fresh_happo_half" else 2. if method == "fresh_happo_double" else 1.
    for actor in actors:
        for group in actor.actor_optimizer.param_groups:
            group["lr"] *= lr_multiplier
    initial_actors = copy.deepcopy(actors)
    n, b, h, frames = len(actors), plan["bank_episodes"], plan["horizon"], plan["frames"]
    assert all(a.ppo_epoch == plan["actor_epochs"] and a.actor_num_mini_batch == 1 for a in actors)
    network = ProgramNetwork(seed, n, plan)
    sizes = [tensor_bytes(list(a.actor.parameters())) for a in actors]
    # Shape metadata is obtained from the native env, never from a return.
    env = base.make_env(task, h); ob, _ = env.reset(seed=seed)
    dims = [len(ob[k]) for k in env.possible_agents]; state_dim = len(env.state()); env.close()
    local_probe_per_row = sum(d * 4 + 8 + 8 for d in dims)
    program_per_round = (local_probe_per_row + 4 * n) * b * h
    public_per_round = (state_dim * 4 + 8) * b * h
    advantage_per_round = n * b * h * 4
    byte_scale = program_per_round + public_per_round + advantage_per_round + sum(sizes)
    # Fixed work-row proxy, not equal FLOPs or measured CPU time.
    compute_scale = (3*n + 1 + plan["actor_epochs"] * n + plan["critic_epochs"]) * b * h
    geometries = ["full_vector"] if method == "full_trace" else plan["geometries"]
    option_specs = [(m,lam,g) for m in plan["probe_episodes"] for lam in plan["young_scales"] for g in geometries]
    targets = (1e12,1e12) if method == "no_price" else (plan["communication_target"],plan["compute_target"])
    controller = AmortizedController(
        [ProbeOption(m, lam) for m,lam,g in option_specs], [g for m,lam,g in option_specs],
        gain_scale=plan["gain_scale"], value_weight=math.sqrt(frames), meta_step=plan["meta_step"],
        resource_targets=targets)
    rng = np.random.default_rng(base.seed_for(seed, ti, 0, plan["controller_role"], 0))
    history, ledger, evaluation, train_ids, eval_ids = [], [], [], [], []
    totals = {k: 0 for k in ("training_transitions", "evaluation_transitions", "sampling_rows",
        "evaluation_rows", "critic_value_rows", "critic_backward_rows", "actor_backward_rows",
        "actor_adam_calls", "source_backward_rows", "gradient_bytes", "gradient_messages",
        "proposal_copy_bytes", "score_rows", "potential_fit_rows", "potential_predict_rows",
        "factor_forward_rows", "baseline_score_rows", "diagnostic_score_rows", "diagnostic_potential_rows", "diagnostic_probe_bytes", "diagnostic_probe_messages", "state_copy_bytes", "interpolation_entries", "geometry_entries", "public_bytes", "public_messages", "advantage_bytes", "advantage_messages",
        "program_bytes", "program_messages", "probe_bytes", "probe_messages", "policy_bytes", "policy_messages")}
    initial_hash = policy_hash(actors)

    def evaluate(frame):
        budget()
        trained = collect(actors, task, ti, seed, frame, plan["evaluation_role"], plan["evaluation_episodes"], h)
        frozen = collect(initial_actors, task, ti, seed, frame, plan["evaluation_role"], plan["evaluation_episodes"], h)
        assert trained["identities"] == frozen["identities"]
        eval_ids.extend(trained["identities"])
        totals["evaluation_transitions"] += 2 * plan["evaluation_episodes"] * h
        totals["evaluation_rows"] += 2 * n * plan["evaluation_episodes"] * h
        evaluation.append(dict(frame=frame, identities=trained["identities"],
            trained_episode_returns=trained["rewards"].sum(axis=1).tolist(),
            frozen_episode_returns=frozen["rewards"].sum(axis=1).tolist(),
            frozen_policy_hash=policy_hash(initial_actors), shared_action_uniforms=True))

    evaluate(0)
    for t in range(frames):
        budget(); full = method in ("current_value","full_trace","no_price"); fresh = method.startswith("fresh_happo") or method == "fresh_parallel"
        source_versions = list(network.versions)
        server_hashes = [optimizer_hash(a) for a in actors]
        ready, arrived = ({}, []) if fresh else network.receive(t)
        variants, proposals = {}, []; baseline_rows = 0; baseline_actions = {}
        if not fresh:
            for i, job in sorted(ready.items()):
                age = sum(v-s for v,s in zip(source_versions,job["source_versions"]))
                line = plan["alphas"] if full or method == "online_kl" else sorted(set((1.,1./(1.+age)))) if method == "age_clip" else (1.,)
                variants[i], amortized = amortized_proposals(actors[i], job["arrays"],
                    line, plan["actor_epochs"], budget=budget)
                baseline_actions[i] = 1./(1.+age) if method == "age_clip" else 1.
                if method == "online_kl":
                    obs = job["arrays"][0]; old = probabilities(actors[i],obs); baseline_rows += len(obs)
                    feasible = []
                    for alpha,candidate in variants[i].items():
                        pp = probabilities(candidate,obs); baseline_rows += len(obs)
                        kl = float((old*(np.log(old)-np.log(pp))).sum(axis=1).mean())
                        if kl <= plan["baseline_kl_target"]:
                            feasible.append(alpha)
                    baseline_actions[i] = max(feasible,default=0.)
                trace = amortized["unit_proposal"]
                trace.update(agent=i, packet_id=job["id"], amortized={k:v for k,v in amortized.items() if k != "unit_proposal"})
                proposals.append(trace)
                assert array_hash(job["arrays"]) == job["data_hash"]
        actions, hashes = menus(actors, variants)
        eligible = sorted(ready)
        coefficients = [[a[i] for i in eligible] for a in actions]
        basis_indices = [actions.index(tuple(1. if j == i else 0. for j in range(n))) for i in eligible] if full else []
        potential, fit_rows = None, 0
        if full and history and ready:
            potential = MonteCarloPotential(h, plan["potential_ridge"]).fit(history[-plan["potential_history"]:])
            fit_rows = sum(len(x["states"]) for x in history[-plan["potential_history"]:])
        proposal_rows = sum(x["rows"] for x in proposals)
        sunk_bytes = program_per_round + public_per_round + advantage_per_round
        state_copies = sum(x["amortized"]["modeled_state_tensor_copy_bytes"] for x in proposals)
        interpolation = sum(x["amortized"]["interpolation_parameter_entries"] for x in proposals)
        program_copies = program_per_round + sum(x["copied_program_bytes"] for x in proposals)
        copy_rows = (state_copies+program_copies)/plan["copy_bytes_per_work_unit"] + interpolation/plan["scalar_entries_per_work_unit"]
        dm = plan["diagnostic_episodes"] if full else 0
        diag_potential = dm*h if potential is not None else 0
        common_rows = (n + 1 + plan["critic_epochs"]) * b * h + proposal_rows + fit_rows + copy_rows + baseline_rows + diag_potential
        candidate_costs = [((sum(sizes[i] for i,a in enumerate(action) if a)
                            + eligible_probe_bytes(dims,[i for i,a in enumerate(action) if a],dm,h))/byte_scale,
                            sum(a != 0 for a in action)*dm*h/compute_scale) for action in actions]
        probe_costs = [(eligible_probe_bytes(dims,eligible,m,h) / byte_scale,
                        ((len(ready)*len(plan["alphas"])+int(potential is not None))*m*h
                         + m*len(actions)*(len(eligible)+1)/plan["scalar_entries_per_work_unit"]) / compute_scale)
                       for m,lam,g in option_specs]
        m, opened = 0, None
        if full:
            opened = controller.begin(coefficients=coefficients, basis_indices=basis_indices,
                candidate_hashes=hashes, candidate_costs=candidate_costs,
                probe_costs=probe_costs, sunk_cost=(sunk_bytes / byte_scale, common_rows / compute_scale), rng=rng)
            m = opened["episodes"]
        before_bank = policy_hash(actors)
        # Full private actor records become visible only via a paid probe slice
        # or a later eligible program. Public state/reward serves the critic.
        data = collect(actors, task, ti, seed, t, plan["training_role"], b, h)
        train_ids.extend(data["identities"])
        assert policy_hash(actors) == before_bank and [optimizer_hash(a) for a in actors] == server_hashes
        score_rows = potential_rows = factor_rows = probe_bytes = 0
        detail = {}; max_weight = 1.; geometry_entries = 0; diagnostic = None; diag_rows = diag_bytes = diag_messages = 0
        if full:
            samples, score_rows, max_weight = feedback(actors, variants, actions, take(data, m, h), potential)
            potential_rows = m * h if potential is not None else 0
            selected = controller.select(samples, candidate_hashes=hashes)
            action = actions[selected["candidate_index"]]
            probe_bytes = eligible_probe_bytes(dims,eligible,m,h)
            geometry_entries = m*len(actions)*(len(eligible)+1)
            assert m <= b-dm
            tail = {"obs":[x[-dm*h:] for x in data["obs"]], "actions":[x[-dm*h:] for x in data["actions"]],
                    "logs":[x[-dm*h:] for x in data["logs"]], "states":data["states"][-dm*h:],
                    "rewards":data["rewards"][-dm:],"identities":data["identities"][-dm:]}
            selected_variants = {i:{a:variants[i][a]} for i,a in enumerate(action) if a}
            heldout,diag_rows,diag_weight = feedback(actors,selected_variants,[(0.,)*n,action],tail,potential)
            active = sorted(selected_variants)
            diag_bytes = eligible_probe_bytes(dims,active,dm,h); diag_messages = len(active)
            diagnostic = dict(episode_ids=tail["identities"], selection_episode_ids=data["identities"][:m],
                selected_contrasts=heldout[:,1].tolist(), selected_training_mean=float(samples[:,selected["candidate_index"]].mean()),
                selected_before_validation=True, used_by_controller=False, weight_max=diag_weight)
            assert set(diagnostic["episode_ids"]).isdisjoint(diagnostic["selection_episode_ids"])
            for i, a in enumerate(action):
                if a:
                    actors[i] = variants[i][a]
            assert policy_hash(actors) == selected["candidate_hash"]
            actual_charge = ((sunk_bytes + probe_bytes + diag_bytes + sum(sizes[i] for i, a in enumerate(action) if a)) / byte_scale,
                             (common_rows + score_rows + potential_rows + diag_rows + geometry_entries/plan["scalar_entries_per_work_unit"]) / compute_scale)
            detail = controller.close(committed_hash=policy_hash(actors), actual_total_charge=actual_charge)
        elif not fresh:
            action = tuple(baseline_actions.get(i,0.) for i in range(n))
            for i, a in enumerate(action):
                if a:
                    actors[i] = variants[i][a]
        else:
            action = (1.,) * n
        base.add_advantages(data, critic, normalizer, p)
        programs = [make_program(data, i) for i in range(n)]
        assert sum(sum(a.nbytes for a in arrays) for arrays in programs) == program_per_round
        if fresh:
            factor = np.ones_like(data["advantages"])
            for i in range(n):
                old = None if method == "fresh_parallel" else base.actor_logs(actors[i], data, i, p)
                candidate, trace = proposal(actors[i], programs[i], 1., plan["actor_epochs"], factor=None if method == "fresh_parallel" else factor, budget=budget)
                trace.update(agent=i, packet_id=None)
                tensor_copies = tensor_bytes(list(actors[i].actor.state_dict().values())) + sum(v.numel()*v.element_size() for st in actors[i].actor_optimizer.state.values() for v in st.values() if torch.is_tensor(v))
                trace["amortized"] = dict(modeled_state_tensor_copy_bytes=tensor_copies, interpolation_parameter_entries=0)
                state_copies += tensor_copies
                proposals.append(trace)
                actors[i] = candidate
                if method != "fresh_parallel":
                    factor *= np.exp(base.actor_logs(actors[i], data, i, p) - old)
                    factor_rows += 2 * b * h
        else:
            for i, arrays in enumerate(programs):
                network.launch(t, i, source_versions, t, arrays)
        network.commit(action)
        base.train_critic(critic, normalizer, data, p)
        history.append({"states": data["states"], "rewards": data["rewards"]})
        if len(history) > plan["potential_history"]:
            history.pop(0)
        step = dict(training_transitions=b*h, evaluation_transitions=0, sampling_rows=n*b*h,
            evaluation_rows=0, critic_value_rows=b*h, critic_backward_rows=plan["critic_epochs"]*b*h,
            actor_backward_rows=sum(x["rows"] for x in proposals), actor_adam_calls=len(proposals)*plan["actor_epochs"],
            source_backward_rows=0, gradient_bytes=0, gradient_messages=0,
            proposal_copy_bytes=sum(x["copied_program_bytes"] for x in proposals),
            score_rows=score_rows, potential_fit_rows=fit_rows, potential_predict_rows=potential_rows,
            factor_forward_rows=factor_rows, baseline_score_rows=baseline_rows,
            diagnostic_score_rows=diag_rows, diagnostic_potential_rows=diag_potential,
            diagnostic_probe_bytes=diag_bytes, diagnostic_probe_messages=diag_messages, state_copy_bytes=state_copies, interpolation_entries=interpolation,
            geometry_entries=geometry_entries, public_bytes=public_per_round, public_messages=1,
            advantage_bytes=advantage_per_round, advantage_messages=n,
            program_bytes=program_per_round, program_messages=n, probe_bytes=probe_bytes,
            probe_messages=len(eligible) if full else 0, policy_bytes=sum(sizes[i] for i,a in enumerate(action) if a),
            policy_messages=sum(a != 0 for a in action))
        for k, v in step.items():
            totals[k] += v
        ledger.append(dict(frame=t, source_versions=source_versions, versions_after=list(network.versions),
            source_policy=before_bank, source_optimizer_hashes=server_hashes,
            committed_policy=policy_hash(actors), committed_optimizer_hashes=[optimizer_hash(a) for a in actors],
            action=list(action), ready=[j["id"] for _,j in sorted(ready.items())],
            arrived=[j["id"] for j in arrived], age=[sum(v-s for v,s in zip(source_versions,j["source_versions"])) for j in arrived],
            candidate_hashes=hashes, candidate_actions=[list(a) for a in actions], proposals=proposals,
            probe_episodes=m, probe_before_bank=full, candidates_before_bank=True,
            receipt_after_bank=True, selected_score_unbiased=False, current_evaluation_used=False,
            weight_max=max_weight, control=detail, coefficients=coefficients, basis_indices=basis_indices,
            opened_meta=opened, diagnostic=diagnostic, step_charges=step, totals=dict(totals),
            program_hashes=[array_hash(a) for a in programs]))
        if (t+1) % plan["evaluation_every"] == 0 or t+1 == frames:
            evaluate(t+1)
    if not (method.startswith("fresh_happo") or method == "fresh_parallel"):
        network.drain()
    assert set(train_ids).isdisjoint(eval_ids) and len(set(train_ids)) == len(train_ids)
    return dict(task=task["name"], seed=seed, method=method, frames=frames,
        initial_hash=initial_hash, final_hash=policy_hash(actors), frozen_final_hash=policy_hash(initial_actors),
        actor_parameter_bytes=sizes, local_observation_dimensions=dims, state_dimension=state_dim,
        communication_scale=byte_scale, compute_scale=compute_scale, ledger=ledger,
        evaluation=evaluation, charges=totals, program_events=network.events,
        reserved_program_bytes=network.reserved, transmitted_program_bytes=network.transmitted,
        training_episode_ids=train_ids, evaluation_episode_ids=eval_ids,
        evaluation_used_by_controller=False, logical_transport_only=True,
        gradient_payload_replaced_by_data_program=True,
        initial_state_copy_bytes=sum(tensor_bytes(list(a.actor.state_dict().values())) for a in initial_actors),
        amortized_native=True, effective_resource_targets=list(targets), learning_rate_multiplier=lr_multiplier)
