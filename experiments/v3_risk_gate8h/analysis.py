"""All-cell descriptive evidence and paired seed analysis; no promotion gate."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

from core import (POLICIES, LEARNERS, Family, Action, design, deployment_decision,
                  best_action, program_risk)
from guard import dump, sha

KEY = ["seed","family","regime","learner","scale"]


def audit_row(row):
    for name in ("terminal_estimate","terminal_squared_risk","messages_used","environment_used"):
        if not math.isfinite(row[name]):
            raise AssertionError("Nonfinite "+name)
    if row["terminal_squared_risk"]<0 or not math.isclose(row["terminal_estimate"]**2,row["terminal_squared_risk"],abs_tol=1e-14):
        raise AssertionError("Risk/estimate mismatch")
    if row["usable"]!=max(0,row["scheduled"]-row["D"]):
        raise AssertionError("FIFO terminal withholding mismatch")
    pm = row["probe_n"]*(8+row["probe_q"])
    pe = row["probe_n"]*row["probe_b"]
    if row["messages_used"]!=pm+row["scheduled"]*(8+row["q"]):
        raise AssertionError("Message ledger mismatch")
    if row["environment_used"]!=pe+row["scheduled"]*row["b"]:
        raise AssertionError("Environment ledger mismatch")
    if row["messages_used"]>row["message_budget"] or row["environment_used"]>row["environment_budget"]:
        raise AssertionError("Budget exceeded")
    expected_M = min((row["message_budget"]-pm)//(8+row["q"]),(row["environment_budget"]-pe)//row["b"])
    if expected_M!=row["scheduled"] or row["initialization"]!=1. or row["physical_startup_charge"]!=0:
        raise AssertionError("Horizon/initialization mismatch")
    family = Family(int(row["Q"]),float(row["theta_high"]),float(row["lambda_public"]),int(row["D"]))
    public = design(row["learner"],family,int(row["scale"]))
    if row["family"]!=family.key or row["message_budget"]!=public["bm"] or row["environment_budget"]!=public["be"]:
        raise AssertionError("Public family/budget mapping mismatch")
    if row["policy"] not in POLICIES or row["regime"] not in ("low","high"):
        raise AssertionError("Undeclared policy/regime")
    if row["policy"]==POLICIES[-1]:
        theta = family.theta_low if row["regime"]=="low" else family.theta_high
        used,action = False,best_action(row["learner"],public["bm"],public["be"],family,theta)
    else:
        used,action = deployment_decision(row["policy"],public,row["identified"])
    if bool(row["probe_used"])!=used or Action(int(row["q"]),int(row["b"]))!=action:
        raise AssertionError("Deployment role/action separation failed")
    if row["probe_n"]!=(public["probe_n"] if used else 0):
        raise AssertionError("Probe ledger differs from fixed plan")
    if row["policy"]==POLICIES[0]:
        allowance = .1 if row["regime"]=="low" else 0.
        bound = row["certified_population_upper"]
        if not math.isfinite(bound) or bound>row["fallback_population_risk"]+allowance+1e-12:
            raise AssertionError("Deterministic gate shield failed")


def load_rows(directory,check=lambda:None):
    frames = []
    for p in sorted(Path(directory).glob("seed_*.jsonl")):
        check()
        frame = pd.read_json(p,lines=True,precise_float=True)
        frames.append(frame)
    if not frames:
        raise AssertionError("Missing seed parts")
    return pd.concat(frames,ignore_index=True)


def bootstrap(values,rng,repetitions=2000):
    values = np.asarray(values,dtype=float)
    indices = rng.randint(0,len(values),size=(repetitions,len(values)))
    q = np.quantile(values[indices].mean(axis=1),[.025,.975])
    return float(q[0]),float(q[1])


def write_csv(path,frame):
    with Path(path).open("x",encoding="utf-8",newline="") as f:
        frame.to_csv(f,index=False,lineterminator="\n",float_format="%.17g")


def analyze(science,replay,out,protocol,check=lambda:None):
    out = Path(out)
    out.mkdir(exist_ok=False)
    manifest_a = {p.name:sha(p) for p in sorted(Path(science).glob("seed_*.jsonl"))}
    manifest_b = {p.name:sha(p) for p in sorted(Path(replay).glob("seed_*.jsonl"))}
    if manifest_a!=manifest_b:
        raise AssertionError("Full semantic replay differs")
    frame = load_rows(science,check)
    assert len(frame)==196608 and frame.seed.nunique()==128 and len(manifest_a)==128
    assert set(frame.policy)==set(POLICIES) and set(frame.learner)==set(LEARNERS)
    assert frame.family.nunique()==16 and not frame.duplicated(KEY+["policy"]).any()
    assert set(frame.seed)==set(range(20571001,20571129))
    assert (frame.groupby(KEY).size()==6).all()
    _write_tables(frame,out,protocol,manifest_a,check)


def _write_tables(frame,out,protocol,manifest_a,check=lambda:None):
    """Table engine also tested on tiny deterministic fixtures; no native entry point."""
    out = Path(out)
    nseeds,nfamilies = frame.seed.nunique(),frame.family.nunique()
    for i,row in enumerate(frame.to_dict("records")):
        if i%512==0:
            check()
        audit_row(row)
    base = frame[frame.policy=="always_all"][KEY+["terminal_squared_risk"]].rename(columns={"terminal_squared_risk":"fallback_observed_risk"})
    frame = frame.merge(base,on=KEY,validate="many_to_one",sort=False)
    frame["paired_improvement"] = frame.fallback_observed_risk-frame.terminal_squared_risk
    rng = np.random.RandomState(protocol["analysis"]["analysis_rng_seed"])
    cell_keys = ["family","regime","learner","scale","policy"]
    cells = []
    for key,part in frame.groupby(cell_keys,sort=True):
        check()
        assert len(part)==nseeds
        low,high = bootstrap(part.sort_values("seed").paired_improvement.to_numpy(),rng)
        finite_bounds = part.certified_population_upper.dropna()
        cells.append(dict(zip(cell_keys,key), seeds=len(part),
                          mean_risk=float(part.terminal_squared_risk.mean()),
                          fallback_mean_risk=float(part.fallback_observed_risk.mean()),
                          paired_improvement=float(part.paired_improvement.mean()),ci_low=low,ci_high=high,
                          probe_rate=float(part.probe_used.mean()),
                          mean_messages=float(part.messages_used.mean()),mean_environment=float(part.environment_used.mean()),
                          mean_upper=float(finite_bounds.mean()) if len(finite_bounds) else None,
                          upper_minus_observed_mean=(float(finite_bounds.mean()-part.terminal_squared_risk.mean()) if len(finite_bounds) else None)))
    macro = []
    for key,part in frame.groupby(["regime","learner","scale","policy"],sort=True):
        check()
        per_seed = part.groupby("seed").paired_improvement.mean().sort_index()
        low,high = bootstrap(per_seed.to_numpy(),rng)
        macro.append(dict(zip(["regime","learner","scale","policy"],key),
                          family_count=int(nfamilies),seeds=int(nseeds),mean_improvement=float(per_seed.mean()),ci_low=low,ci_high=high))
    assert len(cells)==frame.groupby(cell_keys).ngroups
    assert len(macro)==frame.groupby(["regime","learner","scale","policy"]).ngroups
    probes = frame[frame.policy=="information_only"]
    probes = probes[probes.probe_used].drop_duplicates(["seed","family","regime","learner"])
    directional = []
    for key,part in probes.groupby(["family","regime","learner"],sort=True):
        errors = int(part.direction_error.sum())
        n = len(part)
        low = 0. if errors==0 else float(beta.ppf(.025,errors,n-errors+1))
        high = 1. if errors==n else float(beta.ppf(.975,errors+1,n-errors))
        directional.append(dict(zip(["family","regime","learner"],key),seeds=n,errors=errors,
                                observed_error_rate=errors/n,ci_low=low,ci_high=high,
                                certified_directional_upper=float(part.directional_bound.iloc[0])))
    # Zero-allowance stress reuses recorded branches, not additional trajectories.
    index = frame.set_index(KEY+["policy"])
    stress = []
    for key,part in frame.groupby(KEY,sort=True):
        check()
        decision = bool(part.zero_allowance_admit.iloc[0])
        source = "information_only" if decision else "always_all"
        selected = index.loc[(*key,source)]
        stress.append(dict(zip(KEY,key),source_policy=source,admit=decision,
                           terminal_squared_risk=float(selected.terminal_squared_risk),
                           messages_used=int(selected.messages_used),environment_used=int(selected.environment_used)))
    frame = frame.sort_values(KEY+["policy"],kind="stable")
    write_csv(out/"metrics.csv",frame)
    write_csv(out/"cell_results.csv",pd.DataFrame(cells))
    write_csv(out/"macro_results.csv",pd.DataFrame(macro))
    write_csv(out/"directional_errors.csv",pd.DataFrame(directional))
    write_csv(out/"zero_allowance_stress.csv",pd.DataFrame(stress))
    corrected = pd.DataFrame(cells)
    corrected = corrected[corrected.policy==POLICIES[0]]
    dump(out/"summary.json",dict(status="COMPLETE_DESCRIPTIVE_RESULTS_NOT_AUTOMATIC_MANUSCRIPT_ADMISSION",
                                  original_policy_trajectories=len(frame),full_replay_trajectories=len(frame),
                                  cell_count=len(cells),macro_count=len(macro),policies=list(POLICIES),
                                  corrected_adverse_cell_means=int((corrected.paired_improvement<0).sum()),
                                  corrected_positive_interval_cells=int((corrected.ci_low>0).sum()),
                                  corrected_negative_interval_cells=int((corrected.ci_high<0).sum()),
                                  all_cell_and_cost_tables_retained=True,
                                  confidence_scope="paired seed bootstrap; descriptive cellwise, no multiplicity claim",
                                  negative_or_inconclusive_results_retained=True,manuscript_admission=False))
    dump(out/"integrity.json",dict(full_replay_byte_identical=True,science_parts=manifest_a,
                                    rows=len(frame),seed_count=int(nseeds),all_six_policies=True,
                                    all_ledger_initialization_finiteness_checks=True,
                                    no_true_regime_input_to_deployment_controller=True))


def plots(analysis,out,check=lambda:None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.read_csv(Path(analysis)/"cell_results.csv")
    out = Path(out)
    out.mkdir(exist_ok=False)
    for (learner,regime),part in frame.groupby(["learner","regime"],sort=True):
        check()
        fig,axes = plt.subplots(1,3,figsize=(14,4))
        for policy,g in part.groupby("policy",sort=True):
            means = g.groupby("scale").mean(numeric_only=True)
            for ax,x in zip(axes,(means.index,means.mean_messages,means.mean_environment)):
                ax.plot(x,means.mean_risk,marker="o",label=policy)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(alpha=.25)
        for ax,label in zip(axes,("Public budget scale","Mean charged messages","Mean charged parallel-clock units")):
            ax.set_xlabel(label)
            ax.set_ylabel("Mean terminal squared risk")
        axes[0].legend(fontsize=6)
        fig.suptitle(f"{learner}: {regime}; all 16 families, 128 seeds")
        fig.tight_layout()
        fig.savefig(out/f"risk_{learner}_{regime}.png",dpi=160)
        plt.close(fig)
