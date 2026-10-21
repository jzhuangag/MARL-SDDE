"""Frozen paired-seed analysis, all cells retained; outputs omit runtime metadata."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from model import audit_row
from support import dump,sha

KEYS=['seed','learner','family','scale','regime','error_sign']
ROBUST='conditional_risk_robust_full_catalogue'
FIXED='best_balanced_full_budget_fixed_including_stop'
MARGINAL='posterior_regime_marginal_risk_full_catalogue'
ORACLE='conditional_risk_known_error_evaluation_only'


def load(directory,check=lambda:None):
    rows=[]
    for path in sorted(Path(directory).glob('part_*.jsonl')):
        check()
        with path.open(encoding='utf-8') as stream:
            rows.extend(json.loads(line) for line in stream)
    return pd.DataFrame(rows)


def analyze(science,replay,output,cfg,check=lambda:None):
    output=Path(output);output.mkdir(exist_ok=False)
    frame=load(science,check)
    if len(frame)!=cfg['trajectories_per_phase']:raise AssertionError('Analysis count mismatch')
    if frame.duplicated(KEYS+['policy']).any():raise AssertionError('Duplicate mapping')
    if set(frame.policy)!=set(cfg['policies']):raise AssertionError('Missing policy')
    for i,row in enumerate(frame.to_dict('records')):
        if i%512==0:check()
        audit_row(row)
    cases=frame.groupby(KEYS,sort=True).policy.nunique()
    if not (cases==9).all():raise AssertionError('Incomplete policy group')
    for path in sorted(Path(science).glob('part_*.jsonl')):
        if sha(path)!=sha(Path(replay)/path.name):raise AssertionError('Full semantic replay mismatch')
    deploy=frame[~frame.oracle_evaluation_only]
    invariant=deploy.groupby(['seed','learner','family','scale','regime','policy'])[['q','b','probe_used']].nunique()
    if (invariant>1).any().any():raise AssertionError('Hidden error leaked into deployment')
    frame=frame.sort_values(KEYS+['policy']).reset_index(drop=True)
    frame.to_csv(output/'metrics.csv',index=False,float_format='%.17g',lineterminator='\n')
    cells=frame.groupby(['learner','family','scale','regime','error_sign','policy'],sort=True).agg(
        count=('seed','count'),mean_risk=('terminal_squared_risk','mean'),
        mean_normalized_risk=('normalized_risk','mean'),mean_messages=('messages','mean'),
        mean_clock=('clock','mean'),probe_fraction=('probe_used','mean'),stop_fraction=('stop','mean'))
    cells.to_csv(output/'cell_results.csv',float_format='%.17g',lineterminator='\n')
    pivot=frame.pivot(index=KEYS,columns='policy',values='normalized_risk')
    contrasts=pd.DataFrame({'fixed_minus_robust':pivot[FIXED]-pivot[ROBUST],
        'marginal_minus_robust':pivot[MARGINAL]-pivot[ROBUST],
        'robust_minus_known_error_oracle':pivot[ROBUST]-pivot[ORACLE]})
    design=contrasts.groupby(['learner','family','scale']).mean()
    design.to_csv(output/'design_results.csv',float_format='%.17g',lineterminator='\n')
    seeds=contrasts.groupby('seed').mean()
    if len(seeds)!=cfg['formal_seed_count']:raise AssertionError('Seed count mismatch')
    seeds.to_csv(output/'primary_seed.csv',float_format='%.17g',lineterminator='\n')
    rng=np.random.default_rng(cfg['analysis_seed'])
    indices=rng.integers(0,len(seeds),size=(cfg['bootstrap_repetitions'],len(seeds)))
    estimates={}
    for name in seeds.columns:
        check()
        data=seeds[name].to_numpy();samples=data[indices].mean(axis=1)
        estimates[name]=dict(mean=float(data.mean()),one_sided_995_lower=float(np.quantile(samples,cfg['bootstrap_lower_quantile'])),
            percentile_95=[float(x) for x in np.quantile(samples,[.025,.975])],paired_seeds=len(data))
    good=(estimates['fixed_minus_robust']['one_sided_995_lower']>0 and
          estimates['marginal_minus_robust']['one_sided_995_lower']>0 and
          estimates['fixed_minus_robust']['mean']>=cfg['practical_relative_gain_threshold'])
    summary=dict(decision='POSITIVE_AT_REGISTERED_MODEL_AND_INFORMATION_SCOPE' if good else 'MECHANISM_NOT_CONFIRMED_AT_REGISTERED_THRESHOLD',
        primary=estimates,all_designs=len(design),adverse_design_mean_counts={c:int((design[c]<0).sum()) for c in design.columns[:2]},
        per_learner=contrasts.groupby('learner').mean().to_dict(orient='index'),
        per_budget=contrasts.groupby('scale').mean().to_dict(orient='index'),
        oracle_only_gains_are_not_deployable_evidence=True,
        inference='Paired master-seed percentile bootstrap, approximate coverage; two co-primary one-sided99.5% bounds; secondary contrasts descriptive',
        scope=cfg['scope'],manuscript_admission=False)
    dump(output/'summary.json',summary)
    integrity=dict(rows=len(frame),unique_policy_mappings=True,all_cells_retained=True,full_semantic_replay=True,
        finite_positive_denominators=True,paid_ledgers=True,hidden_sign_action_invariance=True,
        science_payloads={p.name:sha(p) for p in sorted(Path(science).glob('part_*.jsonl'))},
        replay_payloads={p.name:sha(p) for p in sorted(Path(replay).glob('part_*.jsonl'))},
        analysis_seed=cfg['analysis_seed'],bootstrap_repetitions=cfg['bootstrap_repetitions'],
        initial_error_signs=sorted(frame.error_sign.unique().tolist()),
        analyses_are_separate_executions_not_independent_implementations=True)
    dump(output/'integrity.json',integrity)
    frame['message_fraction']=frame.messages/frame.message_budget
    frame['clock_fraction']=frame.clock/frame.clock_budget
    ledger=frame.groupby(['learner','scale','policy']).agg(mean_messages=('messages','mean'),mean_clock=('clock','mean'),
        mean_message_fraction=('message_fraction','mean'),mean_clock_fraction=('clock_fraction','mean'),
        mean_applied=('applied','mean'),mean_scheduled=('M','mean'),stop_fraction=('stop','mean'))
    ledger.to_csv(output/'ledger_summary.csv',float_format='%.17g',lineterminator='\n')
    check()


def plots(analysis,output,cfg,check=lambda:None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9})
    output=Path(output);output.mkdir(exist_ok=False)
    frame=pd.read_csv(Path(analysis)/'metrics.csv')
    labels=['Robust conditional','Known-error oracle','Posterior marginal','Legacy risk gate',
            'Information only','All agents','Fixed q=2','Best balanced fixed','Known-regime oracle']
    specs=[('risk_budget.png','scale','Nominal budget scale'),('risk_messages.png','messages','Mean charged messages'),
           ('risk_clock.png','clock','Mean charged clock units'),('risk_learner_budget.png','scale','Nominal budget scale')]
    for name,x,xlabel in specs:
        check();fig,axes=plt.subplots(1,2,figsize=(8,4.4))
        for ax,learner in zip(axes,cfg['learners']):
            subset=frame[frame.learner==learner]
            for i,policy in enumerate(cfg['policies']):
                values=subset[subset.policy==policy].groupby('scale').agg(
                    risk=('normalized_risk','mean'),messages=('messages','mean'),clock=('clock','mean'))
                if name=='risk_learner_budget.png':
                    values['risk']=subset[subset.policy==policy].groupby(['scale','error_sign']).normalized_risk.mean().groupby('scale').max()
                xx=values.index.to_numpy() if x=='scale' else values[x].to_numpy()
                ax.plot(xx,values.risk.to_numpy(),marker='o',ms=3,lw=1.15,
                        linestyle='--' if 'evaluation_only' in policy else '-',label=labels[i])
            if x=='scale':ax.set_xscale('log',base=2);ax.set_xticks(cfg['public_grid']['scales']);ax.set_xticklabels(cfg['public_grid']['scales'])
            ax.set_xlabel(xlabel);ax.set_ylabel('Risk / best balanced fixed population risk')
            ax.set_title(learner.replace('_',' '),y=-.35);ax.set_box_aspect(1);ax.grid(alpha=.2)
        handles,legend_labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,loc='upper center',ncol=3,frameon=False,fontsize=8)
        fig.subplots_adjust(top=.70,bottom=.24,left=.08,right=.98,wspace=.32)
        fig.savefig(output/name,dpi=170,metadata={'Software':'Frozen conditional2h analysis'})
        plt.close(fig)
