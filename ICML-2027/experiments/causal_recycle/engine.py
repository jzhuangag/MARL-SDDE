"""Complete causal learning loop, not an estimator-only experiment."""
import time
import numpy as np
from native import base,collect,take,menu_feedback
from packet import NetworkPackets,source_gradient,adam_variant,tensor_bytes,digest
from control import RecycleController
from finite_horizon_return_control import MonteCarloPotential


def run_cell(task,ti,seed,method,plan,*,qualification=False):
    p=base.config();p.update(episode_length=plan["horizon"],episodes_per_training_block=plan["bank_episodes"],
        actor_optimizer_steps_per_packet=1,critic_optimizer_steps_per_block=plan["critic_steps"])
    actors,critic,normalizer=base.initialize(task,seed,p);n=len(actors);h=p["episode_length"];b=plan["bank_episodes"]
    frames=plan["qualification_frames"] if qualification else plan["frames"]
    initial=digest([v for a in actors for v in a.actor.parameters()]);net=NetworkPackets(seed,n)
    policy_bytes=[tensor_bytes(list(a.actor.parameters())) for a in actors];full_bytes=sum(policy_bytes)
    controller=RecycleController(n,batch_episodes=b,horizon=h,frames=plan["frames"],
        probes=tuple(plan["probes"]),outbound_target=plan["outbound_target"],forward_target=plan["forward_target"],
        eta_scale=plan["eta_scale"],meta_scale=plan["meta_scale"],use_dual=method!="recycle_no_dual")
    actions=controller.actions;history=[];curves=[];ledger=[];all_training_ids=[];all_eval_ids=[]
    rng=np.random.default_rng(base.seed_for(seed,ti,0,891,0));started=time.process_time()
    charges={k:0 for k in ("training_transitions","evaluation_transitions","sampling_policy_rows","source_backward_rows",
        "critic_backward_rows","extra_score_policy_rows","extra_potential_rows","proposal_adam_steps","committed_adam_steps",
        "source_actor_steps","critic_steps","gradient_bytes","policy_broadcast_bytes","trajectory_bytes","advantage_bytes",
        "gradient_messages","policy_messages","trajectory_messages","advantage_messages","evaluation_policy_rows")}

    def evaluate(frame):
        count=plan["qualification_eval_episodes"] if qualification else plan["evaluation_episodes"]
        data=collect(actors,task,ti,seed,frame,893,count,h)
        all_eval_ids.extend(data["identities"]);charges["evaluation_transitions"]+=count*h
        charges["evaluation_policy_rows"]+=n*count*h
        curves.append({"frame":frame,"training_transitions":charges["training_transitions"],
            "logical_communication_bytes":charges["gradient_bytes"]+charges["policy_broadcast_bytes"]+charges["trajectory_bytes"]+charges["advantage_bytes"],
            "return":float(data["rewards"].sum(axis=1).mean())})

    evaluate(0)
    for t in range(frames):
        fresh=method.startswith("happo_");full=method.startswith("recycle")
        ready,arrived=({},[]) if fresh else net.receive_latest(t)
        variants={};proposal_steps=0
        for i,job in ready.items():
            needed=(.5,1.) if full else ((1.,) if method=="async_latest" else ())
            variants[i]={alpha:adam_variant(actors[i],job["gradient"],alpha) for alpha in needed}
            proposal_steps+=len(needed)
        effective=[tuple(alpha if i in ready else 0. for i,alpha in enumerate(a)) for a in actions]
        costs=np.array([sum(policy_bytes[i] for i,alpha in enumerate(a) if alpha>0)/full_bytes for a in effective])
        potential=None;hints=np.zeros(len(actions));hint_rows=potential_rows=0;weight_max=1.
        if full and history and ready:
            potential=MonteCarloPotential(h,plan["potential_ridge"]).fit(history[-plan["potential_history_banks"]:])
            potential_rows=sum(len(x["states"]) for x in history[-plan["potential_history_banks"]:])
            if method!="recycle_no_hint":
                old=take(history[-1],plan["hint_episodes"],h)
                hint_sample,hint_rows,_=menu_feedback(actors,variants,actions,old,potential,current_bank=False)
                hints=np.clip(hint_sample.mean(axis=0)/h,-1.,1.)
                potential_rows+=len(old["states"])
        denominator=2*n*b*h
        # Count only forwards that actually run; the availability fraction is predictable.
        availability=len(ready)/n
        if full:
            # The probe cost catalogue must equal the actual cached forward work.
            ai,m=controller.decide(hints,costs,hint_rows/denominator,rng,availability)
            action=effective[ai]
        else:
            m=0;action=[0.]*n
            if ready:
                if method=="async_latest":action=[1. if i in ready else 0. for i in range(n)]
                elif method=="async_delay":
                    action=[1/(1+sum(v-s for v,s in zip(net.versions,ready[i]["source_versions"]))) if i in ready else 0. for i in range(n)]
                elif method=="async_norm":
                    i=max(ready,key=lambda j:sum(float((g*g).sum()) for g in ready[j]["gradient"]));action[i]=1.
                elif method=="async_cyclic":
                    for j in range(n):
                        i=(t+j)%n
                        if i in ready:action[i]=1.;break
                else:raise ValueError(method)
            action=tuple(action)
            # Standard baselines commit immediately, before collecting the next bank.
            for i,alpha in enumerate(action):
                if alpha:
                    if alpha in variants[i]:actors[i]=variants[i][alpha]
                    else:actors[i]=adam_variant(actors[i],ready[i]["gradient"],alpha);proposal_steps+=1
            net.commit(action)
        before_bank=digest([v for a in actors for v in a.actor.parameters()]);source_versions=tuple(net.versions)
        source_snapshot=list(actors)
        data=collect(actors,task,ti,seed,t,892,b,h)
        all_training_ids.extend(data["identities"])
        if full:
            samples,score_rows,weight_max=menu_feedback(actors,variants,actions,take(data,m,h),potential,current_bank=True)
            potential_rows+=m*h if potential is not None else 0
            outbound=sum(policy_bytes[i] for i,alpha in enumerate(action) if alpha>0)
            detail=controller.feedback(samples,outbound/full_bytes,(hint_rows+score_rows)/denominator)
            for i,alpha in enumerate(action):
                if alpha:actors[i]=variants[i][alpha]
            net.commit(action)
        else:score_rows=0;outbound=sum(policy_bytes[i] for i,alpha in enumerate(action) if alpha>0);detail={}
        # Captured bank behavior, not the post-commit server, owns this source gradient.
        sources=source_snapshot
        if digest([v for a in sources for v in a.actor.parameters()])!=before_bank:
            raise AssertionError("current-bank gradient source mutated after receipt decision")
        base.add_advantages(data,critic,normalizer,p)
        if fresh:
            scale=float(method.split("_")[1])
            for actor in actors:
                for group in actor.actor_optimizer.param_groups:group["lr"]*=scale
            base.sequential_update(actors,data,list(range(n)),p)
            for actor in actors:
                for group in actor.actor_optimizer.param_groups:group["lr"]/=scale
            action=tuple([1.]*n);outbound=full_bytes;net.commit(action)
            charges["source_actor_steps"]+=n;charges["source_backward_rows"]+=n*b*h
            # HAPPO factor updates evaluate old/new actor probabilities.
            charges["extra_score_policy_rows"]+=2*n*b*h
        else:
            for i,actor in enumerate(sources):
                gradient=source_gradient(actor,data,i)
                delay=plan["delay_early"] if t<frames//2 else plan["delay_late"]
                net.launch(t,i,source_versions,gradient,*delay)
            charges["source_actor_steps"]+=n;charges["source_backward_rows"]+=n*b*h
        base.train_critic(critic,normalizer,data,p)
        history.append(data)
        if len(history)>plan["potential_history_banks"]:history.pop(0)
        trajectory_bytes=sum(x.nbytes for x in data["obs"])+sum(x.nbytes for x in data["actions"])+sum(x.nbytes for x in data["logs"])+data["states"].nbytes+data["rewards"].nbytes
        charges["training_transitions"]+=b*h;charges["sampling_policy_rows"]+=n*b*h
        charges["critic_backward_rows"]+=plan["critic_steps"]*b*h;charges["critic_steps"]+=plan["critic_steps"]
        charges["proposal_adam_steps"]+=proposal_steps;charges["committed_adam_steps"]+=sum(x>0 for x in action)
        charges["extra_score_policy_rows"]+=hint_rows+score_rows;charges["extra_potential_rows"]+=potential_rows
        charges["gradient_bytes"]+=full_bytes;charges["policy_broadcast_bytes"]+=outbound;charges["trajectory_bytes"]+=trajectory_bytes
        charges["advantage_bytes"]+=n*data["advantages"].nbytes
        charges["gradient_messages"]+=n;charges["policy_messages"]+=sum(x>0 for x in action)
        charges["trajectory_messages"]+=n+1;charges["advantage_messages"]+=n
        ledger.append({"frame":t,"decision_before_bank":full,"current_bank_used_only_after_decision":full,
            "source_versions":list(source_versions),"server_versions_after":list(net.versions),
            "source_policy_hash":before_bank,"action":list(action),"ready_packet_ids":[ready[i]["id"] for i in sorted(ready)],
            "arrived_packet_ids":[j["id"] for j in arrived],"arrival_version_ages":[sum(v-s for v,s in zip(source_versions,j["source_versions"])) for j in arrived],
            "probe_episodes":m,"hint_policy_rows":hint_rows,"score_policy_rows":score_rows,"potential_rows":potential_rows,
            "proposal_adam_steps":proposal_steps,"outbound_bytes":outbound,"trajectory_bytes":trajectory_bytes,
            "weight_max":weight_max,"control":detail,"charges":dict(charges)})
        if (t+1)%plan["evaluation_every"]==0 or t+1==frames:evaluate(t+1)
    if not fresh:net.drain()
    if len(all_training_ids)!=len(set(all_training_ids)) or set(all_training_ids)&set(all_eval_ids):raise AssertionError("episode leakage")
    return {"task":task["name"],"seed":seed,"method":method,"frames":frames,"initial_hash":initial,
        "final_hash":digest([v for a in actors for v in a.actor.parameters()]),"actor_parameter_bytes":policy_bytes,"charges":charges,"curves":curves,
        "ledger":ledger,"packet_events":net.events,"packet_transmitted_bytes":net.transmitted_gradient_bytes,
        "training_episode_ids":all_training_ids,"evaluation_episode_ids":all_eval_ids,
        "evaluation_used_by_controller":False,"native_environment":True,"logical_transport_not_measured_network":True,
        "cpu_seconds":time.process_time()-started}
