"""Independent replay, exact resource recurrence, and oracle metadata erratum."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from learning_value import calibration_cost
from run_cpu import CONFIG, dump, sha256


def audit(source:Path,replay:Path,out:Path) -> None:
    if out.exists():
        raise FileExistsError("refusing to overwrite an audit")
    hashes={}
    for name in ("endpoints.csv","certificates.csv","summary.json"):
        hashes[name]=sha256(source/name)
        if hashes[name]!=sha256(replay/name):
            raise AssertionError("scientific replay mismatch: "+name)
    frame=pd.read_csv(source/"endpoints.csv")
    summary=json.loads((source/"summary.json").read_text())
    base=json.loads(CONFIG.read_text())
    frame["derived_probe_messages"]=frame.messages-frame.updates*(frame.overhead+frame.q)
    frame["derived_probe_actor_transitions"]=frame.actor_transitions-frame.updates*frame.q*frame.spacing
    expected_m=(64+1)*(frame.overhead+8)
    expected_e=(64+1)*8
    free=frame.derived_probe_messages.eq(0)&frame.derived_probe_actor_transitions.eq(0)
    paid=frame.derived_probe_messages.eq(expected_m)&frame.derived_probe_actor_transitions.eq(expected_e)
    if not (free|paid).all():
        raise AssertionError("invalid resource recurrence")
    n_expected=np.minimum((base["message_budget"]-frame.derived_probe_messages)//(frame.overhead+frame.q),
                          (frame.actor_budget-frame.derived_probe_actor_transitions)//(frame.q*frame.spacing))
    if not frame.updates.eq(n_expected).all():
        raise AssertionError("wrong usable horizon")
    mismatch=frame.probe_messages.ne(frame.derived_probe_messages)|frame.probe_actor_transitions.ne(frame.derived_probe_actor_transitions)
    if mismatch.any() and not frame.loc[mismatch,"method"].eq("admissible_oracle").all():
        raise AssertionError("executed-controller prefix ledger mismatch")
    # Diagnostic oracle totals and usable horizons already include acquisition.
    # Preserve raw data; provide corrected prefix columns only in this sidecar.
    out.mkdir(parents=True)
    ledger=frame[["seed","cell","method","updates","messages","actor_transitions",
                  "probe_messages","probe_actor_transitions","derived_probe_messages",
                  "derived_probe_actor_transitions"]]
    ledger.to_csv(out/"derived_accounting_ledger.csv",index=False,lineterminator="\n")
    report={"scientific_replay_byte_identical":True,"scientific_hashes":hashes,
            "source_all_frozen_gates_pass":summary["all_gates_without_replay_pass"],
            "all_actual_totals_and_horizons_exact":True,"executed_controller_prefixes_exact":True,
            "original_diagnostic_oracle_prefix_metadata_mismatches":int(mismatch.sum()),
            "oracle_metadata_erratum":"For paid admissible-oracle rows the raw prefix fields were zero although total costs and usable horizons correctly included acquisition. The sidecar reconstructs these labels arithmetically; risks, decisions, total costs, original artifacts, seeds, and all gate thresholds are unchanged.",
            "rows":len(frame),"scope":summary["scope"]}
    dump(out/"audit.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--replay",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    audit(args.source,args.replay,args.out)
