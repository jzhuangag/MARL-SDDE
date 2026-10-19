"""Deterministic non-scientific toy fixtures; never generate formal trajectories."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name] = "1"
import copy
import inspect
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.polynomial.hermite import hermgauss
from core import (Action, Family, LEARNERS, POLICIES, weights, direct_learner,
                  program_risk, logdet_prefix, posterior_variance, probe_plan,
                  likelihood, deployment_decision, group_rows, scheduled, families,
                  InstanceRisk, risk_upper, admit)
from analysis import audit_row, bootstrap, write_csv, load_rows, _write_tables
from guard import HERE, RESULT, Budget, SingleOwner, dump, predecessor_ready, sha, verify_sources
from prepare import numeric_overlap, structured_overlap, seed_registry_audit


class ToyTests(unittest.TestCase):
    pass


def case(name,fn):
    def test(self):
        fn(self)
    setattr(ToyTests,"test_"+name,test)


for learner in LEARNERS:
    for D in (0,1,3,8):
        for M in (0,1,3,8,9,13,24,49):
            def identity(self,learner=learner,D=D,M=M):
                values = np.linspace(-.71,.93,M)
                direct,N = direct_learner(values,learner,D)
                d,w = weights(learner,M,D)
                self.assertEqual(N,max(0,M-D))
                self.assertAlmostEqual(d+float(w@values[:N]),direct,places=12)
            case(f"direct_weights_{learner}_{D}_{M}",identity)

for learner in LEARNERS:
    for D in (0,2,7):
        for action in (Action(1,1),Action(2,3),Action(4,2)):
            for scale in (3,19):
                def dense(self,learner=learner,D=D,action=action,scale=scale):
                    f = Family(4,.7,.31,D,theta_low=.13,h=5)
                    bm,be = 9*scale,3*scale
                    J,d,k = program_risk(learner,action,bm,be,f,.7)
                    M = scheduled(action,bm,be,f.h)
                    _,w = weights(learner,M,D)
                    ix = np.arange(len(w))
                    cov = .7*(f.lam**action.b)**np.abs(ix[:,None]-ix[None,:])+np.eye(len(w))/action.q
                    self.assertAlmostEqual(J,d*d+float(w@cov@w),places=12)
                    self.assertAlmostEqual(k,float(w@(f.lam**(action.b*(ix+1)))),places=12)
                case(f"dense_risk_{learner}_{D}_{action.q}_{action.b}_{scale}",dense)

for theta in (.13,.7,1.3):
    for lam in (.31,.67):
        for n in (1,3,11):
            def gaussian(self,theta=theta,lam=lam,n=n):
                q = 4
                ix = np.arange(n)
                R = lam**np.abs(ix[:,None]-ix[None,:])
                cov = np.eye(n)+q*theta*R
                y = np.linspace(-.63,.91,n)
                sign,ld = np.linalg.slogdet(cov)
                self.assertEqual(sign,1)
                self.assertAlmostEqual(logdet_prefix(theta,q,lam,n),ld,places=12)
                expected = -.5*(n*math.log(2*math.pi)+ld+float(y@np.linalg.solve(cov,y)))
                self.assertAlmostEqual(likelihood(y,theta,q,lam),expected,places=12)
                cross = math.sqrt(q)*theta*lam**(n-1-ix)
                P = theta-float(cross@np.linalg.solve(cov,cross))
                self.assertAlmostEqual(posterior_variance(theta,lam,q,n),P,places=12)
            case(f"gaussian_dense_{theta}_{lam}_{n}",gaussian)

for lam in (.31,.67):
    for high in (.7,1.3):
        def affinity(self,lam=lam,high=high):
            f = Family(4,high,lam,2,theta_low=.13,h=5)
            n,a = probe_plan(f)
            self.assertGreater(n,0)
            def exact(m):
                R = lam**np.abs(np.arange(m)[:,None]-np.arange(m)[None,:])
                l = [np.linalg.slogdet(np.eye(m)+4*t*R)[1] for t in (.13,high,(.13+high)/2)]
                return math.exp(.25*(l[0]+l[1])-.5*l[2])
            self.assertAlmostEqual(a,exact(n),places=12)
            self.assertLessEqual(a,.025)
            if n>1:
                self.assertGreater(exact(n-1),.025)
        case(f"minimal_affinity_{lam}_{high}",affinity)


def gamma_moment(self):
    item = InstanceRisk(.4,.5,.3,.6,.2,.7,.17,.09,.9)
    _,carry = risk_upper(item)
    x,w = hermgauss(12)
    z = np.sqrt(2*.17)*x
    A = 2*(item.wrong_d*item.wrong_k-item.correct_d*item.correct_k)
    B = item.wrong_k**2-item.correct_k**2
    polynomial = A*z+B*(z*z-.17)
    gamma2 = float(w@(polynomial*polynomial))/math.sqrt(math.pi)
    self.assertAlmostEqual(carry**2,item.directional_error_upper*gamma2,places=14)
case("gaussian_gamma_exact_quadrature",gamma_moment)


def gate_boundaries(self):
    low = InstanceRisk(.4,.4,0,0,0,0,0,0,.3)
    high = InstanceRisk(.2,.2,0,0,0,0,0,0,.3)
    self.assertTrue(admit(low,high,.10000000000000003,True))
    self.assertFalse(admit(low,high,.0999,True))
    self.assertFalse(admit(low,high,.2,False))
    self.assertFalse(admit(high,high,0,False))
    self.assertEqual(risk_upper(high),(.2,0.))
    same = InstanceRisk(.3,.3,0,0,0,0,0,0,.3)
    self.assertFalse(admit(same,same,0,True))
    for delta in (-.01,1.01,float("nan")):
        bad = InstanceRisk(.1,.2,0,0,0,0,0,delta,.3)
        with self.assertRaises(ValueError):
            risk_upper(bad)
case("gate_boundaries_and_invalid_certificate",gate_boundaries)


for policy in POLICIES[:-1]:
    for decision in ("low","high"):
        def decision_role(self,policy=policy,decision=decision):
            public = dict(feasible=True,corrected=True,no_carryover=True,
                          fallback=Action(4,1),low=Action(1,2),high=Action(2,3))
            used,a = deployment_decision(policy,public,decision)
            expected = Action(2,1) if policy=="fixed_q2_b1" else (public[decision] if policy!="always_all" else public["fallback"])
            self.assertEqual(a,expected)
            self.assertEqual(used,policy not in ("fixed_q2_b1","always_all"))
        case(f"deployment_{policy}_{decision}",decision_role)


def controller_signature(self):
    self.assertEqual(tuple(inspect.signature(deployment_decision).parameters),
                     ("policy","public_design","identified"))
    self.assertNotIn("regime",inspect.signature(deployment_decision).parameters)
    with self.assertRaises(ValueError):
        deployment_decision("information_only",dict(feasible=True,fallback=Action(4,1)),None)
case("controller_no_hidden_truth_argument",controller_signature)


def toy_rows(D=2):
    f = Family(4,.7,.31,D,theta_low=.05)
    # Entirely deterministic potential fixture, not a random scientific stream.
    length = 4*97+1
    common = np.sin(np.arange(length)*.19)*.2
    private = np.cumsum(np.cos(np.arange(4*length).reshape(4,length)*.11)*.3,axis=0)
    with patch("core.potential_observations",return_value=(common,private,"TOY-NOT-A-SEED")) as generator:
        rows = list(group_rows(-17,f,"low",LEARNERS[1],[9,97]))
        generator.assert_called_once()
    return rows


def toy_pipeline(self):
    a,b = toy_rows(),toy_rows()
    self.assertEqual(json.dumps(a,sort_keys=True),json.dumps(b,sort_keys=True))
    self.assertEqual(len(a),12)
    for row in a:
        audit_row(row)
    with tempfile.TemporaryDirectory(prefix="v3-toy-") as tmp:
        p = Path(tmp)
        with (p/"seed_-17.jsonl").open("x",encoding="utf-8") as f:
            for row in a:
                f.write(json.dumps(row)+"\n")
        frame = load_rows(p)
        for row in frame.to_dict("records"):
            audit_row(row)
        write_csv(p/"a.csv",frame)
        write_csv(p/"b.csv",frame)
        self.assertEqual(sha(p/"a.csv"),sha(p/"b.csv"))
        for field in ("messages_used","environment_used","usable","q","family"):
            row = copy.deepcopy(a[0])
            row[field] = "WRONG" if field=="family" else row[field]+1
            with self.assertRaises((AssertionError,ValueError)):
                audit_row(row)
case("toy_stream_replay_csv_and_corrupt_ledgers",toy_pipeline)


def all_analysis_tables(self):
    import pandas as pd
    rows = []
    for seed,multiplier in ((-17,.7),(-18,1.),(-19,1.3)):
        for saved in toy_rows():
            row = dict(saved,seed=seed)
            row["terminal_estimate"] *= multiplier
            row["terminal_squared_risk"] = row["terminal_estimate"]**2
            rows.append(row)
    frame = pd.DataFrame(rows)
    plan = dict(analysis=dict(analysis_rng_seed=71))
    with tempfile.TemporaryDirectory(prefix="v3-table-toy-") as tmp:
        a,b = Path(tmp)/"a",Path(tmp)/"b"
        a.mkdir()
        b.mkdir()
        _write_tables(frame.copy(),a,plan,{"toy":"not-scientific"})
        _write_tables(frame.copy(),b,plan,{"toy":"not-scientific"})
        names = {p.name for p in a.iterdir()}
        self.assertEqual(len(names),7)
        for name in names:
            self.assertEqual(sha(a/name),sha(b/name),name)
        summary = json.loads((a/"summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["original_policy_trajectories"],36)
        self.assertEqual(summary["cell_count"],12)
        self.assertFalse(summary["manuscript_admission"])
case("all_seven_analysis_outputs_byte_identical_toy",all_analysis_tables)


def bootstrap_repeat(self):
    values = np.array([-.7,.2,.9,-.3,.4])
    a = bootstrap(values,np.random.RandomState(71))
    b = bootstrap(values,np.random.RandomState(71))
    self.assertEqual(a,b)
    self.assertLess(a[0],0)
    self.assertGreater(a[1],0)
case("paired_bootstrap_repeat_and_negative_values",bootstrap_repeat)


def integer_ledger(self):
    self.assertEqual(scheduled(Action(4,3),31,8,5),2)
    self.assertEqual(scheduled(Action(4,3),0,8,5),0)
    with self.assertRaises(ValueError):
        scheduled(Action(4,3),-1,8,5)
    estimate,n = direct_learner(np.array([2.,3.]),LEARNERS[1],3)
    self.assertEqual((estimate,n),(1.,0))
case("integer_horizon_and_paid_withheld_updates",integer_ledger)


def registration_rejection(self):
    with tempfile.TemporaryDirectory(prefix="v3-lock-toy-") as tmp:
        p = Path(tmp)/"input"
        p.write_text("deterministic toy",encoding="utf-8")
        with patch("guard.ROOT",Path(tmp)):
            verify_sources(dict(source_sha256={"input":sha(p)}))
            with self.assertRaises(AssertionError):
                verify_sources(dict(source_sha256={"input":"0"*64}))
    with self.assertRaises(TimeoutError):
        Budget(28800)()
case("source_drift_and_cumulative_budget_rejected",registration_rejection)


def predecessor_fixtures(self):
    with tempfile.TemporaryDirectory(prefix="v3-audit-toy-") as tmp:
        p = Path(tmp)
        root = p/"old"
        root.mkdir()
        receipt = p/"receipt.json"
        self.assertFalse(predecessor_ready(receipt,root,lambda pid:False,p)[0])
        dump(root/"workflow_complete.json",dict(toy=True))
        dump(p/"audit.json",dict(toy=True))
        base = dict(audit_complete=True,all_preserved_evidence_reviewed=True,
                    terminal_decision_recorded=True,terminal_decision="TOY_ONLY",
                    native_root=str(root),driver_pid=123,terminal_marker="workflow_complete.json",
                    terminal_marker_sha256=sha(root/"workflow_complete.json"),
                    audit_report="audit.json",audit_report_sha256=sha(p/"audit.json"))
        for change,alive,expected in (({},False,True),({},True,False),
                                     ({"audit_complete":False},False,False),
                                     ({"terminal_decision":""},False,False),
                                     ({"terminal_marker_sha256":"bad"},False,False),
                                     ({"audit_report_sha256":"bad"},False,False),
                                     ({"native_root":str(p/"wrong")},False,False),
                                     ({"audit_report":"../outside.json"},False,False)):
            receipt.write_text(json.dumps(dict(base,**change)),encoding="utf-8")
            self.assertEqual(predecessor_ready(receipt,root,lambda pid:alive,p)[0],expected)
case("exit_AND_independent_audit_prerequisites",predecessor_fixtures)


def shared_mutex(self):
    # Same-thread Windows mutexes are reentrant; a separate toy thread must fail.
    import threading
    failures = []
    name = "Local\\SDDEV3ToyMutex"+str(os.getpid())
    def competing():
        try:
            with SingleOwner(name):
                failures.append("unexpected acquisition")
        except RuntimeError:
            failures.append("rejected")
    with SingleOwner(name):
        thread = threading.Thread(target=competing)
        thread.start()
        thread.join(5)
        self.assertFalse(thread.is_alive())
    self.assertEqual(failures,["rejected"])
case("duplicate_owner_rejected_by_mutex",shared_mutex)


def formal_mapping_only(self):
    p = json.loads((HERE/"protocol.json").read_text(encoding="utf-8"))
    self.assertEqual(len(families(p)),16)
    self.assertEqual(len(set(f.key for f in families(p))),16)
    self.assertEqual(16*4*2*2*6*128,p["formal_policy_trajectories"])
    self.assertEqual(tuple(p["policies"]),POLICIES)
    self.assertEqual(p["formal_seeds"]["end"]-p["formal_seeds"]["start"]+1,128)
    self.assertFalse(RESULT.exists(),"Formal data must not exist during preparation")
case("approved_cardinality_without_formal_simulation",formal_mapping_only)


def historical_seeds(self):
    targets = {20571001,20571002}
    self.assertEqual(numeric_overlap("old 20571000--20571003",targets),sorted(targets))
    self.assertEqual(structured_overlap(dict(formal_seed_start=20570999,formal_seed_count=5),targets),targets)
    self.assertEqual(numeric_overlap("20_571_001",targets),[20571001])
    protocol = json.loads((HERE/"protocol.json").read_text(encoding="utf-8"))
    audit = seed_registry_audit(protocol)
    self.assertEqual(audit["status"],"PASS")
    self.assertGreaterEqual(audit["old_files_checked"],6)
case("historical_seed_nonoverlap_and_range_parser",historical_seeds)


def explicit_encoding(self):
    import ast
    for path in HERE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="read_text":
                self.assertTrue(any(k.arg=="encoding" for k in node.keywords),str(path))
case("explicit_utf8_under_windows_locale",explicit_encoding)


if __name__=="__main__":
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ToyTests))
    print(json.dumps(dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                          successful=result.wasSuccessful(),non_scientific_toy_only=True,
                          formal_trajectories_generated=0,log=log.getvalue()),sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
