"""Known-model certificate construction checks; no learning trajectories."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
HERE = Path(__file__).resolve().parents[1]
OLD = HERE.parents[1]/"experiments/dependence_delay_linear"
sys.path.insert(0, str(OLD))
from multistate_certificate import build_transfer_mrp, exact_pair_tv, certificate_constants, aggregate_td_noise
from affine_markov_certificate import affine_bound_components, innovation_bound


def run():
    start = time.process_time()
    out = HERE/"internal/content_strengthening_20261019/finite_certificate.json"
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for persistence in (0.0, .9, .98):
        model = build_transfer_mrp(persistence)
        pi, M = model["pair_weights"], model["pair_transition"]
        assert np.allclose(pi@M, pi, atol=1e-14)
        assert np.allclose(np.einsum("m,mi->i", pi, model["td_noise"]), 0, atol=1e-14)
        for q in (1, 2, 4, 8, 16, 32):
            base = certificate_constants(model, q, 0, [0]*q, 0)
            target = base["monotonicity"]/(8*base["lipschitz"])
            # TV contracts under a Markov kernel: binary search is valid here.
            left, right = 0, 1
            while min(1, (q+1)*exact_pair_tv(model, right)) > target:
                right *= 2
                assert right <= 65536
            while right-left > 1:
                middle = (left+right)//2
                if min(1, (q+1)*exact_pair_tv(model, middle)) <= target:
                    right = middle
                else:
                    left = middle
            delta = min(1, (q+1)*exact_pair_tv(model, right))
            for rho in (0.0, .9):
                for D in (0, 8):
                    constants = certificate_constants(model, q, rho, [D]*q, delta)
                    omega, G = aggregate_td_noise(model, q, rho), innovation_bound(model)
                    eta = constants["effective_monotonicity"]/(4*(2*constants["curvature"]+4*constants["lipschitz"]**2*delta+1))
                    for halving in range(100):
                        values = affine_bound_components(constants, omega, G, delta, eta)
                        if values["a_delta"] > 0 and values["contraction"] < 1:
                            break
                        eta /= 2
                    else:
                        raise AssertionError("No certified witness")
                    assert np.isfinite(list(values.values())).all()
                    rows.append(dict(persistence=persistence, q=q, rho=rho, D=D,
                                     physical_pair_start_spacing=right, joint_tv_upper=delta,
                                     mu=constants["monotonicity"], L=constants["lipschitz"], G=G,
                                     K=constants["curvature"], Omega=omega, eta=eta,
                                     halving_steps=halving, c=values["contraction"], d=values["forcing"]))
    assert len(rows) == 72
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [OLD/"multistate_certificate.py", OLD/"affine_markov_certificate.py", Path(__file__)]}
    report = dict(status="KNOWN_MODEL_CONSTRUCTIVE_INPUT_AND_SHIELD_CHECKS_PASS",
                  rows=rows, count=len(rows), source_sha256=hashes,
                  new_trajectories=0, new_random_draws=0, cpu_seconds=time.process_time()-start,
                  limitations=["known source kernels and dependence parameters",
                               "certified witnesses, not empirically optimized actions",
                               "no finite-sample unknown-model coverage claim",
                               "offline calibration cost is not O(qd)"])
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    run()
