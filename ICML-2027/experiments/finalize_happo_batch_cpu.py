"""Replay-aware final decision and immutable package manifest for A1."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ICML-2027/results"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize():
    analysis = RESULTS / "happo_async_batch_a1_analysis_20261015"
    replay = RESULTS / "happo_async_batch_a1_analysis_replay_20261015"
    summary = json.loads((analysis / "summary.json").read_text())
    diagnosis_path = RESULTS / "happo_async_batch_a1_diagnosis_20261015.json"
    diagnosis = json.loads(diagnosis_path.read_text())
    analyzer_replay = {file: sha(analysis / file) == sha(replay / file)
                       for file in ("metrics.csv", "summary.json", "integrity.json")}
    if not all(analyzer_replay.values()) or not summary["registered_payload_replay_pass"]:
        raise AssertionError("registered replay or analyzer replay failed")
    if not summary["integrity_pass"] or not all(diagnosis["checks"].values()):
        raise AssertionError("frozen or supplementary integrity failed")
    gates = summary["scientific_gates"]
    decision = {
        "experiment_id": summary["experiment_id"],
        "source_commit": diagnosis["source_commit"],
        "payloads": summary["payloads"],
        "registered_replay_payloads": 6,
        "registered_replay_scientific_files": 18,
        "integrity_pass": summary["integrity_pass"],
        "registered_payload_replay_pass": summary["registered_payload_replay_pass"],
        "analyzer_byte_replay": analyzer_replay,
        "supplementary_integrity": diagnosis["checks"],
        "scientific_gate_pass_count": sum(gates.values()),
        "scientific_gate_count": len(gates),
        "failed_scientific_gates": [name for name, passed in gates.items() if not passed],
        "stage_a_pass": bool(all(gates.values())),
        "stage_b_authorized": bool(summary["stage_b_authorized"] and all(analyzer_replay.values())),
        "gpu_authorized_by_this_result": False,
        "metrics": summary["metrics"],
        "decision": "STOP_TESTED_CANDIDATE" if not all(gates.values()) else "STAGE_B_QUALIFICATION_ONLY",
        "causal_findings": [
            "Independent search-transfer diagnostic does not reproduce the positive selected search margins.",
            "Search consumes 46.17x and 11.58x the respective training transition allowances.",
            "Reference stale-cyclic delay schedule aliases with the cyclic launch clock and uses zero stale packets.",
            "Both baseline-learning gates pass, so a completely absent learning signal is not the explanation."
        ],
        "implemented_correction": "Unique-job causal transport, separate service streams, one-use raw-gradient packets, server-owned optimizer state, and sunk-cost accounting.",
        "remaining_entry_conditions": ["causal observable score", "resource-feasible paired precision",
                                        "nonvacuous uniform return certificate", "multi-step launch/receipt drift interface"],
        "interpretation": "This finite noisy menu does not prove that all asynchronous policy-allocation headroom is absent. No successful successor algorithm or ICML readiness is claimed."
    }
    out = RESULTS / "happo_async_batch_a1_decision_20261015"
    if out.exists():
        raise FileExistsError("fresh final decision directory required")
    out.mkdir()
    (out / "decision.json").write_text(json.dumps(decision, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    paths = []
    for folder in ("happo_async_batch_a1_scientific_20261015", "happo_async_batch_a1_registered_replay_20261015",
                   "happo_async_batch_a1_analysis_20261015", "happo_async_batch_a1_analysis_replay_20261015",
                   "happo_async_batch_a1_figures_20261015"):
        paths.extend(path for path in (RESULTS / folder).rglob("*") if path.is_file())
    paths.extend([diagnosis_path, out / "decision.json"])
    paths.extend(ROOT / name for name in (
        "ICML-2027/internal/happo_batch_cpu_a1_decision.md",
        "ICML-2027/internal/causal_packet_redesign_after_a1.md",
        "ICML-2027/internal/happo_batch_cpu_source_lock_a1.json",
        "ICML-2027/experiments/happo_batch_cpu.py", "ICML-2027/experiments/analyze_happo_batch_cpu.py",
        "ICML-2027/experiments/happo_batch_cpu_preregistration.json",
        "ICML-2027/experiments/causal_packet_ledger.py", "ICML-2027/experiments/diagnose_happo_batch_cpu.py",
        "ICML-2027/experiments/plot_happo_batch_cpu.py", "ICML-2027/experiments/finalize_happo_batch_cpu.py",
        "ICML-2027/tests/test_causal_packet_ledger.py"))
    manifest = {str(path.relative_to(ROOT)).replace("\\", "/"): {"sha256": sha(path), "bytes": path.stat().st_size}
                for path in sorted(paths)}
    (out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision["decision"], "gates_passed": decision["scientific_gate_pass_count"],
                      "gates_total": decision["scientific_gate_count"], "manifest_files": len(manifest)}, indent=2))


if __name__ == "__main__":
    finalize()
