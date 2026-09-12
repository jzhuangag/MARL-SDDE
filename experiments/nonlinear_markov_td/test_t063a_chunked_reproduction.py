"""Static audit for the byte-reproducible chunked reproduction wrapper."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "experiments/nonlinear_markov_td/run_t063a_chunked_reproduction.py").read_text(encoding="utf-8")


def test_chunked_wrapper_uses_frozen_worker_and_analyzer() -> None:
    assert "run_game_seed" in SOURCE
    assert "from experiments.nonlinear_markov_td.run_t061a_reward_free_controller_pilot import" in SOURCE
    assert "analyze(config, endpoints)" in SOURCE


def test_chunked_wrapper_restarts_pool_and_preserves_job_order() -> None:
    assert "for start in range(0, len(jobs), jobs_per_chunk)" in SOURCE
    assert "executor.map(run_game_seed, batch, chunksize=1)" in SOURCE
    assert "endpoints.extend(row for chunk in chunks for row in chunk)" in SOURCE


def test_chunked_wrapper_refuses_overwrite_and_writes_frozen_artifacts() -> None:
    assert "refusing to overwrite" in SOURCE
    assert '"endpoints.csv"' in SOURCE
    assert '"cells.csv"' in SOURCE
    assert '"summary.json"' in SOURCE
