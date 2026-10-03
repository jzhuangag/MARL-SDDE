import json

import pytest

from experiments.reproducible_artifacts import (
    write_execution_metadata,
    write_scientific_summary,
)


def test_scientific_summary_is_byte_stable(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    payload = {"metric": 0.75, "gates": {"A": True}}
    write_scientific_summary(first, payload)
    write_scientific_summary(second, payload)
    assert first.read_bytes() == second.read_bytes()


def test_scientific_summary_rejects_timing_fields(tmp_path):
    with pytest.raises(ValueError):
        write_scientific_summary(tmp_path / "summary.json", {"runtime_seconds": 1.0})


def test_execution_metadata_is_separate(tmp_path):
    path = tmp_path / "execution.json"
    write_execution_metadata(path, runtime_seconds=1.25, workers=8)
    assert json.loads(path.read_text()) == {"runtime_seconds": 1.25, "workers": 8}
