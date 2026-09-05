import json

from experiments.dependence_delay_linear.run_t080_background import completed_chunk


def test_completed_chunk_requires_matching_hash_and_manifest(tmp_path):
    chunk = tmp_path / "chunks" / "chunk-00"
    chunk.mkdir(parents=True)
    cells = chunk / "cells.csv"
    cells.write_text("cell_id\na\n", encoding="utf-8")
    from experiments.dependence_delay_linear.run_t080_chunked_continuous_static_execution import sha256
    (chunk / "manifest.json").write_text(json.dumps({
        "chunk_index": 0,
        "cell_count": 36,
        "cells_sha256": sha256(cells),
        "scientific_configuration_unchanged": True,
    }), encoding="utf-8")
    assert completed_chunk(tmp_path, 0, 36)
    cells.write_text("cell_id\nb\n", encoding="utf-8")
    assert not completed_chunk(tmp_path, 0, 36)
