"""Regression checks tying the TSP manuscript to frozen evidence."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
MAIN = (ROOT / "main.tex").read_text(encoding="utf-8")
APPENDICES = (ROOT / "appendices.tex").read_text(encoding="utf-8")
SUPPLEMENTARY = (ROOT / "supplementary_resource_geometry.tex").read_text(encoding="utf-8")
BIB = (ROOT / "references.bib").read_text(encoding="utf-8")


def test_formal_numbers_match_core_results() -> None:
    path = REPO / "experiments" / "dependence_delay_linear" / "results"
    path = path / "exp016b_formal_20260801" / "analysis" / "core_results.json"
    core = json.loads(path.read_text(encoding="utf-8"))
    assert core["rows"] == 2_752_512
    assert core["seeds"] == 192
    assert f'{100 * core["primary_layer_A"]["relative_difference"]:.2f}' in MAIN
    assert f'{100 * core["layer_B"]["relative_difference"]:.2f}' in MAIN
    assert f'{core["primary_layer_A"]["simultaneous_one_sided_lower"]:.4f}' in MAIN
    assert f'{core["layer_B"]["simultaneous_one_sided_lower"]:.4f}' in MAIN
    assert core["scenario_level_denominator"] == 96
    assert round(core["scenario_level_coverage"] * 96) == 77
    assert core["safety_certificate"]["all_pass"] is True
    assert all(core["gate_results_P1_P11"].values())


def test_prior_studies_match_frozen_validation_reports() -> None:
    exp010b = (REPO / "docs" / "validation_exp010b.md").read_text(encoding="utf-8")
    exp007a = (REPO / "docs" / "experiment_007a_linear_td_correlation.md").read_text(encoding="utf-8")
    for token in ("1,152", "12/12", "18.49", "12.67", "20.77", "228.98", "0.305"):
        assert token in exp010b
        assert token in MAIN
    for token in ("30.996", "1.111", "134,784", "q=16", "q=1"):
        assert token in exp007a
        assert token in MAIN


def test_convergence_confirmation_matches_validation_record() -> None:
    validation_path = ROOT / "internal" / "convergence_curve_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    data_path = ROOT / "data" / "convergence_curve_summary.csv"
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert validation["experiment_id"] == "TSP-CURVE-001"
    assert validation["confirmation_seeds"] == 64
    assert validation["raw_rows"] == 157_440
    assert validation["selected_strong_fixed_q"] == 4
    assert validation["cells_improved"] == 9
    assert validation["cells_tied"] == 3
    assert validation["finite"] is True
    assert validation["within_budget"] is True
    assert digest == validation["aggregate_csv_sha256"]
    for token in ("157,440", "0.9009", "0.8868", "9.91", "11.32", "26.48", "21.67"):
        assert token in MAIN
    assert (ROOT / "figures" / "convergence_curves.pdf").is_file()


def test_mappo_confirmation_matches_frozen_gate() -> None:
    gate_path = ROOT / "internal" / "marl_probe_commit_conf1_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    figure_path = ROOT / "figures" / "marl_probe_commit_return_curves.pdf"
    compact_figure_path = (
        ROOT / "figures" / "marl_probe_commit_return_curves_compact.pdf"
    )
    summary_path = ROOT / "internal" / "marl_probe_commit_conf1_curve_summary.csv"
    assert gate["experiment_id"] == "TSP-MARL-CONF-001"
    assert gate["run_count"] == 48
    assert gate["expected_run_count"] == 48
    assert gate["all_mandatory_gates_pass"] is True
    assert all(gate["gates"].values())
    assert gate["independent_selected_q8_fraction"] == 1.0
    assert gate["shared_selected_q1_fraction"] == 1.0
    assert f'{100 * gate["shared_controller_vs_q8_mean"]:.3f}' in MAIN
    assert f'{100 * gate["mixture_controller_vs_q8_mean"]:.3f}' in MAIN
    assert f'{100 * gate["mixture_controller_vs_q8_lower"]:.3f}' in MAIN
    assert hashlib.sha256(figure_path.read_bytes()).hexdigest() == "ea40c05bbb4bdba346c4fb964792ce7c061232190948a5a04cfc8cf44ad1b748"
    assert compact_figure_path.is_file()
    assert "marl_probe_commit_return_curves_compact.pdf" in MAIN
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == "6e2c5d1af24952f72d83ee6347b06f5322ac5dd55bd2fcbf1df22ca8d39d3c64"


def test_every_citation_resolves_and_every_bib_entry_is_cited() -> None:
    cite_keys = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", MAIN):
        cite_keys.update(key.strip() for key in group.split(","))
    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", BIB, flags=re.MULTILINE))
    assert len(bib_keys) >= 30
    assert cite_keys == bib_keys


def test_source_hygiene_and_front_matter() -> None:
    for source in (MAIN, APPENDICES, SUPPLEMENTARY):
        assert not any(ord(character) < 32 and character not in "\t\n\r" for character in source)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", MAIN, re.DOTALL)
    assert abstract is not None
    words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", abstract.group(1))
    assert 150 <= len(words) <= 220
    keywords = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", MAIN, re.DOTALL)
    assert keywords is not None
    assert len([item for item in keywords.group(1).split(",") if item.strip()]) == 5
    assert "\\qquad" not in MAIN + APPENDICES + SUPPLEMENTARY


def test_labels_and_references_are_closed() -> None:
    source = MAIN + "\n" + APPENDICES
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    refs = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", source)
    assert len(labels) == len(set(labels))
    assert set(refs) <= set(labels)


def test_compiled_manuscript_is_full_tsp_length() -> None:
    pdf = ROOT / "main.pdf"
    supplement = ROOT / "supplementary.pdf"
    assert pdf.is_file()
    assert supplement.is_file()
    assert len(PdfReader(str(pdf)).pages) == 13
    assert len(PdfReader(str(supplement)).pages) == 1
