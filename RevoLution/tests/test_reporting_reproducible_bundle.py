from pathlib import Path

import pytest

from revolution.reporting import generate_report_bundle


def _write_run(run_dir: Path) -> None:
    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text("experiment: {}\n", encoding="utf-8")
    (run_dir / "metadata.json").write_text("{}", encoding="utf-8")

    (raw / "episodes.csv").write_text(
        "generation,genome_id,episode,reward,steps,descriptor,loss,grad_norm\n"
        "0,1,0,1.0,5,\"[0.5,0.5]\",0.0,0.0\n",
        encoding="utf-8",
    )
    (raw / "generations.csv").write_text(
        "generation,best_reward,median_reward,best_novelty,archive_size,num_species,threshold\n"
        "0,1.0,1.0,0.5,1,1,1.0\n",
        encoding="utf-8",
    )
    (raw / "archive.csv").write_text(
        "generation,archive_size,threshold\n0,1,1.0\n", encoding="utf-8"
    )


def test_report_bundle_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("pandas")

    runs_dir = tmp_path / "runs"
    run_one = runs_dir / "run1"
    run_two = runs_dir / "run2"
    _write_run(run_one)
    _write_run(run_two)

    report_dir = tmp_path / "report"
    bundle_one = generate_report_bundle(str(runs_dir), str(report_dir))
    bundle_two = generate_report_bundle(str(runs_dir), str(report_dir))

    report_one = Path(bundle_one.report_path).read_text(encoding="utf-8")
    report_two = Path(bundle_two.report_path).read_text(encoding="utf-8")
    assert report_one == report_two

    checklist_one = (
        Path(bundle_one.report_path).parent / "repro_checklist.md"
    ).read_text(encoding="utf-8")
    checklist_two = (
        Path(bundle_two.report_path).parent / "repro_checklist.md"
    ).read_text(encoding="utf-8")
    assert checklist_one == checklist_two
