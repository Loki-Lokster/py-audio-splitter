import os


def test_setup_logging_rotation(tmp_path):
    from scripts.logging import setup_logging

    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create 7 old logs; only 4 should remain after cleanup (leaving room for the new one).
    for i in range(7):
        (log_dir / f"audio_splitter_20250101_00000{i}.log").write_text("old\n", encoding="utf-8")

    log_file = setup_logging(log_dir=str(log_dir))
    assert os.path.exists(log_file)

    remaining = sorted([p.name for p in log_dir.iterdir() if p.suffix == ".log"])
    assert len(remaining) == 5

