from scripts import utils


def test_create_windows_shortcut_writes_utf8_batch(tmp_path, monkeypatch):
    home = tmp_path / "home"
    desktop = home / "Desktop"
    desktop.mkdir(parents=True)

    monkeypatch.setattr(utils.os.path, "expanduser", lambda _: str(home))
    monkeypatch.setattr(utils.sys, "executable", r"C:\Python\python.exe")

    assert utils.create_windows_shortcut(r"C:\repo\audio_split.py") is True

    bat = desktop / "Audio Splitter.bat"
    content = bat.read_text(encoding="utf-8")
    assert "chcp 65001" in content
    assert "PYTHONIOENCODING=utf-8" in content

