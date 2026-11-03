from pathlib import Path

from src.utils import atomic_write_text


def test_atomic_write_retries_when_file_locked(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    calls = {"n": 0}
    orig = Path.replace

    def flaky(self, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("[WinError 5] Zugriff verweigert")
        return orig(self, dst)
    monkeypatch.setattr(Path, "replace", flaky)
    assert atomic_write_text(target, '{"a": 1}')
    assert target.read_text(encoding="utf-8") == '{"a": 1}' and calls["n"] == 3


def test_atomic_write_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "replace", lambda self, dst: (_ for _ in ()).throw(PermissionError("gesperrt")))
    assert atomic_write_text(tmp_path / "x.json", "{}", attempts=2) is False
