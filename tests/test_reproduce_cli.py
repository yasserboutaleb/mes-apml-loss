"""Fast offline tests for the single-command reproducibility entrypoint."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reproduce.py"
spec = importlib.util.spec_from_file_location("mes_reproduce", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_check_and_figures_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "invoke", lambda *a: calls.append(a))
    assert mod.main(["check"]) == 0
    assert calls[-1] == ("-m", "pytest", "-q")
    assert mod.main(["figures"]) == 0
    assert calls[-1] == ("scripts/make_figures.py", "--out", "figures")


def test_shapes_preset_never_loads_mnist(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(mod, "invoke", lambda *a: calls.append(a))
    target = tmp_path / "fresh"
    assert mod.main(["smoke", "--out", str(target)]) == 0
    cmd = calls[-1]
    assert "--dataset" in cmd and cmd[cmd.index("--dataset") + 1] == "shapes"
    assert "--no-preview" in cmd
    assert "--video" not in cmd


def test_mnist_preset_documents_real_data_and_video(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(mod, "invoke", lambda *a: calls.append(a))
    assert mod.main(["mnist", "--out", str(tmp_path / "fresh"),
                     "--seed", "44", "--video"]) == 0
    cmd = calls[-1]
    assert cmd[cmd.index("--dataset")+1] == "mnist"
    assert cmd[cmd.index("--digits")+1] == "1,2,3,4"
    assert cmd[cmd.index("--seed")+1] == "44"
    assert "--video" in cmd


def test_refuses_overwriting_any_existing_output(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(mod, "invoke", lambda *a: calls.append(a))
    with pytest.raises(SystemExit):
        mod.main(["smoke", "--out", str(tmp_path)])
    assert calls == []
