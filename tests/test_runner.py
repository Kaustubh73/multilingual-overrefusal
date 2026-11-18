import pytest

from multilingual_overrefusal import runner


def write(tmp_path, body):
    p = tmp_path / "c.toml"
    p.write_text(body)
    return p


def test_rejects_unknown_step(tmp_path):
    p = write(tmp_path, f'name="t"\noutput_dir="{tmp_path}/o"\nsteps=["nope"]\n')
    with pytest.raises(ValueError, match="unknown step"):
        runner.load_config(p)


def test_rejects_missing_key(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        runner.load_config(write(tmp_path, 'name="t"\n'))


def test_skips_finished_steps(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(runner.STEPS, "demo", lambda cfg: calls.append(1))
    p = write(tmp_path, f'name="t"\noutput_dir="{tmp_path}/o"\nsteps=["demo"]\n')
    cfg = runner.load_config(p)
    runner.run(cfg)
    runner.run(cfg)                       # second run skips
    assert len(calls) == 1
    runner.run(cfg, force=["demo"])       # forced rerun
    assert len(calls) == 2


def test_overrides():
    cfg = runner.apply_overrides({"generate": {}}, ["generate.backend=modal", "seed=3"])
    assert cfg["generate"]["backend"] == "modal" and cfg["seed"] == 3


def test_rerunning_a_step_clears_later_markers(tmp_path, monkeypatch):
    """Forcing an earlier step must make later steps run again, not skip with stale outputs."""
    calls = []
    monkeypatch.setitem(runner.STEPS, "a", lambda cfg: calls.append("a"))
    monkeypatch.setitem(runner.STEPS, "b", lambda cfg: calls.append("b"))
    cfg = {"name": "t", "output_dir": str(tmp_path / "o"), "steps": ["a", "b"]}
    runner.run(cfg)
    calls.clear()
    runner.run(cfg, force=["a"])
    assert calls == ["a", "b"]
    calls.clear()
    runner.run(cfg, only="a")                 # --only a also invalidates b for the next full run
    runner.run(cfg)
    assert calls == ["a", "b"]
